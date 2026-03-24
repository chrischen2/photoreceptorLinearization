"""Model inversion for stimulus estimation from photocurrent.

Port of EstimateStmFromPhotocurrent.m
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import gaussian


@dataclass
class InversionResult:
    """Results from model inversion."""

    estimate: np.ndarray  # Final estimated stimulus (smoothed/corrected)
    raw_estimate: np.ndarray  # Raw estimate before post-processing
    residual: np.ndarray  # Residual error (original - estimate)
    correlation: float  # Correlation coefficient
    r_squared: float  # R-squared after post-processing
    raw_r_squared: float  # R-squared before post-processing


class InvertedModel:
    """Inverts the biophysical model to estimate stimulus from target photocurrent.

    Args:
        params: Photoreceptor parameters (from init_params).
    """

    def __init__(self, params):
        self.params = params

    def estimate_stimulus(
        self,
        target_response: np.ndarray,
        original_stimulus: np.ndarray,
        time_step: float,
        smooth_pts: int = 100,
        match_power: bool = False,
        match_mean: bool = False,
    ) -> InversionResult:
        """Estimate the stimulus that would produce the target photocurrent.

        Args:
            target_response: Target photocurrent response array.
            original_stimulus: Original stimulus for validation/correction.
            time_step: Simulation time step in seconds.
            smooth_pts: Number of points for Gaussian smoothing window.
            match_power: If True, match power spectrum of estimate to original.
            match_mean: If True, match mean of estimate to original.

        Returns:
            InversionResult with estimated stimulus and quality metrics.
        """
        p = self.params
        num_pts = len(original_stimulus)
        tme = np.arange(1, num_pts + 1) * time_step
        dark_current = p.dark_current

        # Impulse response filters
        opsin_filter = p.gamma * np.exp(-tme * p.sigma)
        pde_filter = np.exp(-tme * p.phi)

        # Steady-state derived quantities
        cur2ca = p.beta * p.cdark / dark_current
        gdark = (dark_current / p.k) ** (1.0 / p.n)
        cyclase_max = (p.eta / p.phi) * gdark * (1.0 + (p.cdark / p.kGC) ** p.m)

        current_response = target_response
        orig_stim = original_stimulus

        # Generate cGMP from current
        target_cgmp = (-current_response / p.k) ** (1.0 / p.n)

        # Generate calcium from current via convolution
        calcium_filter = time_step * np.exp(-tme * p.beta)
        target_calcium = -cur2ca * np.real(
            np.fft.ifft(np.fft.fft(current_response) * np.fft.fft(calcium_filter))
        )

        # Calculate cyclase rate from calcium
        cyclase_rate = cyclase_max / (1.0 + (target_calcium / p.kGC) ** p.m)

        # Calculate PDE from cGMP derivative (Eq. 3)
        g_deriv = np.diff(target_cgmp) / time_step
        g_deriv = np.append(g_deriv, g_deriv[-1])  # match length

        pde = (cyclase_rate - g_deriv) / target_cgmp
        pde = pde - p.eta / p.phi

        # Invert cascade via FFT deconvolution (Eqs. 1 and 2)
        estimated_stimulus = np.real(
            np.fft.ifft(
                np.fft.fft(pde) / (np.fft.fft(opsin_filter) * np.fft.fft(pde_filter))
            )
        )
        estimated_stimulus = estimated_stimulus / time_step

        # Raw R-squared (computed on last 90% of data, matching MATLAB)
        start = num_pts // 10
        raw_r_squared = 1.0 - (
            np.mean((estimated_stimulus[start:] - orig_stim[start:]) ** 2)
            / np.mean((orig_stim[start:] - np.mean(orig_stim[start:])) ** 2)
        )

        # Match mean if requested
        if match_mean:
            raw_estimate = (
                estimated_stimulus - np.mean(estimated_stimulus) + np.mean(orig_stim)
            )
        else:
            raw_estimate = estimated_stimulus.copy()

        # Match power spectrum if requested
        if match_power:
            stm_ps = np.real(np.fft.fft(orig_stim) * np.conj(np.fft.fft(orig_stim)))
            est_ps = np.real(
                np.fft.fft(estimated_stimulus) * np.conj(np.fft.fft(estimated_stimulus))
            )
            weights = np.sqrt(stm_ps / est_ps) * time_step
            corrected_estimate = np.real(
                np.fft.ifft(np.fft.fft(estimated_stimulus) * weights)
            )
        else:
            corrected_estimate = estimated_stimulus.copy()

        # Smooth with Gaussian window (causal FIR filter, matching MATLAB filter())
        win = gaussian(smooth_pts, std=(smooth_pts - 1) / 5.0)
        win_sum = np.sum(win)

        corrected_estimate = (
            lfilter(win, 1.0, corrected_estimate / time_step) / win_sum
        )
        smoothed_orig = lfilter(win, 1.0, orig_stim) / win_sum

        # Correlation
        xc = np.corrcoef(corrected_estimate, smoothed_orig)
        correlation = xc[0, 1]

        # Final R-squared
        r_squared = 1.0 - (
            np.mean((corrected_estimate - smoothed_orig) ** 2)
            / np.mean((smoothed_orig - np.mean(smoothed_orig)) ** 2)
        )

        # Residual
        residual = smoothed_orig - corrected_estimate

        return InversionResult(
            estimate=corrected_estimate,
            raw_estimate=raw_estimate,
            residual=residual,
            correlation=correlation,
            r_squared=r_squared,
            raw_r_squared=raw_r_squared,
        )
