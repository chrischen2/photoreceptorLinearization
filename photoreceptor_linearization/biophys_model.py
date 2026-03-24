"""Biophysical and linear photoreceptor models.

Port of BiophysModel.m
"""

import numpy as np

from .coefficients import LinearCoefficients
from .params import PhotoreceptorParams


class BiophysModel:
    """Photoreceptor response model supporting biophysical and linear modes.

    Args:
        params: Photoreceptor parameters (from init_params).
    """

    def __init__(self, params: PhotoreceptorParams):
        self.params = params

    def run_biophysical(self, stimulus: np.ndarray, time_step: float) -> np.ndarray:
        """Simulate the full biophysical phototransduction cascade.

        Args:
            stimulus: Light stimulus array (units: R*/s * time_step).
            time_step: Simulation time step in seconds.

        Returns:
            Photocurrent response array (pA).
        """
        p = self.params
        dark_current = p.dark_current

        # Recompute gdark from dark current (matches MATLAB line 39)
        gdark = (dark_current / p.k) ** (1.0 / p.n)

        # Derived steady-state quantities
        cur2ca = p.beta * p.cdark / dark_current
        smax = (p.eta / p.phi) * gdark * (1.0 + (p.cdark / p.kGC) ** p.m)

        num_pts = len(stimulus)
        dt = time_step

        # State variables
        g = np.empty(num_pts)
        s = np.empty(num_pts)
        c = np.empty(num_pts)
        pde = np.empty(num_pts)
        r = np.empty(num_pts)

        # Initial conditions (steady state in darkness)
        g[0] = gdark
        s[0] = gdark * p.eta / p.phi
        c[0] = p.cdark
        pde[0] = p.eta / p.phi
        r[0] = 0.0

        # Euler integration
        for i in range(1, num_pts):
            # Rhodopsin activity
            r[i] = r[i - 1] + dt * (-p.sigma * r[i - 1])
            r[i] += p.gamma * stimulus[i - 1]

            # Phosphodiesterase activity
            pde[i] = pde[i - 1] + dt * (r[i - 1] + p.eta - p.phi * pde[i - 1])

            # Calcium concentration
            c[i] = c[i - 1] + dt * (cur2ca * p.k * g[i - 1] ** p.n - p.beta * c[i - 1])

            # Cyclase synthesis rate
            s[i] = smax / (1.0 + (c[i] / p.kGC) ** p.m)

            # cGMP concentration
            g[i] = g[i - 1] + dt * (s[i - 1] - pde[i - 1] * g[i - 1])

        # Photocurrent
        response = -p.k * g**p.n
        return response

    def run_linear(
        self,
        stimulus: np.ndarray,
        time_step: float,
        coefficients: LinearCoefficients,
    ) -> np.ndarray:
        """Simulate the linear filter approximation.

        Args:
            stimulus: Light stimulus array (units: R*/s * time_step).
            time_step: Simulation time step in seconds.
            coefficients: Linear model coefficients (ScFact, TauR, TauD).

        Returns:
            Photocurrent response array (pA).
        """
        num_pts = len(stimulus)
        tme = np.arange(1, num_pts + 1) * time_step

        # Parameterized temporal filter
        t_over_tau_r = tme / coefficients.tau_r
        filt = (
            coefficients.sc_fact
            * (t_over_tau_r**3 / (1.0 + t_over_tau_r**3))
            * np.exp(-tme / coefficients.tau_d)
        )

        # FFT convolution
        response = np.real(np.fft.ifft(np.fft.fft(stimulus) * np.fft.fft(filt)))
        response -= self.params.dark_current

        return response
