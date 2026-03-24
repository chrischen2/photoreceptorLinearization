"""Linear model fitting utilities.

Port of fitLinearPRModel.m and PhotoreceptorModelLinWrapper.m
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

from .biophys_model import BiophysModel
from .coefficients import LinearCoefficients
from .params import PhotoreceptorParams


@dataclass
class FitResult:
    """Results from linear model fitting."""

    coefficients: LinearCoefficients  # Fitted coefficients
    error_history: list[float] = field(default_factory=list)  # Error at each evaluation
    final_error: float = 0.0  # Final normalized MSE


def _objective(
    coef: np.ndarray,
    target_resp: np.ndarray,
    target_stm: np.ndarray,
    dark_current: float,
    time_step: float,
    error_history: list[float],
    params: PhotoreceptorParams,
) -> float:
    """Objective function: normalized MSE between linear and biophysical responses."""
    coefficients = LinearCoefficients(
        sc_fact=coef[0], tau_r=coef[1], tau_d=coef[2]
    )

    model = BiophysModel(params)
    response = model.run_linear(target_stm, time_step, coefficients)
    response = response - np.mean(response)

    err = float(np.mean((response - target_resp) ** 2) / np.mean(target_resp**2))
    error_history.append(err)
    return err


def fit_linear_model(
    params: PhotoreceptorParams,
    mean_intensity: float,
    initial_coefficients: LinearCoefficients,
    num_pts: int = 500000,
    smooth_pts: int = 30,
    max_iterations: int = 2,
    max_fun_evals: int = 100,
    seed: int = 1,
) -> FitResult:
    """Fit linear model coefficients to match the biophysical model response.

    Generates a low-contrast Gaussian noise stimulus, runs the biophysical model,
    then optimizes the linear model coefficients to minimize the normalized MSE.

    Args:
        params: Photoreceptor parameters.
        mean_intensity: Mean light intensity in R*/s.
        initial_coefficients: Starting coefficients for optimization.
        num_pts: Number of stimulus points.
        smooth_pts: Gaussian smoothing window size for stimulus.
        max_iterations: Number of optimization restarts.
        max_fun_evals: Maximum function evaluations per optimization run.
        seed: Random seed for reproducibility.

    Returns:
        FitResult with fitted coefficients and error history.
    """
    time_step = params.time_step
    dark_current = params.dark_current

    # Generate smoothed Gaussian noise stimulus
    rng = np.random.default_rng(seed)
    mean_pho_flux = mean_intensity * time_step
    win = gaussian(smooth_pts, std=(smooth_pts - 1) / 5.0)
    win_sum = np.sum(win)
    raw_stim = rng.normal(mean_pho_flux, mean_pho_flux, num_pts)
    target_stm = lfilter(win, 1.0, raw_stim) / win_sum

    # Run biophysical model (double the stimulus for settling, take second half)
    model = BiophysModel(params)
    double_stm = np.concatenate([target_stm, target_stm])
    response = model.run_biophysical(double_stm, time_step)
    target_resp = response[num_pts:]
    target_resp = target_resp - np.mean(target_resp)

    # Optimize linear model coefficients
    coef = np.array([
        initial_coefficients.sc_fact,
        initial_coefficients.tau_r,
        initial_coefficients.tau_d,
    ])

    error_history: list[float] = []

    for _ in range(max_iterations):
        result = minimize(
            _objective,
            coef,
            args=(target_resp, target_stm, dark_current, time_step, error_history, params),
            method="Nelder-Mead",
            options={"maxfev": max_fun_evals},
        )
        coef = result.x

    final_coefficients = LinearCoefficients(
        sc_fact=coef[0], tau_r=coef[1], tau_d=coef[2]
    )

    return FitResult(
        coefficients=final_coefficients,
        error_history=error_history,
        final_error=error_history[-1] if error_history else 0.0,
    )
