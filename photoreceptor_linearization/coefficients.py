"""Pre-fitted linear model coefficients.

Port of defineLinearModelCoefficients.m
"""

from dataclasses import dataclass


@dataclass
class LinearCoefficients:
    """Coefficients for the linear photoreceptor model."""

    sc_fact: float  # Scaling factor
    tau_r: float  # Rising time constant (sec)
    tau_d: float  # Decaying time constant (sec)


# Pre-fitted coefficients: {model_type: {light_level: (sc_fact, tau_r, tau_d)}}
_COEFFICIENTS: dict[str, dict[float, tuple[float, float, float]]] = {
    "peripheral_primate_cone": {
        1250: (12.6177, 0.0273, 0.0123),
        2500: (4.0624, 0.0196, 0.0153),
        5000: (1.6550, 0.0152, 0.0181),
        10000: (0.6963, 0.0121, 0.0210),
        20000: (0.3065, 0.0106, 0.0236),
    },
    "primate_rod": {
        1: (14.7078, 0.2170, 0.2702),
        3: (11.5524, 0.1945, 0.2374),
        10: (5.2782, 0.1412, 0.2085),
        30: (1.6408, 0.0904, 0.2102),
    },
    "mouse_cone": {
        1250: (0.5836, 0.0304, 0.0452),
        2500: (0.2795, 0.0276, 0.0481),
        5000: (0.1312, 0.0267, 0.0503),
        10000: (0.0621, 0.0277, 0.0525),
        20000: (0.0308, 0.0316, 0.0567),
    },
    "mouse_rod": {
        1: (15.6556, 0.2164, 0.2918),
        3: (11.3591, 0.1842, 0.2185),
        10: (3.6669, 0.1152, 0.1848),
        30: (0.9139, 0.0671, 0.1944),
    },
}


def get_linear_coefficients(model_type: str, light_level: float) -> LinearCoefficients:
    """Get pre-fitted linear model coefficients for a specific light level.

    Args:
        model_type: One of 'peripheral_primate_cone', 'primate_rod',
                    'mouse_cone', 'mouse_rod'.
        light_level: Light level in R*/photoreceptor/s.

    Returns:
        LinearCoefficients for the specified conditions.
    """
    if model_type not in _COEFFICIENTS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_COEFFICIENTS.keys())}"
        )
    levels = _COEFFICIENTS[model_type]
    if light_level not in levels:
        raise ValueError(
            f"No coefficients for light level {light_level}. "
            f"Available levels: {sorted(levels.keys())}"
        )
    sc_fact, tau_r, tau_d = levels[light_level]
    return LinearCoefficients(sc_fact=sc_fact, tau_r=tau_r, tau_d=tau_d)


def get_available_light_levels(model_type: str) -> list[float]:
    """Get available light levels for a model type.

    Args:
        model_type: Photoreceptor model type.

    Returns:
        Sorted list of light levels (R*/photoreceptor/s).
    """
    if model_type not in _COEFFICIENTS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_COEFFICIENTS.keys())}"
        )
    return sorted(_COEFFICIENTS[model_type].keys())
