"""Photoreceptor model parameters.

Port of initPhotoreceptorParams.m
"""

from dataclasses import dataclass


@dataclass
class PhotoreceptorParams:
    """Biophysical parameters for a photoreceptor model."""

    sigma: float  # Receptor activity decay rate (1/sec)
    phi: float  # Phosphodiesterase activity decay rate (1/sec)
    eta: float  # Phosphodiesterase activation rate constant (1/sec)
    gdark: float  # Concentration of cGMP in darkness
    k: float  # Constant relating cGMP to current
    n: float  # Cooperativity for cGMP->current
    cdark: float  # Dark calcium concentration
    beta: float  # Rate constant for calcium removal (1/sec)
    m: float  # Cooperativity for cyclase (Hill coefficient)
    kGC: float  # Hill affinity for cyclase
    gamma: float  # Scaling factor for stimulus (rate of increase in opsin activity per R*/sec)
    time_step: float = 1e-4  # Simulation time step (sec)

    @property
    def dark_current(self) -> float:
        """Steady-state current in darkness."""
        return self.gdark**self.n * self.k


# Parameter sets for supported photoreceptor types
_PARAM_SETS = {
    "peripheral_primate_cone": dict(
        sigma=22.0, phi=22.0, eta=2000.0, gdark=35.0,
        k=0.01, n=3, cdark=1.0, beta=9.0, m=4, kGC=0.5, gamma=10.0,
    ),
    "primate_rod": dict(
        sigma=7.07, phi=7.07, eta=2.53, gdark=15.5,
        k=0.01, n=3, cdark=1.0, beta=25.0, m=4, kGC=0.5, gamma=4.2,
    ),
    "mouse_cone": dict(
        sigma=9.74, phi=9.74, eta=761.0, gdark=20.0,
        k=0.01, n=3, cdark=1.0, beta=2.64, m=4, kGC=0.4, gamma=10.0,
    ),
    "mouse_rod": dict(
        sigma=7.66, phi=7.66, eta=1.62, gdark=13.4,
        k=0.01, n=3, cdark=1.0, beta=25.0, m=4, kGC=0.40, gamma=8.0,
    ),
}


def init_params(model_type: str) -> PhotoreceptorParams:
    """Initialize parameters for a photoreceptor model type.

    Args:
        model_type: One of 'peripheral_primate_cone', 'primate_rod',
                    'mouse_cone', 'mouse_rod'.

    Returns:
        PhotoreceptorParams with consensus parameter values.
    """
    if model_type not in _PARAM_SETS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_PARAM_SETS.keys())}"
        )
    return PhotoreceptorParams(**_PARAM_SETS[model_type])
