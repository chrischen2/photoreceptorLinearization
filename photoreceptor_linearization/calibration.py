"""Calibration utilities for photoreceptor isomerization rates.

Port of calcIsomPerWatt.m and IsomerizationConverting.m
"""

from pathlib import Path

import numpy as np


# Physical constants
_PLANCK = 6.62607004e-34  # m^2*kg/s
_SPEED_OF_LIGHT = 299792458  # m/s

# Data directory
_DATA_DIR = Path(__file__).parent / "data"

# Collecting areas (um^2) for reference
COLLECTING_AREAS = {
    "mouse_rod": 0.5,
    "mouse_cone": 0.2,
    "primate_rod": 1.0,
    "primate_cone": 0.37,
}


def load_spectrum(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a spectrum from a two-column text file.

    Args:
        file_path: Path to the spectrum file.

    Returns:
        Tuple of (wavelengths, values) arrays.
    """
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]


def get_spectrum_path(category: str, name: str) -> Path:
    """Get path to a bundled spectrum data file.

    Args:
        category: 'devices', 'sources/mouse', or 'sources/primate'.
        name: Spectrum name, e.g. 'blue_led', 'rod', 's_cone'.

    Returns:
        Path to the spectrum text file.
    """
    path = _DATA_DIR / category / f"{name}_spectrum.txt"
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")
    return path


def calc_isom_per_watt(
    device_wavelengths: np.ndarray,
    device_values: np.ndarray,
    receptor_wavelengths: np.ndarray,
    receptor_values: np.ndarray,
) -> float:
    """Calculate photoreceptor isomerization rate per watt of incident light.

    Args:
        device_wavelengths: Device spectrum wavelengths (nm or m).
        device_values: Device spectral power distribution.
        receptor_wavelengths: Photoreceptor spectrum wavelengths (nm or m).
        receptor_values: Photoreceptor quantum efficiency values.

    Returns:
        Isomerization rate (R*/Watt).
    """
    # Convert from nm to meters if needed
    if np.max(receptor_wavelengths) > 1:
        receptor_wavelengths = receptor_wavelengths * 1e-9
    if np.max(device_wavelengths) > 1:
        device_wavelengths = device_wavelengths * 1e-9

    # Resample device spectrum at photoreceptor wavelengths
    device_resampled = np.interp(
        receptor_wavelengths, device_wavelengths, device_values
    )

    # Ensure no negative values
    device_resampled = np.maximum(device_resampled, 0.0)
    receptor_values = np.maximum(receptor_values, 0.0)

    # Differential wavelengths for integration
    d_lambda = np.diff(receptor_wavelengths)
    d_lambda = np.append(d_lambda, d_lambda[-1])

    # Numerator: overlap of device spectrum and photoreceptor sensitivity
    numerator = np.sum(device_resampled * receptor_values * d_lambda)

    # Denominator: device spectral irradiance weighted by photon energy
    photon_energy = _PLANCK * _SPEED_OF_LIGHT / receptor_wavelengths
    denominator = np.sum(device_resampled * photon_energy * d_lambda)

    return float(numerator / denominator)
