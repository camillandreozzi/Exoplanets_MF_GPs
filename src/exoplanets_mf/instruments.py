"""JWST instrument-mode definitions for the combined WASP-80b spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InstrumentMode:
    """A half-open wavelength interval, except for the final inclusive mode."""

    label: str
    plot_label: str
    wavelength_min: float
    wavelength_max: float
    color: str


INSTRUMENT_MODES = (
    InstrumentMode(
        label="NIRCam F322W2",
        plot_label="NIRCam\nF322W2",
        wavelength_min=2.4,
        wavelength_max=4.0,
        color="#e8f1fb",
    ),
    InstrumentMode(
        label="NIRCam F444W",
        plot_label="NIRCam\nF444W",
        wavelength_min=4.0,
        wavelength_max=5.0,
        color="#fbecdc",
    ),
    InstrumentMode(
        label="MIRI LRS",
        plot_label="MIRI LRS",
        wavelength_min=5.0,
        wavelength_max=12.0,
        color="#eaf6ea",
    ),
)


def instrument_mode_masks(wavelengths) -> tuple[tuple[InstrumentMode, np.ndarray], ...]:
    """Assign every wavelength to F322W2, F444W, or MIRI LRS exactly once.

    Adjacent intervals use the convention ``[lower, upper)``. The final
    MIRI interval includes its upper edge, so the nominal coverage is
    F322W2 [2.4, 4), F444W [4, 5), and MIRI LRS [5, 12] micrometres.
    """
    wl = np.asarray(wavelengths, dtype=float)
    assignments = []

    for index, mode in enumerate(INSTRUMENT_MODES):
        if index == len(INSTRUMENT_MODES) - 1:
            mask = (wl >= mode.wavelength_min) & (wl <= mode.wavelength_max)
        else:
            mask = (wl >= mode.wavelength_min) & (wl < mode.wavelength_max)
        assignments.append((mode, mask))

    coverage = np.sum([mask for _, mask in assignments], axis=0)
    if np.any(coverage != 1):
        uncovered = wl[coverage == 0]
        overlapping = wl[coverage > 1]
        raise ValueError(
            "Each wavelength must belong to exactly one instrument mode; "
            f"uncovered={uncovered.tolist()}, overlapping={overlapping.tolist()}"
        )

    return tuple(assignments)
