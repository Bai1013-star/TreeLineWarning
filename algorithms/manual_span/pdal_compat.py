"""Runtime PDAL compatibility shim for the manual single-span route.

The frozen FAST methods call PDAL for two things: SMRF Ground/HAG (reused
verbatim from a validated cache for the same source) and the covariance
Dimensionality features behind vegetation routing.  The covariance features
(Linearity / Planarity / Scattering / Verticality) are computed here by the
same formulas on the same covariance matrix using NumPy when the Python PDAL
binding is unavailable on this machine.  The frozen modules are never edited;
this module patches the function at runtime inside the manual process only.
"""

from __future__ import annotations

import time
from typing import Any


def _pdal_available() -> bool:
    try:
        import pdal  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - binding may be installed but broken
        return False


def ensure_pdal_compat() -> bool:
    """Patch the covariance-feature backend when PDAL is unavailable.

    Returns ``True`` when PDAL is available (no patch), ``False`` after a
    successful scipy/numpy substitution.
    """

    if _pdal_available():
        return True
    from algorithms.poles.evidence import local_pca_features
    from algorithms.vegetation import fast_vegetation_ab, fast_weighted_preprocess

    def _scipy_covariance_features(
        xyz_local_m: Any, *, config: Any
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        features = local_pca_features(
            xyz_local_m, neighbor_count=int(config.knn)
        )
        elapsed = time.perf_counter() - started
        fields = {
            "linearity": features["linearity"],
            "planarity": features["planarity"],
            "scattering": features["scattering"],
            "verticality": features["verticality"],
        }
        return fields, elapsed

    # ``fast_weighted_preprocess`` binds the original function by ``from ...
    # import``, so both module references must be patched for the manual process.
    fast_vegetation_ab._pdal_covariance_features = _scipy_covariance_features  # type: ignore[assignment]
    fast_weighted_preprocess._pdal_covariance_features = _scipy_covariance_features  # type: ignore[assignment]
    return False


__all__ = ["ensure_pdal_compat"]
