"""Deterministic, fail-closed slice-track reconstruction for span conductors.

This module deliberately consumes only span-local wire-role representatives.  It
does not read COPC/viewer data and does not make any clearance decision.  A
track becomes ``CONFIRMED`` only after its raw geometric observations support a
model over the exact published ``s`` interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CONFIRMED = "CONFIRMED"
REVIEW_ONLY = "REVIEW_ONLY"
REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class SliceTrack:
    """A fitted span-local track; only ``CONFIRMED`` is distance-ready."""

    rows: np.ndarray
    inlier_rows: np.ndarray
    status: str
    reasons: tuple[str, ...]
    model_type: str
    model_u_coefficients: np.ndarray
    model_z_coefficients: np.ndarray
    catenary_parameters: np.ndarray
    supported_s_min_m: float
    supported_s_max_m: float
    supported_slice_count: int
    supported_slice_ratio: float
    coverage_ratio: float
    maximum_gap_m: float
    inlier_ratio: float
    median_residual_m: float
    p95_residual_m: float
    median_u_m: float
    u_residual_p95_m: float
    raw_support_count: int
    # These are the exact slice-association identities already used by this
    # reconstruction pass.  They are retained solely for a read-only evidence
    # artifact; they do not participate in any fit or confirmation decision.
    support_slice_ids: np.ndarray
    inlier_slice_ids: np.ndarray
    eligible_slice_count: int

    def diagnostic_dict(
        self, track_id: int, *, approach_start: bool, approach_end: bool
    ) -> dict[str, Any]:
        return {
            "track_id": track_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "model_type": self.model_type,
            "representative_count": int(len(self.rows)),
            "raw_support_count": self.raw_support_count,
            "supported_slice_count": self.supported_slice_count,
            "supported_slice_ratio": self.supported_slice_ratio,
            "coverage_ratio": self.coverage_ratio,
            "maximum_gap_m": self.maximum_gap_m,
            "supported_s_domain_m": [self.supported_s_min_m, self.supported_s_max_m],
            "inlier_ratio": self.inlier_ratio,
            "median_residual_m": self.median_residual_m,
            "p95_residual_m": self.p95_residual_m,
            "median_u_m": self.median_u_m,
            "u_residual_p95_m": self.u_residual_p95_m,
            "approach_start": approach_start,
            "approach_end": approach_end,
        }


def _components(values: np.ndarray, radius_u_m: float, radius_z_m: float) -> list[np.ndarray]:
    """Return deterministic connected components in one longitudinal slice."""
    from scipy.spatial import cKDTree

    if not len(values):
        return []
    scaled = np.column_stack((values[:, 0] / radius_u_m, values[:, 1] / radius_z_m))
    neighbours = cKDTree(scaled).query_ball_point(scaled, r=1.0)
    remaining = set(range(len(values)))
    result: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            component.append(current)
            stack.extend(member for member in neighbours[current] if member in remaining)
        result.append(np.asarray(sorted(component), dtype=np.int64))
    return result


def _robust_polynomial(
    s: np.ndarray, values: np.ndarray, *, degree: int, tolerance_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded deterministic residual rejection in a centred local coordinate."""
    center = float(np.median(s)) if len(s) else 0.0
    scale = max(float(np.ptp(s)), 1.0)
    x = (np.asarray(s, dtype=np.float64) - center) / scale
    y = np.asarray(values, dtype=np.float64)
    inliers = np.ones(len(x), dtype=bool)
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    for _ in range(8):
        if int(inliers.sum()) < degree + 1:
            break
        coefficients = np.polyfit(x[inliers], y[inliers], degree)
        residual = np.abs(np.polyval(coefficients, x) - y)
        next_inliers = residual <= tolerance_m
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers
    residual = np.abs(np.polyval(coefficients, x) - y)
    # Coefficients are stored as [centre, scale, poly descending] so callers
    # can evaluate in the same stable coordinate system without origin loss.
    return np.concatenate(([center, scale], coefficients)), residual, inliers


def evaluate_polynomial(coefficients: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.polyval(
        coefficients[2:], (np.asarray(s, dtype=np.float64) - coefficients[0]) / coefficients[1]
    )


def _fit_catenary(
    s: np.ndarray, z: np.ndarray, inliers: np.ndarray, config: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a stable sag catenary only when observed sag justifies it."""
    if int(inliers.sum()) < 6:
        return None
    s_fit, z_fit = (
        np.asarray(s[inliers], dtype=np.float64),
        np.asarray(z[inliers], dtype=np.float64),
    )
    if float(np.ptp(z_fit)) < config.wire_catenary_min_sag_m:
        return None
    try:
        from scipy.optimize import least_squares
    except ImportError:  # pragma: no cover - SciPy is already a project dependency
        return None
    span = max(float(np.ptp(s_fit)), 1.0)
    lo_a, hi_a = span / 20.0, max(span * 5_000.0, 10.0)
    initial = np.asarray([float(np.min(z_fit)), np.log(span * 10.0), float(np.median(s_fit))])

    def model(parameters: np.ndarray, values: np.ndarray) -> np.ndarray:
        c, log_a, s0 = parameters
        a = float(np.exp(log_a))
        argument = np.clip((values - s0) / a, -10.0, 10.0)
        return c + a * (np.cosh(argument) - 1.0)

    try:
        fitted = least_squares(
            lambda parameters: model(parameters, s_fit) - z_fit,
            initial,
            bounds=(
                np.asarray(
                    [float(np.min(z_fit) - 20.0), np.log(lo_a), float(np.min(s_fit) - span)]
                ),
                np.asarray(
                    [float(np.max(z_fit) + 20.0), np.log(hi_a), float(np.max(s_fit) + span)]
                ),
            ),
            max_nfev=200,
        )
    except (ValueError, FloatingPointError):
        return None
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    return fitted.x, np.abs(model(fitted.x, s) - z)


def evaluate_catenary(parameters: np.ndarray, s: np.ndarray) -> np.ndarray:
    c, log_a, s0 = np.asarray(parameters, dtype=np.float64)
    a = float(np.exp(log_a))
    return c + a * (np.cosh(np.clip((np.asarray(s, dtype=np.float64) - s0) / a, -10.0, 10.0)) - 1.0)


def reconstruct_slice_tracks(
    candidate_rows: np.ndarray,
    s_values: np.ndarray,
    u_values: np.ndarray,
    z_values: np.ndarray,
    raw_count: np.ndarray,
    span_length_m: float,
    config: Any,
) -> list[SliceTrack]:
    """Associate `(u,z)` slice observations into deterministic conductor tracks.

    The published domain is derived from robust inlier slices and a track with a
    gap larger than the configured evidence tolerance fails closed.
    """
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    candidate_rows = candidate_rows[
        (s_values[candidate_rows] >= 0.0) & (s_values[candidate_rows] <= span_length_m)
    ]
    if not len(candidate_rows):
        return []
    width = config.wire_slice_width_m
    slice_ids = np.floor(s_values[candidate_rows] / width).astype(np.int64)
    observations: list[tuple[int, np.ndarray, float, float, float]] = []
    for slice_id in np.unique(slice_ids):
        rows = candidate_rows[slice_ids == slice_id]
        for component in _components(
            np.column_stack((u_values[rows], z_values[rows])),
            config.wire_slice_observation_u_radius_m,
            config.wire_slice_observation_z_radius_m,
        ):
            member = rows[component]
            observations.append(
                (
                    int(slice_id),
                    member,
                    float(np.median(s_values[member])),
                    float(np.median(u_values[member])),
                    float(np.median(z_values[member])),
                )
            )
    observations.sort(key=lambda item: (item[0], item[3], item[4], int(item[1][0])))
    tracks: list[list[tuple[int, np.ndarray, float, float, float]]] = []
    for slice_id in sorted({item[0] for item in observations}):
        current = [item for item in observations if item[0] == slice_id]
        possible: list[tuple[float, int, int]] = []
        for track_id, track in enumerate(tracks):
            previous = track[-1]
            gap_m = (slice_id - previous[0]) * width
            if gap_m > config.wire_track_max_unsupported_gap_m:
                continue
            for observation_id, observation in enumerate(current):
                du = abs(observation[3] - previous[3])
                dz = abs(observation[4] - previous[4])
                if (
                    du <= config.wire_track_association_u_max_m
                    and dz <= config.wire_track_association_z_max_m
                ):
                    possible.append(
                        (
                            du / config.wire_track_association_u_max_m
                            + dz / config.wire_track_association_z_max_m,
                            track_id,
                            observation_id,
                        )
                    )
        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        for _, track_id, observation_id in sorted(possible):
            if track_id not in used_tracks and observation_id not in used_observations:
                tracks[track_id].append(current[observation_id])
                used_tracks.add(track_id)
                used_observations.add(observation_id)
        tracks.extend(
            [
                [observation]
                for observation_id, observation in enumerate(current)
                if observation_id not in used_observations
            ]
        )

    results: list[SliceTrack] = []
    for observations_for_track in tracks:
        rows = np.unique(np.concatenate([item[1] for item in observations_for_track])).astype(
            np.int64
        )
        support_s = np.asarray([item[2] for item in observations_for_track], dtype=np.float64)
        support_u = np.asarray([item[3] for item in observations_for_track], dtype=np.float64)
        support_z = np.asarray([item[4] for item in observations_for_track], dtype=np.float64)
        support_slice_ids = np.asarray(
            [item[0] for item in observations_for_track], dtype=np.int64
        )
        coeff_u, u_residual, inlier_u = _robust_polynomial(
            support_s, support_u, degree=1, tolerance_m=config.wire_confirmed_max_u_residual_m
        )
        coeff_z, z_residual, inlier_z = _robust_polynomial(
            support_s, support_z, degree=2, tolerance_m=config.wire_residual_tolerance_m
        )
        inliers = inlier_u & inlier_z
        catenary = _fit_catenary(support_s, support_z, inliers, config)
        model_type = "ROBUST_QUADRATIC"
        catenary_parameters = np.empty(0, dtype=np.float64)
        if catenary is not None:
            parameters, catenary_residual = catenary
            if (
                np.quantile(catenary_residual[inliers], 0.95)
                <= np.quantile(z_residual[inliers], 0.95) * 1.10 + 1e-9
            ):
                model_type = "CATENARY"
                catenary_parameters = parameters
                z_residual = catenary_residual
                inliers = inlier_u & (z_residual <= config.wire_residual_tolerance_m)
        supported_s = support_s[inliers]
        if len(supported_s):
            ordered = np.sort(supported_s)
            maximum_gap = float(np.max(np.diff(ordered))) if len(ordered) > 1 else float("inf")
            s_min, s_max = float(ordered[0]), float(ordered[-1])
        else:
            maximum_gap, s_min, s_max = float("inf"), 0.0, 0.0
        supported_slices = int(np.count_nonzero(inliers))
        possible_slices = max(1, int(np.floor((s_max - s_min) / width)) + 1)
        supported_ratio = supported_slices / possible_slices
        coverage = max(0.0, s_max - s_min) / max(span_length_m, 1e-9)
        residual = np.hypot(u_residual, z_residual)
        inlier_ratio = float(np.mean(inliers)) if len(inliers) else 0.0
        reasons: list[str] = []
        if supported_slices < config.wire_confirmed_min_supported_slices:
            reasons.append("insufficient_supported_slices")
        if supported_ratio < config.wire_confirmed_min_supported_slice_ratio:
            reasons.append("low_supported_slice_ratio")
        if coverage < config.wire_min_longitudinal_coverage_frac:
            reasons.append("low_longitudinal_coverage")
        if maximum_gap > config.wire_track_max_unsupported_gap_m:
            reasons.append("unsupported_internal_gap")
        if inlier_ratio < config.wire_confirmed_min_inlier_ratio:
            reasons.append("low_robust_inlier_ratio")
        if (
            len(supported_s)
            and float(np.median(residual[inliers])) > config.wire_confirmed_max_median_residual_m
        ):
            reasons.append("high_median_model_residual")
        if (
            len(supported_s)
            and float(np.quantile(residual[inliers], 0.95))
            > config.wire_confirmed_max_p95_residual_m
        ):
            reasons.append("high_p95_model_residual")
        if float(np.quantile(u_residual, 0.95)) > config.wire_confirmed_max_u_residual_m:
            reasons.append("unstable_transverse_position")
        status = (
            CONFIRMED
            if not reasons
            else (
                REVIEW_ONLY if coverage >= config.wire_min_longitudinal_coverage_frac else REJECTED
            )
        )
        results.append(
            SliceTrack(
                rows=rows,
                inlier_rows=np.unique(
                    np.concatenate(
                        [observations_for_track[index][1] for index in np.flatnonzero(inliers)]
                    )
                ).astype(np.int64)
                if np.any(inliers)
                else np.empty(0, dtype=np.int64),
                status=status,
                reasons=tuple(reasons),
                model_type=model_type,
                model_u_coefficients=coeff_u,
                model_z_coefficients=coeff_z,
                catenary_parameters=catenary_parameters,
                supported_s_min_m=s_min,
                supported_s_max_m=s_max,
                supported_slice_count=supported_slices,
                supported_slice_ratio=float(supported_ratio),
                coverage_ratio=float(coverage),
                maximum_gap_m=maximum_gap,
                inlier_ratio=inlier_ratio,
                median_residual_m=float(np.median(residual[inliers]))
                if np.any(inliers)
                else float("inf"),
                p95_residual_m=float(np.quantile(residual[inliers], 0.95))
                if np.any(inliers)
                else float("inf"),
                median_u_m=float(np.median(support_u)),
                u_residual_p95_m=float(np.quantile(u_residual, 0.95)),
                raw_support_count=int(np.sum(raw_count[rows], dtype=np.int64)),
                support_slice_ids=support_slice_ids,
                inlier_slice_ids=support_slice_ids[inliers],
                eligible_slice_count=possible_slices,
            )
        )
    return sorted(results, key=lambda track: (track.median_u_m, track.supported_s_min_m))


__all__ = [
    "CONFIRMED",
    "REVIEW_ONLY",
    "REJECTED",
    "SliceTrack",
    "evaluate_catenary",
    "evaluate_polynomial",
    "reconstruct_slice_tracks",
]
