"""Raw stem sections first, followed by bounded multi-source crown growth.

Section circles here locate axes, not DBH measurements. DBH remains the separate
original-ground + 1.30 m evidence workflow. Unassigned objects are quarantined in
the original region, never erased from the source LAS.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from algorithms.manual_span.dbh_recovery import DBH_RECOVERY_CONFIG, _section_circle_fit
from algorithms.spans.stem_guided_hybrid import (
    HybridTreeCandidate, StemEvidence, StemGuidedHybridResult, TreeQualityState,
    detect_validated_stems,
)
from algorithms.vegetation.coarse_to_fine import estimate_fine_vegetation_hag

RAW_STEM_FIRST_VERSION = "RAW_SECTIONS_STEM_FIRST_V2_STRUCTURAL_PLANES"
NON_TREE_CLASSES = (2, 6, 7, 18)  # ground, building, low/high noise


def structural_plane_mask(xyz, hag, stems, planes):
    """Quarantine planar infrastructure, protecting measured lower shafts.

    Plane coefficients and points share the scene-local metre coordinate frame.
    Never change the original LAS or pretend to distinguish coincident returns.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    mask = np.zeros(len(xyz), dtype=bool)
    for plane in planes:
        normal = np.asarray(plane['normal'], dtype=np.float64)
        near = np.abs(xyz @ normal - plane['offset_m']) <= plane['tolerance_m']
        tangent = np.asarray(plane['tangent'], dtype=np.float64)
        along = xyz @ tangent
        near &= (along >= plane['along_min_m'] - .10) & (along <= plane['along_max_m'] + .10)
        near &= (xyz[:, 2] >= plane['z_min_m'] - .10) & (xyz[:, 2] <= plane['z_max_m'] + .10)
        mask |= near
    # Protect the original lower shaft surface, not the previous wide cylinder
    # that also reinstated adjacent railing points after denoising.
    for stem in stems:
        radial = np.linalg.norm(xyz[:, :2] - stem.xy_local_m, axis=1)
        protected = (radial <= stem.cross_section_radius_m * 1.12 + .035) & (hag <= 2.5) & (hag >= .05)
        mask[protected] = False
    return mask


def detect_structural_planes(xyz, hag, stems):
    """Bounded, deterministic RANSAC for LARGE, densely occupied vertical panels.

    Local leaf planes/linear branches are insufficient evidence. A panel must
    span >=2 m horizontally and >=1.5 m vertically with a populated 20 cm grid.
    Geometry-identical or occluded objects remain ambiguous, not guessed away.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 180:
        return []
    _, reps = np.unique(np.floor(xyz / .12).astype(np.int32), axis=0, return_index=True)
    sample = xyz[reps]
    rng = np.random.default_rng(20261002)
    if len(sample) > 30_000:
        sample = sample[np.sort(rng.choice(len(sample), 30_000, replace=False))]
    remaining = sample.copy(); planes = []
    for _ in range(4):
        if len(remaining) < 180:
            break
        best = None; best_count = 0
        for _ in range(160):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal /= length
            if abs(normal[2]) > .12:  # vertical panel, not foliage or a roof
                continue
            offset = float(a @ normal)
            inliers = np.abs(remaining @ normal - offset) <= .025
            count = int(inliers.sum())
            if count > best_count:
                best = inliers; best_count = count
        if best_count < 180:
            break
        support = remaining[best]
        center = support.mean(axis=0)
        eigen, vectors = np.linalg.eigh(np.cov(support, rowvar=False))
        normal = vectors[:, 0]; offset = float(center @ normal)
        tangent = np.array([-normal[1], normal[0], 0.]); tangent /= np.linalg.norm(tangent)
        along = support @ tangent
        horizontal = float(np.ptp(along)); vertical = float(np.ptp(support[:, 2]))
        occupied = len(np.unique(np.floor(np.column_stack((along, support[:, 2])) / .20).astype(np.int32), axis=0))
        coverage = occupied / max(1., (horizontal / .20 + 1) * (vertical / .20 + 1))
        if abs(normal[2]) <= .12 and horizontal >= 2. and vertical >= 1.5 and coverage >= .40 and eigen[0] / max(eigen[1], 1e-9) < .002:
            planes.append(dict(normal=normal.tolist(), offset_m=offset, tangent=tangent.tolist(),
                along_min_m=float(along.min()), along_max_m=float(along.max()),
                z_min_m=float(support[:, 2].min()), z_max_m=float(support[:, 2].max()),
                tolerance_m=.03, representative_count=best_count, grid_coverage=coverage))
        remaining = remaining[~best]
    return planes


def _components(xy: np.ndarray, step: float = .07) -> list[np.ndarray]:
    """Cluster occupied cells, not millions of raw returns or a dense NxN matrix."""
    keys, inverse = np.unique(np.floor(xy / step).astype(np.int32), axis=0, return_inverse=True)
    pairs = cKDTree(keys).query_pairs(1.5, output_type="ndarray")
    if len(pairs):
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(keys), len(keys)))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(keys))
    row_labels = labels[inverse]
    order = np.argsort(row_labels, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(row_labels[order])) + 1)


def detect_raw_stem_axes(region, ground: np.ndarray, origin: tuple[float, float, float]):
    """Retain original low returns BEFORE canopy/noise filtering; fit many levels."""
    sections = []
    # Coarse HAG is only used to propose a vertical shaft. It is NOT the ground
    # datum or a DBH value published by the independent evidence calculation.
    xyz_parts, hag_parts = [], []
    for start in range(0, region.count, 500_000):
        xyz = np.asarray(region.xyz()[start:start + 500_000], dtype=np.float64)
        hag = estimate_fine_vegetation_hag(xyz, origin, ground)
        cls = np.asarray(region.classification()[start:start + 500_000])
        mask = (hag >= .35) & (hag <= 3.9) & ~np.isin(cls, NON_TREE_CLASSES)
        xyz_parts.append(xyz[mask]); hag_parts.append(hag[mask])
    xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3))
    hag = np.concatenate(hag_parts) if hag_parts else np.empty(0)
    proposal_cfg = replace(DBH_RECOVERY_CONFIG, minimum_angular_span_deg=110., minimum_occupied_sectors=4)
    for level in (.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8):
        points = xyz[np.abs(hag - level) <= .09, :2]
        if len(points) < 12:
            continue
        # 1 cm cells reduce repeated returns without losing trunk sections.
        _, first = np.unique(np.floor(points / .01).astype(np.int32), axis=0, return_index=True)
        points = points[first]
        for rows in _components(points):
            xy = points[rows]
            if len(xy) < 12 or np.max(np.ptp(xy, axis=0)) > 1.6:
                continue  # long railing/wall/car panels are not stem sections
            diagnostic, center, radius = _section_circle_fit(xy, proposal_cfg)
            if center is not None and radius is not None and .025 <= radius <= .70:
                sections.append((level, center, float(radius), len(xy), diagnostic["fit_rmse_m"]))
    # A trunk can touch a railing in a horizontal slice, making its connected
    # component long. Enumerate persistent narrow columns, then validate each
    # against ORIGINAL local sections rather than discarding that trunk with
    # the connected panel. Column geometry alone is never accepted as an axis.
    if len(xyz):
        _, reps = np.unique(np.floor(xyz / .10).astype(np.int32), axis=0, return_index=True)
        columns = detect_validated_stems(xyz[reps], hag[reps])
        index = cKDTree(xyz[:, :2])
        local_cfg = replace(proposal_cfg, minimum_inlier_fraction=.30)
        for column in columns:
            nearby = np.asarray(index.query_ball_point(column.xy_local_m, .65), dtype=np.int64)
            for level in (.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8):
                xy = np.unique(xyz[nearby[np.abs(hag[nearby] - level) <= .09], :2], axis=0)
                if len(xy) < 12:
                    continue
                diagnostic, center, radius = _section_circle_fit(xy, local_cfg)
                if center is None or radius is None or np.linalg.norm(center - column.xy_local_m) > .4:
                    continue
                if any(abs(s[0]-level) < .01 and np.linalg.norm(s[1]-center) < .12 for s in sections):
                    continue
                sections.append((level, center, float(radius), len(xy), diagnostic["fit_rmse_m"]))
    tracks = []
    for section in sorted(sections, key=lambda s: (s[0], s[1][0], s[1][1])):
        eligible = [
            (np.linalg.norm(section[1] - track[-1][1]), i)
            for i, track in enumerate(tracks)
            if 0 < section[0] - track[-1][0] <= .81
            and np.linalg.norm(section[1] - track[-1][1]) <= .20 + .12 * (section[0] - track[-1][0])
            and max(section[2], track[-1][2]) / min(section[2], track[-1][2]) < 2.0
        ]
        if eligible:
            tracks[min(eligible)[1]].append(section)
        else:
            tracks.append([section])
    stems = []
    for track in tracks:
        levels = np.asarray([s[0] for s in track]); centers = np.asarray([s[1] for s in track])
        if len(track) < 3 or np.ptp(levels) < 1.19 or min(levels) > 1.4:
            continue
        center = np.median(centers, axis=0)
        drift = float(np.max(np.linalg.norm(centers - center, axis=1)))
        if drift > .30:
            continue
        radius = float(np.median([s[2] for s in track]))
        if any(np.linalg.norm(center - np.asarray(s.xy_local_m)) < max(.15, radius * 1.5) for s in stems):
            continue
        gap = float(1 - len(track) / (round(np.ptp(levels) / .4) + 1))
        slope = np.polyfit(levels, centers, 1)[0]
        verticality = float(1. / np.sqrt(1. + np.dot(slope, slope)))
        confidence = float(np.clip((verticality + 1. - gap + 1. - drift / .30) / 3., 0., 1.))
        stems.append(StemEvidence(len(stems), tuple(center), float(np.ptp(levels)), len(track),
            gap, verticality, radius, sum(s[3] for s in track), drift, 1. - gap, confidence))
    return tuple(stems), {"section_proposal_count": len(sections), "axis_track_count": len(tracks),
        "accepted_raw_axis_count": len(stems), "raw_lower_return_count": len(xyz),
        "axis_sections_are_not_dbh": True}


def segment_from_raw_stems(xyz: np.ndarray, hag: np.ndarray, stems: tuple[StemEvidence, ...], *, classification=None, excluded_pole_axes=()):
    """Grow connected crowns upward from raw shafts; no CHM peak requirement."""
    xyz = np.asarray(xyz, dtype=np.float64); hag = np.asarray(hag)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    diagnostic = {"version": RAW_STEM_FIRST_VERSION, "seed_count": len(stems),
        "source_points_deleted": False, "unassigned_policy": "QUARANTINE_NOT_SOURCE_DELETION"}
    if not stems or not len(xyz):
        return StemGuidedHybridResult(labels, (), stems, diagnostic)
    centers = np.asarray([s.xy_local_m for s in stems])
    radii = np.asarray([max(.16, s.cross_section_radius_m * 1.4 + .06) for s in stems])
    distance, nearest = cKDTree(centers).query(xyz[:, :2])
    shaft = (distance <= radii[nearest]) & (hag >= .08) & (hag <= 4.)
    usable = np.isfinite(hag) & ((hag >= 2.5) | shaft)
    if classification is not None:
        usable &= ~np.isin(classification, NON_TREE_CLASSES)
    plane_rows = np.isfinite(hag) & (hag >= .05)
    if classification is not None:
        plane_rows &= ~np.isin(classification, NON_TREE_CLASSES)
    planes = detect_structural_planes(xyz[plane_rows], hag[plane_rows], stems)
    planar_junk = structural_plane_mask(xyz, hag, stems, planes)
    usable &= ~planar_junk
    diagnostic.update(structural_planes=planes, structural_plane_excluded_fine_count=int(planar_junk.sum()),
        non_tree_las_classes=list(NON_TREE_CLASSES))
    for pole in excluded_pole_axes:
        radial = np.linalg.norm(xyz[:, :2] - pole.xy_local_m, axis=1)
        usable &= radial > pole.cross_section_radius_m * 1.3 + .08
    rows = np.flatnonzero(usable)
    # Grow on 20 cm cells with bounded neighbours, then map labels exactly to
    # fine representatives. Low non-shaft cells cannot become a downward path
    # from a crown into a fence, car or bush.
    keys, inverse = np.unique(np.floor(xyz[rows] / .20).astype(np.int32), axis=0, return_inverse=True)
    node_xyz = (keys + .5) * .20
    count = np.bincount(inverse)
    node_hag = np.bincount(inverse, weights=hag[rows]) / count
    node_tree = np.full(len(keys), -1, dtype=np.int32)
    for stem in stems:
        tube = shaft[rows] & (nearest[rows] == stem.stem_seed_id)
        node_tree[np.unique(inverse[tube])] = stem.stem_seed_id
    seeds = np.flatnonzero(node_tree >= 0)
    if not len(seeds):
        return StemGuidedHybridResult(labels, (), stems, diagnostic)
    pairs = cKDTree(node_xyz).query_pairs(.65, output_type="ndarray")
    delta = node_xyz[pairs[:, 0]] - node_xyz[pairs[:, 1]]
    pairs = pairs[(np.linalg.norm(delta, axis=1) <= .36) | (np.linalg.norm(delta[:, :2], axis=1) <= .25)]
    left = np.concatenate((pairs[:, 0], pairs[:, 1])); right = np.concatenate((pairs[:, 1], pairs[:, 0]))
    ascending = node_hag[right] >= node_hag[left] - .12
    left, right = left[ascending], right[ascending]
    length = np.linalg.norm(node_xyz[left] - node_xyz[right], axis=1)
    graph = coo_matrix((length, (left, right)), shape=(len(keys), len(keys))).tocsr()
    distances, _, sources = dijkstra(graph, directed=True, indices=seeds, min_only=True, return_predecessors=True)
    reachable = np.isfinite(distances)
    node_owner = np.full(len(keys), -1, dtype=np.int32)
    node_owner[reachable] = node_tree[sources[reachable]]
    reached = np.flatnonzero(reachable)
    # Stop propagation into a neighbouring disconnected/unobserved crown. A
    # stem cannot claim an entire street merely because high foliage touches.
    radial = np.linalg.norm(node_xyz[reached, :2] - centers[node_owner[reached]], axis=1)
    outside = radial > np.minimum(5., 2.5 + .20 * node_hag[reached])
    node_owner[reached[outside]] = -1
    labels[rows] = node_owner[inverse]
    candidates, accepted = [], []
    for stem in stems:
        members = np.flatnonzero(labels == stem.stem_seed_id)
        crown = members[hag[members] >= 2.5]
        if len(crown) < 24:
            labels[members] = -1; continue
        spread = np.ptp(xyz[crown, :2], axis=0)
        # Bare posts have no spreading, volumetric branched crown. A broad
        # planar panel also fails its 3-D covariance/vertical-volume evidence.
        eigen = np.linalg.eigvalsh(np.cov(xyz[crown], rowvar=False))
        crown_ok = min(spread) >= max(.50, 3 * stem.cross_section_radius_m) and eigen[0] / max(eigen[-1], 1e-9) > .025 and np.ptp(hag[crown]) >= 1.
        # A lamp/utility post beside a low roof or shrub must not borrow that
        # low object as its "crown". Require spreading foliage in the actual
        # upper part as well; a shaft/cable is horizontally one-dimensional.
        upper = crown[hag[crown] >= max(2.5, .65 * float(np.max(hag[crown])))]
        if len(upper) < 12:
            crown_ok = False
        else:
            upper_spread = np.quantile(xyz[upper, :2], .90, axis=0) - np.quantile(xyz[upper, :2], .10, axis=0)
            upper_eigen = np.linalg.eigvalsh(np.cov(xyz[upper, :2], rowvar=False))
            crown_ok &= min(upper_spread) >= .45 and upper_eigen[0] >= .035
            off_axis = np.linalg.norm(xyz[upper, :2] - stem.xy_local_m, axis=1) > max(.45, 2.5 * stem.cross_section_radius_m)
            crown_ok &= np.mean(off_axis) >= .35
            upper_volume = np.linalg.eigvalsh(np.cov(xyz[upper], rowvar=False))
            crown_ok &= upper_volume[0] / max(upper_volume[-1], 1e-9) >= .06
        if not crown_ok:
            labels[members] = -1; continue
        tree_id = len(candidates)
        # Avoid remap collisions with later stem ids by remapping after loop.
        high = members[np.argmax(hag[members])]
        metrics = {"raw_stem_x_m": float(stem.xy_local_m[0]), "raw_stem_y_m": float(stem.xy_local_m[1]),
            "raw_stem_radius_m": float(stem.cross_section_radius_m), "raw_stem_sections": stem.occupied_height_bins,
            "crown_volume_ratio": float(eigen[0] / eigen[-1]),
            "upper_off_axis_fraction": float(np.mean(off_axis)), "upper_xy_spread_m": float(min(upper_spread)),
            "upper_volume_ratio": float(upper_volume[0] / max(upper_volume[-1], 1e-9))}
        candidates.append(HybridTreeCandidate(tree_id, TreeQualityState.CONFIRMED_TREE, members,
            (stem.stem_seed_id,), (tree_id,), tuple(xyz[high, :2]), True, True, True, True,
            len(members), 1, metrics, ("RAW_SHAFT_TO_CONNECTED_CROWN",)))
        accepted.append(stem.stem_seed_id)
    old = labels.copy(); labels[:] = -1
    for tree_id, stem_id in enumerate(accepted):
        labels[old == stem_id] = tree_id
    diagnostic.update(accepted_tree_count=len(candidates), rejected_axis_count=len(stems) - len(candidates),
        quarantined_fine_count=int(np.sum(labels < 0)), growth_node_count=len(keys), growth_edge_count=len(left))
    return StemGuidedHybridResult(labels, tuple(candidates), stems, diagnostic)
