"""Immutable original-row pole/conductor evidence, separate from tree edits."""
from pathlib import Path
import numpy as np


def original_pole_query(root, cache, source, base, height):
    """Cover the pole neighbourhood without widening the DBH search contract."""
    from core.stage_b.dbh_evidence import original_raw_query
    clouds, identities = [], []
    for dx in (-.8, 0., .8):
        for dy in (-.8, 0., .8):
            raw, idx, _ = original_raw_query(root, cache, source, np.asarray(base[:2])+[dx,dy],
                                             (base[2]-.2, base[2]+height+.5))
            clouds.append(raw); identities.append(idx)
    rows = np.concatenate(identities)
    unique, positions = np.unique(rows, return_index=True)
    return np.concatenate(clouds)[positions], unique


def fit_selected_pole(xyz, indices, base, height):
    """Fit a continuous shaft near the operator's seed, never invent a pole."""
    xyz, indices, base = np.asarray(xyz), np.asarray(indices), np.asarray(base)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(indices) != len(xyz)
            or base.shape != (3,) or not np.isfinite(xyz).all()
            or not np.isfinite(base).all() or not np.isfinite(height) or height <= 0):
        raise ValueError('SELECTED_POLE_INPUT_INVALID')
    # The manual click can be on a crossarm or sampled surface, not shaft centre.
    # Look for a narrow, vertically continuous column near that click first.
    near = np.linalg.norm(xyz[:, :2] - base[:2], axis=1) <= 1.2
    near &= (xyz[:, 2] >= base[2] + .25) & (xyz[:, 2] <= base[2] + height + .5)
    proposal = xyz[near]
    seeds = [base[:2]]
    if len(proposal):
        cells = np.floor(proposal[:, :2] / .15).astype(np.int64)
        levels = np.floor((proposal[:, 2]-base[2])/.4).astype(np.int64)
        unique = np.unique(np.column_stack((cells, levels)), axis=0)
        columns, counts = np.unique(unique[:, :2], axis=0, return_counts=True)
        ranked = np.lexsort((np.linalg.norm((columns+.5)*.15-base[:2],axis=1), -counts))
        seeds = []
        for index in ranked:
            center = (columns[index]+.5)*.15
            if np.linalg.norm(center-base[:2]) > 1.2:
                continue
            if all(np.linalg.norm(center-other) > .18 for other in seeds):
                seeds.append(center)
            if len(seeds) >= 20:
                break
    failures = []
    for center in seeds:
        try:
            return _fit_selected_column(xyz, base, height, center)
        except ValueError as error:
            failures.append(str(error))
    raise ValueError('SELECTED_POLE_AXIS_INCONSISTENT' if 'SELECTED_POLE_AXIS_INCONSISTENT' in failures
                     else 'SELECTED_POLE_SHAFT_EVIDENCE_INSUFFICIENT')


def _fit_selected_column(xyz, base, height, center_seed):
    """Try alternative nearby columns without relaxing the fit-quality limits."""
    inside = (np.linalg.norm(xyz[:, :2] - center_seed, axis=1) <= .35)
    inside &= (xyz[:, 2] >= base[2] + .25) & (xyz[:, 2] <= base[2] + height + .5)
    rows = np.flatnonzero(inside)
    centers, elevations = [], []
    for low in np.arange(base[2] + .25, base[2] + height, .4):
        band = rows[(xyz[rows, 2] >= low) & (xyz[rows, 2] < low + .4)]
        if len(band) < 8:
            continue
        center = np.median(xyz[band, :2], axis=0)
        if np.quantile(np.linalg.norm(xyz[band, :2] - center, axis=1), .8) > .4:
            continue
        centers.append(center); elevations.append(float(np.median(xyz[band, 2])))
    if len(centers) < 5 or np.ptp(elevations) < max(1.5, height * .45):
        raise ValueError('SELECTED_POLE_SHAFT_EVIDENCE_INSUFFICIENT')
    z = np.asarray(elevations)
    design = np.column_stack((z - base[2], np.ones(len(z))))
    coefficients = np.linalg.lstsq(design, centers, rcond=None)[0]
    residual = np.linalg.norm(np.asarray(centers) - design @ coefficients, axis=1)
    if np.quantile(residual, .9) > .12 or np.linalg.norm(coefficients[0]) > .15:
        raise ValueError('SELECTED_POLE_AXIS_INCONSISTENT')
    axis_xy = np.column_stack((xyz[:, 2] - base[2], np.ones(len(xyz)))) @ coefficients
    radial = np.linalg.norm(xyz[:, :2] - axis_xy, axis=1)
    radius = min(.38, max(.10, float(np.quantile(radial[rows], .8)) + .04))
    protected = (radial <= radius) & (xyz[:, 2] >= base[2] - .1) & (xyz[:, 2] <= base[2] + height + .5)
    curve_z = np.linspace(base[2], base[2] + height, 40)
    curve_xy = np.column_stack((curve_z - base[2], np.ones(40))) @ coefficients
    return protected, {
        'xyz': np.column_stack((curve_xy, curve_z)).tolist(),
        'fit_rmse_m': float(np.sqrt(np.mean(residual ** 2))),
        'fit_search_seed_xy_m': np.asarray(center_seed).tolist(),
        'fit_height_basis': 'OPERATOR_CONFIRMED_BASE_AND_HEIGHT',
        'support_sections': len(z), 'radius_m': radius,
        'original_point_count': int(protected.sum()),
    }


def protected_infrastructure(root, result, project):
    """Read original evidence once per editor; fits stay fixed across revisions."""
    from core.stage_b.dbh_recovery import _context
    from core.stage_b.dbh_evidence import confined, _verified_original_file, strong_source
    from desktop.current_report import current_wire_curves
    root = Path(root)
    empty = dict(xyz=[], original_index=[], pole_segments=[], diagnostics=[], editable=False,
                 status='UNAVAILABLE', coordinate_space='scene_local_m')
    if not project.towers:
        return empty
    _, prep, _, origin = _context(root, result)
    origin = np.asarray(origin)
    source = prep['ground_preparation']['source_identity']
    cache = confined(root, prep['vegetation_role_tile_index_reference'])
    scan = next(s for s in project.scans if s.scan_id == result.scan_id)
    raw_path = Path(scan.point_cloud_source).resolve()
    expected = strong_source(source)
    stat = raw_path.stat()
    if stat.st_size != expected['file_size_bytes']:
        raise ValueError('PROTECTED_INFRASTRUCTURE_SOURCE_SIZE_MISMATCH')
    _verified_original_file(str(raw_path), stat.st_size, stat.st_mtime_ns, expected['sha256'])
    indices, points, poles, diagnostics = [], [], [], []
    for tower in project.towers:
        base = np.asarray(tower.position)
        height = float(tower.attributes.get('height_m', 0))
        try:
            if height <= 0:
                raise ValueError('SELECTED_POLE_HEIGHT_MISSING')
            raw, idx = original_pole_query(root, cache, source, base, height)
            try:
                keep, fit = fit_selected_pole(raw, idx, base, height)
            except ValueError:
                # Failed fitting cannot unlock the selected original pole
                # neighbourhood. Show it as evidence, NOT a fabricated fit.
                reserved = np.linalg.norm(raw[:, :2]-base[:2], axis=1) <= .6
                indices.append(idx[reserved]); points.append(raw[reserved]-origin)
                raise
            indices.append(idx[keep]); points.append(raw[keep] - origin)
            fit['xyz'] = (np.asarray(fit['xyz']) - origin).tolist()
            fit.update(tower_id=tower.tower_id, label=tower.display_name, editable=False)
            poles.append(fit)
        except (ValueError, OSError) as error:
            diagnostics.append(f'{tower.display_name}: {error}')
    # Exact confirmed raw conductor identities are distinct from fitted curves.
    wire_ids = []
    for span in result.span_results.values():
        wire = getattr(span, 'wire_result', None)
        if wire is not None:
            wire_ids.append(np.asarray(wire.confirmed_member_source_index, dtype=np.int64))
    wire_ids = np.unique(np.concatenate(wire_ids)) if wire_ids else np.empty(0, dtype=np.int64)
    if len(wire_ids):
        import laspy
        # The raw queries above verify this same source's SHA-256 and scale.
        with laspy.open(scan.point_cloud_source) as reader:
            h = reader.header
            if h.point_count != expected['source_point_count']:
                raise ValueError('PROTECTED_INFRASTRUCTURE_SOURCE_COUNT_MISMATCH')
            if wire_ids[0] < 0 or wire_ids[-1] >= h.point_count:
                raise ValueError('PROTECTED_INFRASTRUCTURE_WIRE_INDEX_OUT_OF_RANGE')
            sample = wire_ids[np.linspace(0, len(wire_ids)-1, min(12000,len(wire_ids)), dtype=int)]
            if h.are_points_compressed:
                xyz = np.empty((len(sample),3))
                start = 0
                while start < len(sample):
                    stop = start + 1
                    # Sparse source IDs must not allocate an unbounded LAZ
                    # interval (millions of rows for 64 display points).
                    while (stop < len(sample) and stop-start < 64
                           and sample[stop]-sample[start] < 65536):
                        stop += 1
                    ids = sample[start:stop]
                    reader.seek(int(ids[0]))
                    block = reader.read_points(int(ids[-1]-ids[0])+1)
                    local = ids-ids[0]
                    xyz[start:start+len(ids)] = np.column_stack((block.x[local],block.y[local],block.z[local]))
                    start = stop
                points.append(xyz-origin)
            else:
                raw = np.memmap(scan.point_cloud_source, mode='r', dtype=h.point_format.dtype(),
                                offset=h.offset_to_point_data, shape=(h.point_count,))
                xyz = np.column_stack([raw[n][sample].astype(float)*h.scales[i]+h.offsets[i]
                                       for i,n in enumerate(('X','Y','Z'))])
                del raw
                points.append(xyz-origin)
        indices.append(wire_ids)
    merged = np.unique(np.concatenate(indices)) if indices else np.empty(0, dtype=np.int64)
    cloud = np.concatenate(points) if points else np.empty((0, 3))
    if len(cloud) > 30000:
        cloud = cloud[np.linspace(0,len(cloud)-1,30000,dtype=int)]
    wires = current_wire_curves(result)
    return dict(xyz=cloud.tolist(), original_index=merged, pole_segments=poles,
                diagnostics=diagnostics, editable=False, coordinate_space='scene_local_m',
                status='AVAILABLE' if len(poles)==len(project.towers) and wires['status']=='AVAILABLE' else 'PARTIAL',
                protected_point_count=len(merged), pole_count=len(poles), wire_member_count=len(wire_ids))
