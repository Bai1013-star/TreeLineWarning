"""Task B - PUBLIC_GT_PROPOSAL_TEST.

Question: can the public publisher ground truth actually produce
``CURRENT_PROPOSAL -> GT_CORRECT_MASK`` training pairs for a local refiner?

Two proposal sources are measured and never mixed:

``REAL_ALGORITHM_PROPOSAL``
    The product's own ``segment_stem_guided_hybrid`` - the crown/stem refinement
    the shipped ``CROWN_STEM_HYBRID`` backend calls - run on a real local patch of
    public forest. The input substitutions that public data forces are recorded
    on every row rather than hidden.

``SIMULATED_PROPOSAL``
    Deterministic corruptions of the publisher ground truth. Each row names its
    corruption class and severity.

Every row records ``GT_COVERAGE_IN_ROI``: the fraction of the target's own
ground-truth points inside the ROI. A point outside the ROI can never be
recovered by any refiner, so this bounds what AI repair could ever achieve.

All scoring is done on the ROI subset, because that is the tensor a refiner would
actually receive. Scoring on the whole patch would charge the algorithm for
points the ROI never showed it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import v2_config as cfg
import v2_index as index_module
import v2_scene as scene

SEVERITIES = ('LOW', 'MEDIUM', 'HIGH')
SEVERITY_SCALE = {'LOW': 0.10, 'MEDIUM': 0.25, 'HIGH': 0.45}

#: The corruptions a simulated proposal is drawn from, each at three severities.
#: All five defect classes the protocol makes a gate of are represented.
CORRUPTIONS = ('ADD_CONTAMINATION', 'REMOVE_MISSING', 'MERGED_NEIGHBOR',
               'CROWN_INCOMPLETE', 'STEM_MISSING')


def _rng(*parts) -> np.random.Generator:
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(np.random.SeedSequence(int.from_bytes(digest[:8], 'little')))


@dataclass
class RoiView:
    """One target's ROI, as the refiner would receive it."""

    patch_id: str
    site: str
    target_instance: int
    target_tree_id: str
    rows: np.ndarray            # indices into the patch
    xyz: np.ndarray             # (N, 3) local metric
    instance: np.ndarray        # (N,) patch instance ids inside the ROI
    ground_z: np.ndarray
    target_mask: np.ndarray     # (N,) bool
    containment: float          # fraction of the file's points inside the window
    points_file: int

    @property
    def target_points(self) -> int:
        return int(self.target_mask.sum())

    @property
    def n(self) -> int:
        return int(self.rows.size)


def qualifying_targets(patch: scene.Patch, minimum_points: int) -> list[dict]:
    """Instances that can be a meaningful refinement target.

    Two conditions, both about the truth rather than about the algorithm:

    * the instance must hold at least ``minimum_points`` points *inside the
      window*, so it is a tree and not a fragment;
    * it must keep at least ``PROPOSAL_MIN_TARGET_CONTAINMENT`` of its own file
      inside the window. A tree clipped by the window edge has a truncated
      "ground truth", and scoring a proposal against a truncated mask would
      charge the algorithm for points the window removed.
    """
    out = []
    for instance_id in patch.instance_ids:
        meta = patch.point_counts.get(instance_id, {})
        inside = int(meta.get('points_in_window', 0))
        total = int(meta.get('points_total', inside))
        containment = inside / max(total, 1)
        if inside < minimum_points or containment < cfg.PROPOSAL_MIN_TARGET_CONTAINMENT:
            continue
        out.append({'instance': instance_id, 'points_in_window': inside,
                    'points_file': total, 'containment': float(containment),
                    'tree_id': str(meta.get('tree_id', '')),
                    'partition': str(meta.get('partition', ''))})
    return out


def build_roi(patch: scene.Patch, target: dict) -> RoiView:
    mask = scene.roi_mask(patch, target['instance'])
    rows = np.flatnonzero(mask)
    return RoiView(
        patch_id=patch.patch_id, site=patch.site,
        target_instance=int(target['instance']), target_tree_id=target['tree_id'],
        rows=rows, xyz=patch.xyz[rows], instance=patch.instance[rows],
        ground_z=patch.ground_z[rows],
        target_mask=(patch.instance[rows] == target['instance']),
        containment=float(target['containment']), points_file=int(target['points_file']))


def simulate(view: RoiView, corruption: str, severity: str, seed_parts) -> dict:
    """One deterministic corruption of a correct mask. Never reported as REAL.

    Every branch removes or adds an exact count of points, computed from the
    target's own size rather than from a quantile threshold. A quantile threshold
    ties on real laser data and silently removes the wrong fraction.
    """
    rng = _rng(*seed_parts, corruption, severity)
    proposal = view.target_mask.copy()
    detail: dict = {'corruption': corruption, 'severity': severity}
    target_rows = np.flatnonzero(view.target_mask)
    scale = SEVERITY_SCALE[severity]

    if corruption == 'MERGED_NEIGHBOR':
        other = np.flatnonzero((~view.target_mask) & (view.instance > 0))
        if other.size == 0:
            return {'mask': proposal,
                    'detail': {**detail, 'skipped': 'no neighbour in ROI'}}
        neighbours = view.instance[other]
        centre = view.xyz[target_rows].mean(axis=0)
        distances = []
        for instance_id in np.unique(neighbours):
            points = view.xyz[other][neighbours == instance_id]
            distances.append((float(np.linalg.norm(points.mean(axis=0) - centre)),
                              int(instance_id)))
        distances.sort()
        nearest = distances[0][1]
        add = other[neighbours == nearest]
        proposal[add] = True
        detail.update({'merged_neighbour_instance': nearest,
                       'neighbour_distance_m': distances[0][0],
                       'added_points': int(add.size)})
    elif corruption == 'ADD_CONTAMINATION':
        pool = np.flatnonzero((~view.target_mask) & (view.instance > 0))
        if pool.size == 0:
            return {'mask': proposal,
                    'detail': {**detail, 'skipped': 'no other tree in ROI'}}
        take = min(int(round(scale * target_rows.size)), pool.size)
        chosen = rng.choice(pool, size=take, replace=False)
        proposal[chosen] = True
        detail['added_points'] = int(take)
    elif corruption == 'CROWN_INCOMPLETE':
        order = target_rows[np.argsort(-view.xyz[target_rows, 2], kind='stable')]
        take = max(1, int(round(scale * order.size)))
        proposal[order[:take]] = False
        detail['removed_points'] = int(take)
        detail['region'] = 'highest points'
    elif corruption == 'REMOVE_MISSING':
        take = max(1, int(round(scale * target_rows.size)))
        chosen = rng.choice(target_rows, size=min(take, target_rows.size - 1),
                            replace=False)
        proposal[chosen] = False
        detail['removed_points'] = int(len(chosen))
        detail['region'] = 'uniformly sampled'
    elif corruption == 'STEM_MISSING':
        hag = view.xyz[target_rows, 2] - view.ground_z[target_rows]
        low = target_rows[hag <= 2.0]
        if low.size == 0:
            return {'mask': proposal, 'detail': {**detail, 'skipped': 'no low points'}}
        order = low[np.argsort(-view.ground_z[low], kind='stable')]
        take = max(1, int(round(scale * order.size)))
        proposal[order[:take]] = False
        detail['removed_points'] = int(take)
        detail['region'] = 'lowest 2 m above ground'
    else:
        raise ValueError(f'unknown corruption {corruption!r}')
    return {'mask': proposal, 'detail': detail}


def score(view: RoiView, mask: np.ndarray, proposal_kind: str, extra: dict) -> dict:
    """Full protocol metric set for one proposal, scored on the ROI only."""
    gt_target = view.target_mask
    gt_other = (view.instance > 0) & (~gt_target)
    gt_background = view.instance < 0
    metrics = scene.proposal_metrics(mask, gt_target, gt_other, gt_background)
    missing_in_roi = int(np.count_nonzero(gt_target & ~mask))
    coverage = float(view.target_points / max(int(gt_target.sum()), 1))
    return {
        'patch_id': view.patch_id, 'site': view.site,
        'target_instance': view.target_instance,
        'target_tree_id': view.target_tree_id,
        'proposal_kind': proposal_kind, **metrics,
        'roi_points': view.n,
        'target_points_in_roi': view.target_points,
        'gt_coverage_in_roi': coverage,
        'missing_points_in_roi': missing_in_roi,
        'missing_recoverable_fraction': float(
            missing_in_roi / max(metrics['missing_points'], 1)),
        'target_containment_in_window': view.containment,
        'target_points_file': view.points_file,
        **extra,
    }


def real_route(patch: scene.Patch, minimum_points: int, verbose: bool = True,
               crown_source: str = 'GT_INSTANCE_LABELS', labels=None) -> dict:
    """Run the real algorithm once per patch, then score it per target ROI."""
    started = time.time()
    perturbation = None
    if labels is None and crown_source != 'GT_INSTANCE_LABELS':
        labels, perturbation = scene.perturb_crown_labels(patch, crown_source,
                                                          cfg.SEED, fraction=0.20, k=2)
    try:
        result = scene.real_proposals(patch, labels=labels, crown_source=crown_source)
    except Exception as exc:                                    # noqa: BLE001
        return {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}',
                'seconds': round(time.time() - started, 2), 'rows': [],
                'patch_id': patch.patch_id, 'site': patch.site,
                'crown_source': crown_source}
    proposals = result['proposals']
    matched = scene.match_proposals(proposals, patch.instance, patch.instance_ids)
    by_instance: dict[int, list[int]] = {}
    for row in matched['rows']:
        if row['instance'] >= 0 and row['unique']:
            by_instance.setdefault(row['instance'], []).append(row['proposal'])

    targets = qualifying_targets(patch, minimum_points)
    rows = []
    for target in targets:
        positions = by_instance.get(target['instance'], ())
        if not positions:
            continue
        view = build_roi(patch, target)
        baseline = score(view, view.target_mask, 'GT_CORRECT',
                         {'corruption': 'NONE', 'severity': 'NONE'})
        for position in positions:
            mask = proposals[position]['mask'][view.rows]
            row = score(view, mask, scene.REAL, {
                'crown_source': crown_source,
                'algorithm': result['algorithm'],
                'algorithm_state': proposals[position]['state'],
                'algorithm_stem_seed_count': proposals[position]['stem_seed_count'],
                'match_share': matched['rows'][position]['share'],
                'match_runner_up': matched['rows'][position]['runner_up'],
                'input_crown_iou': baseline['iou'],
                'substitutions': result['substitutions']})
            row['delta_iou_vs_input_crown'] = row['iou'] - baseline['iou']
            rows.append(row)
    if verbose:
        print(f'  {patch.patch_id} [{crown_source}]: {len(proposals)} proposals, '
              f'{len(targets)} qualifying targets, {len(rows)} unique pairs, '
              f'{time.time() - started:.1f}s', flush=True)
    return {'status': 'OK', 'patch_id': patch.patch_id, 'site': patch.site,
            'crown_source': crown_source,
            'crown_perturbation': perturbation,
            'crown_count': int(np.unique(patch.instance).size),
            'proposal_count': len(proposals), 'pair_count': len(rows),
            'targets_considered': len(targets),
            'seconds': round(time.time() - started, 2),
            'validated_stem_count': result.get('validated_stems'),
            'algorithm_diagnostics': result.get('diagnostics', {}),
            'substitutions': result['substitutions'],
            'proposal_states': _counts(str(p['state']) for p in proposals),
            'rows': rows}


def simulated_route(patch: scene.Patch, minimum_points: int,
                    verbose: bool = True) -> dict:
    """Deterministic corrupted proposals from the same real patch, per ROI."""
    started = time.time()
    targets = qualifying_targets(patch, minimum_points)
    rows: list[dict] = []
    for position, target in enumerate(targets):
        view = build_roi(patch, target)
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                simulated = simulate(view, corruption, severity,
                                     (patch.patch_id, target['instance']))
                rows.append(score(view, simulated['mask'], scene.SIMULATED,
                                  simulated['detail']))
        if verbose and (position + 1) % 20 == 0:
            print(f'    simulated {position + 1}/{len(targets)} targets, '
                  f'{len(rows)} rows, {time.time() - started:.1f}s', flush=True)
    return {'status': 'OK', 'patch_id': patch.patch_id, 'site': patch.site,
            'pair_count': len(rows), 'targets': len(targets),
            'seconds': round(time.time() - started, 2), 'rows': rows}


def correct_pairs(patch: scene.Patch, minimum_points: int) -> list[dict]:
    """The ``CURRENT_MASK -> GT_MASK`` pairs a refiner would train on.

    A correct proposal is the publisher ground truth itself, scored through the
    identical metric path as every corrupt one so the numbers are comparable.
    """
    rows = []
    for target in qualifying_targets(patch, minimum_points):
        view = build_roi(patch, target)
        rows.append(score(view, view.target_mask, 'GT_CORRECT', {
            'corruption': 'NONE', 'severity': 'NONE',
            'crown_source': 'GT_INSTANCE_LABELS'}))
    return rows


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def run(window_m: float = None, minimum_points: int = None, seed: int = None,
        verbose: bool = True, real_crown_sources=('GT_INSTANCE_LABELS',)) -> dict:
    started = time.time()
    seed = int(seed if seed is not None else cfg.SEED)
    window_m = float(window_m or cfg.PROPOSAL_WINDOW_M)
    minimum_points = int(minimum_points or cfg.PROPOSAL_MIN_TARGET_POINTS)

    patches, real_results, simulated_rows, correct_rows = [], [], [], []
    for site in index_module.available_sites():
        site_index = index_module.load_site_index(site)
        if verbose:
            print(f'[B] {site}: index {site_index["count"]} trees '
                  f'({site_index["digest"][:12]})', flush=True)
        patch = scene.build_patch(site, window_m, site_index,
                                  patch_id=f'{site}_w{int(window_m)}m_p0',
                                  verbose=verbose)
        patches.append({'patch_id': patch.patch_id, 'site': site,
                        'points': int(len(patch.xyz)),
                        'instances': len(patch.instance_ids),
                        'window_m': window_m,
                        'index_digest': site_index['digest'],
                        'points_per_instance': {
                            str(k): int(v.get('points_in_window', 0))
                            for k, v in sorted(patch.point_counts.items())}})
        if verbose:
            print(f'[B] {patch.patch_id}: {len(patch.xyz):,} points, '
                  f'{len(patch.instance_ids)} instances', flush=True)
        for crown_source in real_crown_sources:
            real_results.append(real_route(patch, minimum_points, verbose=verbose,
                                           crown_source=crown_source))
        simulated = simulated_route(patch, minimum_points, verbose=verbose)
        simulated_rows.extend(simulated['rows'])
        correct_rows.extend(correct_pairs(patch, minimum_points))
        if verbose:
            print(f'    simulated {simulated["pair_count"]} rows for '
                  f'{simulated["targets"]} targets ({simulated["seconds"]}s)', flush=True)

    primary = [r for r in real_results if r['crown_source'] == 'GT_INSTANCE_LABELS']
    real_rows = [row for result in primary for row in result['rows']]
    real_pairs = [row for row in real_rows if row['iou'] >= scene.MIN_PAIR_IOU]
    per_site_pairs: dict[str, int] = {}
    for row in real_pairs:
        per_site_pairs[row['site']] = per_site_pairs.get(row['site'], 0) + 1

    sites_present = sorted({entry['site'] for entry in patches})
    per_site_ok = all(per_site_pairs.get(site, 0) >= 5 for site in sites_present)
    usable = len(real_pairs) >= 10 and per_site_ok
    partial = len(real_pairs) > 0
    verdict = 'PASS' if usable else ('PARTIAL' if partial else 'FAIL')

    coverage = [row['gt_coverage_in_roi']
                for row in real_rows + simulated_rows + correct_rows]
    recoverable = [row['missing_recoverable_fraction']
                   for row in real_rows + simulated_rows]
    roi_points = [row['roi_points'] for row in correct_rows + simulated_rows]
    all_real_rows = [row for result in real_results for row in result['rows']]

    return {
        'schema': cfg.SCHEMA,
        'task': 'PUBLIC_GT_PROPOSAL_TEST',
        'config_digest': cfg.config_digest(),
        'seed': seed, 'window_m': window_m, 'minimum_points': minimum_points,
        'min_target_containment': cfg.PROPOSAL_MIN_TARGET_CONTAINMENT,
        'patches': patches,
        'real_routes': [{k: v for k, v in result.items() if k != 'rows'}
                        for result in real_results],
        'real_rows': real_rows,
        'real_rows_all_crown_sources': all_real_rows,
        'simulated_rows': simulated_rows,
        'correct_rows': correct_rows,
        'real_pairs_above_min_iou': len(real_pairs),
        'real_pairs_per_site': per_site_pairs,
        'real_pairs_total': len(real_rows),
        'simulated_rows_total': len(simulated_rows),
        'gt_correct_rows_total': len(correct_rows),
        'min_pair_iou': scene.MIN_PAIR_IOU,
        'roi_coverage': {
            'rows': len(coverage),
            'mean': float(np.mean(coverage)) if coverage else 0.0,
            'min': float(np.min(coverage)) if coverage else 0.0,
            'fraction_below_0.95': float(np.mean(np.asarray(coverage) < 0.95))
            if coverage else 0.0,
        },
        'roi_points': {
            'rows': len(roi_points),
            'median': float(np.median(roi_points)) if roi_points else 0.0,
            'p10': float(np.percentile(roi_points, 10)) if roi_points else 0.0,
            'p90': float(np.percentile(roi_points, 90)) if roi_points else 0.0,
            'max': float(np.max(roi_points)) if roi_points else 0.0,
        },
        'missing_recoverable_fraction': {
            'mean': float(np.mean(recoverable)) if recoverable else 0.0,
            'min': float(np.min(recoverable)) if recoverable else 0.0,
        },
        'real_defect_profile': _profile(real_pairs),
        'real_profile_all_crown_sources': _profile(all_real_rows),
        'simulated_defect_profile': _profile(simulated_rows),
        'PUBLIC_GT_PAIR_CONSTRUCTION': verdict,
        'seconds': round(time.time() - started, 2),
    }


def _profile(rows) -> dict:
    if not rows:
        return {'rows': 0}
    out = {'rows': len(rows),
           'iou_median': float(np.median([r['iou'] for r in rows])),
           'iou_mean': float(np.mean([r['iou'] for r in rows])),
           'precision_median': float(np.median([r['precision'] for r in rows])),
           'recall_median': float(np.median([r['recall'] for r in rows])),
           'merge_fraction': float(np.mean([r['is_merge'] for r in rows])),
           'incomplete_fraction': float(np.mean([r['is_split_or_incomplete']
                                                 for r in rows])),
           'by_site': {}}
    for site in sorted({r['site'] for r in rows}):
        group = [r for r in rows if r['site'] == site]
        out['by_site'][site] = {
            'rows': len(group),
            'iou_median': float(np.median([r['iou'] for r in group])),
            'iou_p10': float(np.percentile([r['iou'] for r in group], 10)),
            'iou_p90': float(np.percentile([r['iou'] for r in group], 90)),
            'neighbour_contamination_fraction_mean':
                float(np.mean([r['neighbour_contamination_fraction'] for r in group])),
        }
    if 'corruption' in rows[0]:
        out['by_corruption'] = {}
        for corruption in sorted({r['corruption'] for r in rows}):
            group = [r for r in rows if r['corruption'] == corruption]
            out['by_corruption'][corruption] = {
                'rows': len(group),
                'iou_median': float(np.median([r['iou'] for r in group])),
                'iou_min': float(np.min([r['iou'] for r in group])),
            }
        out['by_severity'] = {}
        for severity in sorted({r['severity'] for r in rows}):
            group = [r for r in rows if r['severity'] == severity]
            out['by_severity'][severity] = {
                'rows': len(group),
                'iou_median': float(np.median([r['iou'] for r in group])),
            }
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-m', type=float, default=cfg.PROPOSAL_WINDOW_M)
    parser.add_argument('--minimum-points', type=int,
                        default=cfg.PROPOSAL_MIN_TARGET_POINTS)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--real-crown-sources', nargs='*',
                        default=['CROWN_MERGE', 'CROWN_SPLIT'])
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    window = 12.0 if args.quick else args.window_m
    payload = run(window_m=window, minimum_points=args.minimum_points, seed=args.seed,
                  real_crown_sources=tuple(args.real_crown_sources))
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_QUICK' if args.quick else ''
    out = cfg.REPORT_DIR / f'PUBLIC_GT_PROPOSAL{suffix}.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'PUBLIC_GT_PAIR_CONSTRUCTION = {payload["PUBLIC_GT_PAIR_CONSTRUCTION"]}')
    print(f'real pairs {payload["real_pairs_above_min_iou"]} '
          f'{payload["real_pairs_per_site"]}, simulated rows '
          f'{payload["simulated_rows_total"]}, gt-correct rows '
          f'{payload["gt_correct_rows_total"]}')
    print(f'ROI coverage mean {payload["roi_coverage"]["mean"]:.4f}')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
