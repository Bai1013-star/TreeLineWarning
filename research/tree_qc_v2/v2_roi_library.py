"""ROI library: the materialised training pairs for the mini refiner bakeoff.

Task B establishes where a pair can come from. This module turns that into a
reusable dataset on disk, so the three architectures are trained on byte-identical
inputs and the comparison is about architecture and nothing else.

What one row is:

``current_mask``
    the proposal under review. Either the real product algorithm's output on a
    publicly-reachable local scene (``REAL_ALGORITHM_PROPOSAL``) or a
    deterministic corruption of the publisher truth (``SIMULATED_PROPOSAL``).
``gt_mask``
    the publisher's own instance membership, restricted to the same ROI.
``roi_xyz``
    the local ROI in native metric coordinates, plus the membership channels the
    V2 input contract defines.

The library stores the ROI points once per target and one boolean pair per
(proposal, target), so 16 proposals of one tree cost 16 bitmasks rather than 16
copies of the cloud.

No held-out data and no product data are reachable from this module: the sites
come from ``v2_index.available_sites``, which only ever reports the two public
extraction roots that exist locally.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import v2_config as cfg
import v2_index as index_module
import v2_proposal_test as pt
import v2_scene as scene

LIBRARY_VERSION = 'tree_qc_v2_roi_library_v1'

#: Proposal kinds a training row can carry.
CURRENT_KINDS = ('REAL_ALGORITHM_PROPOSAL', 'SIMULATED_PROPOSAL', 'GT_CORRECT')


def _target_id(patch_id: str, instance: int) -> str:
    return f'{patch_id}#{instance}'


def _stride(xyz: np.ndarray, arrays: dict, cap: int) -> tuple:
    """Deterministic spatial stride so a huge ROI fits the library.

    Sorted by height then strided, which keeps the vertical profile intact - the
    same rule the frozen V1 geometry cache uses for its per-tree cap. Every array
    in ``arrays`` is strided by the same index set, so masks and points stay
    aligned by construction.
    """
    if len(xyz) <= cap:
        return xyz, dict(arrays), 1
    order = np.lexsort((xyz[:, 1], xyz[:, 0], xyz[:, 2]))
    step = int(np.ceil(len(xyz) / cap))
    keep = np.sort(order[::step][:cap])
    return xyz[keep], {k: v[keep] for k, v in arrays.items()}, step


def patch_real_proposals(patch: scene.Patch, real_crown_sources) -> dict:
    """Run the real algorithm once per patch, for every requested crown source.

    The result depends only on the patch and the crown labelling, not on which
    target is being looked at, so running it once per patch instead of once per
    target is both faster and removes any chance of two targets disagreeing about
    what the algorithm returned.
    """
    per_source = {}
    for crown_source in real_crown_sources:
        labels = None
        if crown_source != 'GT_INSTANCE_LABELS':
            labels, _perturbation = scene.perturb_crown_labels(
                patch, crown_source, cfg.SEED, fraction=0.20, k=2)
        try:
            result = scene.real_proposals(patch, labels=labels,
                                          crown_source=crown_source)
        except Exception as exc:                                # noqa: BLE001
            per_source[crown_source] = {'status': 'FAILED',
                                        'error': f'{type(exc).__name__}: {exc}'}
            continue
        matched = scene.match_proposals(result['proposals'], patch.instance,
                                        patch.instance_ids)
        by_instance: dict[int, list[int]] = {}
        for row in matched['rows']:
            if row['instance'] >= 0 and row['unique']:
                by_instance.setdefault(row['instance'], []).append(row['proposal'])
        per_source[crown_source] = {
            'status': 'OK', 'result': result, 'matched': matched,
            'by_instance': by_instance,
            'validated_stems': result.get('validated_stems')}
    return per_source


def build_target(patch: scene.Patch, target: dict, real_proposals: dict | None = None,
                 minimum_points: int = None, verbose: bool = False) -> dict:
    """One target's ROI plus every proposal that can be attached to it."""
    minimum_points = int(minimum_points or cfg.PROPOSAL_MIN_TARGET_POINTS)
    view = pt.build_roi(patch, target)
    masks: dict[str, np.ndarray] = {'gt': view.target_mask, 'instance': view.instance}
    rows: list[dict] = []

    rows.append({
        'proposal_id': 'GT_CORRECT', 'proposal_kind': 'GT_CORRECT',
        'corruption': 'NONE', 'severity': 'NONE', 'crown_source': 'GT_INSTANCE_LABELS',
        'skipped': None})

    for corruption in pt.CORRUPTIONS:
        for severity in pt.SEVERITIES:
            simulated = pt.simulate(view, corruption, severity,
                                    (patch.patch_id, target['instance']))
            name = f'SIM_{corruption}_{severity}'
            masks[name] = simulated['mask']
            rows.append({'proposal_id': name, 'proposal_kind': 'SIMULATED_PROPOSAL',
                         'corruption': corruption, 'severity': severity,
                         'crown_source': 'GT_INSTANCE_LABELS',
                         'skipped': simulated['detail'].get('skipped'),
                         'detail': {k: v for k, v in simulated['detail'].items()
                                    if isinstance(v, (int, float, str))}})

    real_records = []
    for crown_source, entry in (real_proposals or {}).items():
        if entry.get('status') != 'OK':
            real_records.append({'crown_source': crown_source,
                                 'status': entry.get('status', 'FAILED'),
                                 'error': entry.get('error')})
            continue
        positions = entry['by_instance'].get(target['instance'], ())
        proposals = entry['result']['proposals']
        matched_rows = entry['matched']['rows']
        for position in positions:
            name = f'REAL_{crown_source}_{position}'
            masks[name] = proposals[position]['mask'][view.rows]
            rows.append({'proposal_id': name, 'proposal_kind': scene.REAL,
                         'corruption': 'ALGORITHM_OUTPUT', 'severity': 'NONE',
                         'crown_source': crown_source, 'skipped': None,
                         'detail': {
                             'algorithm_state': proposals[position]['state'],
                             'match_share': matched_rows[position]['share'],
                             'validated_stems': entry.get('validated_stems')}})
        real_records.append({'crown_source': crown_source, 'status': 'OK',
                             'proposal_count': len(proposals),
                             'matched_positions': positions})

    roi_xyz, trimmed, step = _stride(view.xyz, masks, cfg.ROI_STORE_MAX_POINTS)
    if verbose:
        print(f'    target {target["tree_id"]}: ROI {view.n:,} points, '
              f'{len(rows)} proposals, stride {step}', flush=True)

    stored_instance = trimmed['instance']
    stored_gt = trimmed['gt']
    other = (stored_instance > 0) & (~stored_gt)
    background = stored_instance < 0
    scored = []
    for row in rows:
        proposal_id = row['proposal_id']
        mask = stored_gt if proposal_id == 'GT_CORRECT' else trimmed[proposal_id]
        metrics = scene.proposal_metrics(mask, stored_gt, other, background)
        scored.append({**{k: v for k, v in row.items() if k != 'detail'},
                       **{k: v for k, v in row.get('detail', {}).items()},
                       **metrics,
                       'roi_points_stored': int(len(roi_xyz)),
                       'roi_points_full': int(view.n),
                       'stride': step,
                       'gt_coverage_in_roi': float(
                           view.target_points / max(int(view.target_mask.sum()), 1))})
    return {
        'target_id': _target_id(patch.patch_id, target['instance']),
        'patch_id': patch.patch_id, 'site': patch.site,
        'target_instance': int(target['instance']),
        'target_tree_id': target['tree_id'],
        'partition': target.get('partition', ''),
        'containment': float(target['containment']),
        'roi_xyz': roi_xyz.astype(np.float32),
        'roi_instance': stored_instance.astype(np.int32),
        'masks': {key: value for key, value in trimmed.items()
                  if key not in ('instance', 'gt')},
        'rows': scored,
        'real_records': real_records,
    }


def build_library(window_m: float = None, windows_per_site: int = None,
                  minimum_points: int = None, seed: int = None,
                  verbose: bool = True,
                  real_crown_sources=('CROWN_MERGE', 'CROWN_SPLIT')) -> dict:
    started = time.time()
    window_m = float(window_m or cfg.ROI_WINDOW_M)
    windows_per_site = int(windows_per_site or cfg.ROI_WINDOWS_PER_SITE)
    minimum_points = int(minimum_points or cfg.PROPOSAL_MIN_TARGET_POINTS)

    targets: list[dict] = []
    windows_meta = []
    seen_trees: set[str] = set()
    for site in index_module.available_sites():
        site_index = index_module.load_site_index(site)
        windows = scene.densest_windows(site_index, window_m, windows_per_site,
                                        cell_m=cfg.ROI_WINDOW_GRID_M)
        for number, bounds in enumerate(windows):
            patch_id = f'{site}_w{int(window_m)}m_g{number:02d}'
            try:
                patch = scene.build_patch_at(site, bounds, site_index,
                                             patch_id=patch_id)
            except ValueError:
                continue
            qualifying = pt.qualifying_targets(patch, minimum_points)
            real_proposals = patch_real_proposals(patch, real_crown_sources)
            kept = 0
            for target in qualifying:
                row = patch.point_counts.get(target['instance'], {})
                key = str(row.get('key') or f"{site}|{target['tree_id']}")
                if key in seen_trees:
                    continue
                seen_trees.add(key)
                entry = build_target(patch, target, real_proposals=real_proposals,
                                     minimum_points=minimum_points, verbose=False)
                targets.append(entry)
                kept += 1
            windows_meta.append({'patch_id': patch_id, 'site': site,
                                 'bounds': [float(v) for v in bounds],
                                 'points': int(len(patch.xyz)),
                                 'instances': len(patch.instance_ids),
                                 'qualifying': len(qualifying), 'kept': kept})
            if verbose:
                print(f'  {patch_id}: {len(patch.xyz):,} pts, '
                      f'{len(qualifying)} qualifying, {kept} new targets '
                      f'(total {len(targets)})', flush=True)

    by_site: dict[str, int] = {}
    for target in targets:
        by_site[target['site']] = by_site.get(target['site'], 0) + 1
    return {
        'schema': cfg.SCHEMA,
        'library_version': LIBRARY_VERSION,
        'config_digest': cfg.config_digest(),
        'seed': int(seed if seed is not None else cfg.SEED),
        'window_m': window_m, 'windows_per_site': windows_per_site,
        'minimum_points': minimum_points,
        'windows': windows_meta,
        'targets_per_site': by_site,
        'target_count': len(targets),
        'proposal_rows': int(sum(len(t['rows']) for t in targets)),
        'targets': targets,
        'seconds': round(time.time() - started, 2),
    }


def save_library(library: dict, directory: Path | None = None) -> dict:
    """Persist the library: one npz per target plus an index json."""
    directory = Path(directory or cfg.ROI_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for target in library['targets']:
        name = target['target_id'].replace('#', '__') + '.npz'
        payload = {'xyz': target['roi_xyz'], 'instance': target['roi_instance'],
                   'gt_mask': np.asarray(target['masks']['gt'], dtype=bool)}
        for key, value in target['masks'].items():
            if key == 'gt':
                continue
            payload[f'mask_{key}'] = np.asarray(value, dtype=bool)
        path = directory / name
        with path.open('wb') as handle:
            np.savez_compressed(handle, **payload)
        digest = _file_digest(path)
        index_rows.append({
            'target_id': target['target_id'], 'site': target['site'],
            'patch_id': target['patch_id'], 'target_tree_id': target['target_tree_id'],
            'partition': target['partition'], 'file': name, 'sha256': digest,
            'roi_points': int(len(target['roi_xyz'])),
            'rows': target['rows'], 'real_records': target['real_records'],
            'containment': target['containment']})
    index = {
        'schema': cfg.SCHEMA, 'library_version': LIBRARY_VERSION,
        'config_digest': library['config_digest'],
        'window_m': library['window_m'], 'minimum_points': library['minimum_points'],
        'target_count': library['target_count'],
        'targets_per_site': library['targets_per_site'],
        'proposal_rows': library['proposal_rows'],
        'windows': library['windows'],
        'targets': index_rows,
        'seconds': library['seconds'],
    }
    target = directory / 'index.json'
    target.write_text(json.dumps(index, indent=2), encoding='utf-8')
    index['index_file'] = str(target)
    index['index_sha256'] = _file_digest(target)
    return index


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_library(directory: Path | None = None) -> dict:
    directory = Path(directory or cfg.ROI_DIR)
    index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
    for row in index['targets']:
        path = directory / row['file']
        if _file_digest(path) != row['sha256']:
            raise RuntimeError(f'ROI library entry changed on disk: {path}')
        with np.load(path, allow_pickle=False) as data:
            row['xyz'] = np.asarray(data['xyz'])
            row['instance'] = np.asarray(data['instance'])
            masks = {key[len('mask_'):]: np.asarray(data[key])
                     for key in data.files if key.startswith('mask_')}
            if 'gt_mask' in data.files:
                masks['gt'] = np.asarray(data['gt_mask'])
            else:
                # An older library entry: the ground-truth mask is exactly the
                # target instance's own points, the instance channel is stored, and
                # the target instance id is the suffix of the target id, so it is
                # reconstructed rather than guessed.
                instance_id = int(str(row['target_id']).rsplit('#', 1)[1])
                masks['gt'] = np.asarray(data['instance']) == instance_id
            row['masks'] = masks
    return index


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-m', type=float, default=cfg.ROI_WINDOW_M)
    parser.add_argument('--windows-per-site', type=int, default=cfg.ROI_WINDOWS_PER_SITE)
    parser.add_argument('--minimum-points', type=int,
                        default=cfg.PROPOSAL_MIN_TARGET_POINTS)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    windows = 3 if args.quick else args.windows_per_site
    library = build_library(window_m=args.window_m, windows_per_site=windows,
                            minimum_points=args.minimum_points, seed=args.seed)
    index = save_library(library)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_QUICK' if args.quick else ''
    out = cfg.REPORT_DIR / f'ROI_LIBRARY{suffix}.json'
    summary = {k: v for k, v in index.items() if k != 'targets'}
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'windows'}, indent=2))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
