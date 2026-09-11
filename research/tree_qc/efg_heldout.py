"""EFG held-out evaluation adapter: the only route to held-out geometry.

Held-out trees are unreachable by construction in the frozen pipeline. Three deliberate
Gate D safeguards make that true, and none of them is touched here:

* ``gate_d_provenance.authorize_donor`` refuses a held-out donor for any candidate, and
  ``assert_scene_sources`` raises ``HeldOutAccessError`` the moment a held-out source
  reaches an example;
* ``gate_d_generator.ExampleSpec`` refuses any partition outside ``train``/``validation``;
* ``gate_d_dataset.DatasetConfig`` refuses a non-model-selection partition, and its
  docstring states that held-out trees are unreachable from it.

So this module does **not** try to make the frozen dataset accept ``held_out``. It adds a
parallel, explicitly-labelled held-out route:

1. model-selection examples are still produced **only** by the frozen
   ``gate_d_generator.generate_example`` through the frozen ``GateDDataset`` — that path
   is reused, never re-implemented;
2. held-out examples are produced by the single corruption body in :func:`corrupt_scene`,
   a faithful port of the frozen sequence that differs in exactly one respect: which
   authorized context a candidate may draw real geometry from. There is no monkeypatching,
   no private state mutation, and no change to any file that participates in the generator
   digest;
3. the port is not trusted. :func:`parity_report` runs genuinely frozen validation scenes
   through :func:`corrupt_scene` and requires identical labels, identical sampler index
   digests and bit-identical feature tensors against the frozen ``GateDDataset``. Any
   drift is ``STOP_HELDOUT_ADAPTER_DRIFT`` and held-out stays locked.

Held-out source discipline (declared, checked, and re-derived from the frozen artifacts):

* a held-out candidate may contribute only its own points and the points of trees in its
  **own frozen held-out context**;
* no train or validation source may ever reach a held-out example, and no held-out source
  may reach a train/validation example (the frozen provenance already enforces the second
  direction);
* high-containment quarantine trees remain forbidden in every role, including held-out;
* the held-out geometry is used for **evaluation only**. No threshold, architecture,
  epoch, hyper-parameter or seed is ever derived from it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import gate_d_dataset as dataset_module
import gate_d_generator as gen
import gate_d_geometry as geometry_module
import gate_d_sampler as sampler
import gate_d_scene as scene_module
import numpy as np
from efg_dataset import NotApplicableExample
from efg_specs import SamplingSpec, enumerate_specs
from gate_d_provenance import HELD_OUT

ADAPTER_VERSION = 'EFG_HELDOUT_ADAPTER_V1'
PARITY_VERSION = 'EFG_HELDOUT_PARITY_V1'
PARITY_PARTITION = 'validation'
DEFAULT_PARITY_EXAMPLES = 63  # 3 validation candidates x 7 classes x 3 severities
PROVENANCE_FILE = 'GATE_C2_CONTEXT_PROVENANCE.json'


class HeldOutError(RuntimeError):
    """Raised with a STOP_* code whenever held-out discipline would be broken."""

    def __init__(self, code: str, detail):
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail


class AdapterDrift(HeldOutError):
    def __init__(self, detail):
        super().__init__('STOP_HELDOUT_ADAPTER_DRIFT', detail)


def _jsonable(value):
    """Make a NotApplicableError detail JSON-safe without losing its meaning."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --- held-out authorization -------------------------------------------------

@dataclass(frozen=True)
class HeldOutView:
    """The frozen held-out partition, read from the same provenance artifact."""

    name: str
    eligible: tuple[str, ...]
    context: dict[str, tuple[str, ...]]
    source_union: tuple[str, ...]


def _reports_root() -> Path:
    return Path(__file__).resolve().parent / 'reports'


def held_out_view(reports: Path | None = None) -> HeldOutView:
    """Read the held-out block of the frozen provenance artifact verbatim."""
    root = Path(reports) if reports is not None else _reports_root()
    payload = json.loads((root / PROVENANCE_FILE).read_text(encoding='utf-8'))
    block = payload[HELD_OUT]
    context = {cid: tuple(v) for cid, v in block['context'].items()}
    return HeldOutView(name=HELD_OUT, eligible=tuple(block['eligible']),
                       context=context, source_union=tuple(block['source_union']))


def assert_heldout_integrity(provenance, view: HeldOutView,
                             expected_candidates: int | None = None) -> dict:
    """Re-derive the held-out invariants from the artifacts instead of trusting them.

    This is what turns "no train/validation point source enters held-out" and "the frozen
    manifest did not change" into measured facts.
    """
    model_selection = set(provenance.train.source_union) | set(
        provenance.validation.source_union)
    held_out_sources = set(view.source_union)
    context_union: set[str] = set()
    for context in view.context.values():
        context_union.update(context)
    problems: list[dict] = []

    crossed_in = sorted(held_out_sources & model_selection)
    if crossed_in:
        problems.append({'invariant': 'heldout_source_in_model_selection',
                         'sources': crossed_in[:20], 'count': len(crossed_in)})
    escaped = sorted(context_union - held_out_sources)
    if escaped:
        problems.append({'invariant': 'heldout_context_outside_heldout_union',
                         'sources': escaped[:20], 'count': len(escaped)})
    quarantined = sorted((held_out_sources | context_union)
                         & set(provenance.high_containment_quarantine))
    if quarantined:
        problems.append({'invariant': 'quarantine_in_heldout',
                         'sources': quarantined[:20], 'count': len(quarantined)})
    quarantine_eligible = sorted(set(view.eligible)
                                 & set(provenance.high_containment_quarantine))
    if quarantine_eligible:
        problems.append({'invariant': 'quarantine_eligible_in_heldout',
                         'sources': quarantine_eligible[:20],
                         'count': len(quarantine_eligible)})
    alias_hits = sum(
        1 for source_id in held_out_sources
        if set(provenance.source_aliases_of.get(source_id, ()))
        & set(provenance.high_containment_aliases))
    if alias_hits:
        problems.append({'invariant': 'quarantine_alias_in_heldout', 'count': alias_hits})

    expected = int(expected_candidates if expected_candidates is not None
                   else gen.__dict__.get('EXPECTED_HELDOUT_UNIQUE', 749))
    if expected_candidates is not None and len(view.eligible) != expected:
        problems.append({'invariant': 'heldout_candidate_count_changed',
                         'expected': expected, 'actual': len(view.eligible)})

    if problems:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': 'held-out provenance violates a frozen invariant',
            'problems': problems})

    return {
        'held_out_candidates': len(view.eligible),
        'held_out_sources': len(held_out_sources),
        'held_out_context_edges': int(sum(len(v) for v in view.context.values())),
        'held_out_context_union': len(context_union),
        'model_selection_sources': len(model_selection),
        'cross_partition_overlap': 0,
        'quarantine_sources_in_heldout': 0,
        'quarantine_alias_sources_in_heldout': 0,
    }


def require_manifest_count(view: HeldOutView, expected: int,
                           runtime_records: int | None = None) -> dict:
    """STOP_HELDOUT_MANIFEST_CHANGED when the runtime count moved under the freeze.

    The expected count is the one recorded in the experiment freeze, which itself was read
    from the frozen provenance at freeze time. Nothing is hard-coded: if the frozen
    artifact changes, this comparison is what notices.
    """
    actual = len(view.eligible)
    if actual != int(expected):
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': 'held-out candidate count differs from the frozen expectation',
            'expected': int(expected), 'actual': actual,
            'runtime_source_records': runtime_records})
    return {'expected': int(expected), 'actual': actual, 'match': True,
            'runtime_source_records': runtime_records}


def authorize_heldout_scene(provenance, view: HeldOutView, candidate_id: str,
                            contributing_sources) -> tuple[str, ...]:
    """The unified fail-closed check for one held-out example.

    Every provenance refusal the frozen module can raise is converted to
    ``STOP_HELDOUT_MANIFEST_CHANGED`` so callers never see a partially-checked held-out
    example: a quarantine tree, a cross-partition source or an unknown identifier all stop
    the phase with one code.
    """
    try:
        return _authorize_heldout_scene(provenance, view, candidate_id,
                                        contributing_sources)
    except HeldOutError:
        raise
    except provenance_error_types() as exc:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': 'the frozen provenance refused a held-out source',
            'candidate': candidate_id,
            'error_type': type(exc).__name__, 'error': str(exc)[:300],
            'sources': sorted({str(s) for s in contributing_sources})[:20]}) from exc


def provenance_error_types() -> tuple:
    import gate_d_provenance as provenance_module
    return (provenance_module.ProvenanceViolation,)


def _authorize_heldout_scene(provenance, view: HeldOutView, candidate_id: str,
                             contributing_sources) -> tuple[str, ...]:
    eligible = set(view.eligible)
    if candidate_id not in view.context or candidate_id not in eligible:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} is not a frozen held-out candidate'})
    provenance.assert_not_quarantined(candidate_id, 'held-out candidate')
    provenance.assert_alias_clean(candidate_id, HELD_OUT)
    context = set(view.context[candidate_id])
    model_selection = set(provenance.train.source_union) | set(
        provenance.validation.source_union)
    unknown = [source for source in contributing_sources
               if source not in provenance.partition_of]
    if unknown:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': 'unknown contributing source', 'sources': unknown[:20]})
    union = tuple(sorted(set(contributing_sources)))
    for source_id in union:
        provenance.assert_not_quarantined(source_id, 'held-out source')
        provenance.assert_alias_clean(source_id, HELD_OUT)
        if source_id in model_selection:
            raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': 'a train/validation point source reached a held-out example',
                'source': source_id, 'candidate': candidate_id})
        if provenance.partition_of[source_id] != HELD_OUT:
            raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': f'source {source_id} is '
                           f'{provenance.partition_of[source_id]}, not held-out',
                'candidate': candidate_id})
        if source_id == candidate_id:
            continue
        if source_id not in context:
            raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': f'{source_id} is not in the frozen held-out context of '
                           f'{candidate_id}'})
    if candidate_id not in union:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} contributed no points to its own held-out scene'})
    return union


# --- held-out scene construction -------------------------------------------

def build_heldout_scene(candidate_id: str, geometry, provenance,
                        view: HeldOutView) -> scene_module.BaseScene:
    """The frozen scene construction, restricted to the held-out partition.

    Structurally identical to ``gate_d_scene.build_base_scene``: same query box, same
    clipping, same concatenation order, same minimum-point rules. The differences are the
    partition checks, which are held-out checks here, and the authorization call, which is
    :func:`authorize_heldout_scene`. The returned object is the frozen ``BaseScene`` type,
    so the corruption body consumes exactly the same structure as in model selection.
    """
    if geometry.candidate_id != candidate_id:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'geometry is for {geometry.candidate_id}, requested '
                       f'{candidate_id}'})
    if candidate_id not in set(view.eligible):
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} is not a frozen held-out candidate'})
    provenance.assert_not_quarantined(candidate_id, 'held-out candidate')
    provenance.assert_alias_clean(candidate_id, HELD_OUT)

    box = scene_module.frozen_query_box(provenance.tree[candidate_id]['bbox'])
    sources: list[scene_module.ScenePoints] = []
    candidate_points = np.asarray(geometry.candidate.points, dtype=np.float32)
    if candidate_points.shape[0] == 0:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} has no candidate geometry'})
    sources.append(scene_module.ScenePoints(source_id=candidate_id,
                                            points=candidate_points,
                                            is_candidate=True))

    authorised = view.context[candidate_id]
    if tuple(authorised) != tuple(geometry.context_ids):
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'geometry context {geometry.context_ids} does not match the '
                       f'frozen held-out context {authorised}'})
    for tree in geometry.context:
        if tree.canonical_tree_id not in set(authorised):
            raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': f'{tree.canonical_tree_id} is not authorised held-out '
                           f'context for {candidate_id}'})
        keep = scene_module.clip_to_query(tree.points, box)
        if keep.size == 0:
            continue
        sources.append(scene_module.ScenePoints(
            source_id=tree.canonical_tree_id,
            points=np.asarray(tree.points[keep], dtype=np.float32), is_candidate=False))

    points = np.concatenate([s.points for s in sources], axis=0)
    gt_source_id = np.concatenate(
        [np.full(s.points.shape[0], index, dtype=np.int32)
         for index, s in enumerate(sources)])
    candidate_mask = np.concatenate(
        [np.full(s.points.shape[0], s.is_candidate, dtype=bool) for s in sources])
    source_ids = tuple(s.source_id for s in sources)

    scene = scene_module.BaseScene(
        candidate_id=candidate_id, partition=HELD_OUT, site=geometry.candidate.site,
        points=points, gt_source_id=gt_source_id, source_ids=source_ids,
        candidate_mask=candidate_mask, query_box=box)
    if scene.point_count < scene_module.MIN_SCENE_POINTS:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} held-out scene has only {scene.point_count} '
                       'points'})
    if scene.candidate_points < scene_module.MIN_CANDIDATE_POINTS:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} has only {scene.candidate_points} candidate '
                       'points'})
    if scene.context_points < scene_module.MIN_CONTEXT_POINTS:
        # Held-out genuine context must satisfy the same minimum as model selection,
        # otherwise GOOD and every defect would be built from a candidate-only scene and
        # the held-out task would not be the same task.
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} has only {scene.context_points} clipped '
                       'held-out context points'})
    scene.audit = {
        'scene_version': scene_module.SCENE_VERSION,
        'query_box': [round(float(v), 6) for v in box],
        'query_pad_m': scene_module.QUERY_PAD_M,
        'candidate_id': candidate_id,
        'source_ids': list(source_ids),
        'source_counts': scene.source_counts(),
        'source_aliases': {source_id: list(provenance.source_aliases_of.get(source_id, ()))
                           for source_id in source_ids},
        'candidate_points': scene.candidate_points,
        'context_points': scene.context_points,
        'context_sources_used': len(source_ids) - 1,
        'context_sources_authorised': len(authorised),
        'adapter_version': ADAPTER_VERSION,
        'partition': HELD_OUT,
    }
    return scene


# --- the shared corruption body --------------------------------------------

@dataclass
class GeneratedExample:
    """Mirror of ``gate_d_generator.SyntheticExample`` for either partition."""

    spec: SamplingSpec
    points: np.ndarray
    membership: np.ndarray
    label: str
    changed_membership: np.ndarray
    source_ids: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def donors(self) -> tuple[str, ...]:
        return tuple(self.provenance.get('donors', ()))


def example_rng(spec: SamplingSpec) -> np.random.Generator:
    """Identical to the frozen ``_rng``: same material, same derivation."""
    material = f'{gen.GENERATOR_VERSION}|{spec.key}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8],
                                                'little'))


def corrupt_scene(spec: SamplingSpec, scene, provenance,
                  authorised_union=None) -> GeneratedExample:
    """The frozen corruption sequence, executed on a supplied scene.

    The RNG is created from the same material as the frozen generator and consumed in the
    same order by the same shared helper functions (``_draw_fraction``, ``_take``,
    ``_unit_direction``, ``_claim_from_sources``, ``_scattered_selection``,
    ``stem_evidence``). :func:`parity_report` compares this function's output with the
    frozen ``generate_example`` on frozen validation scenes across all classes and
    severities, so a divergence in call order or arithmetic is caught before any held-out
    geometry is read.
    """
    rng = example_rng(spec)
    membership = scene.membership_gt()
    gt_membership = membership.copy()
    detail: dict = {'fraction': 0.0, 'drawn_fraction': 0.0,
                    'mechanism': 'unchanged', 'claimed_points': 0, 'dropped_points': 0}
    donors: list[str] = []
    donors_detail: dict[str, int] = {}

    if spec.example_class == gen.GOOD:
        pass

    elif spec.example_class == gen.MERGED_MULTI_TREE:
        sources = gen._context_sources(scene)
        if not sources:
            raise gen.NotApplicableError('no_context_source', {})
        order = rng.permutation(len(sources))
        chosen_source = sources[int(order[0])]
        source_id = scene.source_ids[chosen_source]
        pool = gen._context_indices(scene)
        pool = pool[scene.gt_source_id[pool] == chosen_source]
        if pool.size == 0:
            raise gen.NotApplicableError('donor_source_clipped_empty', {'donor': source_id})
        fraction = gen._draw_fraction(rng, spec.severity)
        want = int(round(pool.size * fraction))
        if want < 1:
            raise gen.NotApplicableError('merged_below_one_point', {'pool': int(pool.size)})
        claimed = pool[gen._take(pool.size, want, rng)]
        membership[claimed] = True
        donors = [source_id]
        donors_detail[source_id] = int(claimed.size)
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_whole_neighbour',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'donor_source': source_id,
                  'donor_claimed_fraction_of_clipped': round(
                      float(claimed.size / pool.size), 6)}

    elif spec.example_class == gen.NEIGHBOR_CONTAMINATION:
        sources = gen._context_sources(scene)
        if not sources:
            raise gen.NotApplicableError('no_context_source', {})
        fraction = gen._draw_fraction(rng, spec.severity)
        claimed = gen._claim_from_sources(scene, sources, fraction, rng, local_only=True)
        membership[claimed] = True
        for source_index in np.unique(scene.gt_source_id[claimed]):
            source_id = scene.source_ids[int(source_index)]
            donors.append(source_id)
            donors_detail[source_id] = int(
                (scene.gt_source_id[claimed] == source_index).sum())
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_contiguous_neighbour_region',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'claim_radius_fraction': 0.5}

    elif spec.example_class == gen.NOISE:
        sources = gen._context_sources(scene)
        if not sources:
            raise gen.NotApplicableError('no_context_source', {})
        fraction = gen._draw_fraction(rng, spec.severity)
        pool = gen._context_indices(scene)
        want = int(round(pool.size * fraction))
        if want < 1:
            raise gen.NotApplicableError('noise_below_one_point', {'pool': int(pool.size)})
        claimed = gen._scattered_selection(scene, pool, want, rng)
        membership[claimed] = True
        for source_index in np.unique(scene.gt_source_id[claimed]):
            source_id = scene.source_ids[int(source_index)]
            donors.append(source_id)
            donors_detail[source_id] = int(
                (scene.gt_source_id[claimed] == source_index).sum())
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_scattered_context_points',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'distinct_cells': int(claimed.size)}

    elif spec.example_class == gen.OVER_SEGMENTED:
        candidate = gen._candidate_indices(scene)
        fraction = gen._draw_fraction(rng, spec.severity)
        axis = gen._unit_direction(rng)
        projection = scene.points[candidate] @ axis
        threshold = float(np.quantile(projection, fraction))
        drop = candidate[projection <= threshold]
        if drop.size < 1:
            raise gen.NotApplicableError('over_segmented_below_one_point',
                                         {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_candidate_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'axis': [round(float(v), 6) for v in axis],
                  'dropped_fraction_of_candidate': round(
                      float(drop.size / candidate.size), 6)}

    elif spec.example_class == gen.CROWN_INCOMPLETE:
        candidate = gen._candidate_indices(scene)
        fraction = gen._draw_fraction(rng, spec.severity)
        z = scene.points[candidate, 2]
        z_low, z_high = float(z.min()), float(z.max())
        threshold = z_high - (z_high - z_low) * fraction
        drop = candidate[z > threshold]
        if drop.size < 1:
            raise gen.NotApplicableError('crown_incomplete_below_one_point',
                                         {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_crown_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'z_cut': round(threshold, 6),
                  'z_span': [round(z_low, 6), round(z_high, 6)],
                  'dropped_fraction_of_candidate': round(
                      float(drop.size / candidate.size), 6)}

    elif spec.example_class == gen.STEM_MISSING:
        evidence = gen.stem_evidence(scene)
        if not evidence['applicable']:
            raise gen.NotApplicableError(gen.STEM_NOT_APPLICABLE, evidence)
        candidate = gen._candidate_indices(scene)
        fraction = gen._draw_fraction(rng, spec.severity)
        z_low, z_high = evidence['z_span']
        z_cut = z_low + (z_high - z_low) * fraction
        axis_x, axis_y = evidence['stem_axis_xy']
        horizontal = np.hypot(scene.points[candidate, 0] - axis_x,
                              scene.points[candidate, 1] - axis_y)
        drop = candidate[(scene.points[candidate, 2] <= z_cut)
                         & (horizontal <= evidence['radius_m'])]
        if drop.size < 1:
            raise gen.NotApplicableError('stem_missing_below_one_point',
                                         {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_stem_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'z_cut': round(z_cut, 6), 'stem_axis_xy': evidence['stem_axis_xy'],
                  'stem_radius_m': evidence['radius_m'],
                  'stem_evidence': {k: v for k, v in evidence.items()
                                    if k != 'lower_mask'}}

    else:  # pragma: no cover - guarded by SamplingSpec
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'unhandled class {spec.example_class}'})

    changed = membership != gt_membership
    union = (tuple(authorised_union) if authorised_union is not None
             else scene_module.scene_source_union(scene))
    label = gen.LABEL_PASS if spec.example_class == gen.GOOD else gen.LABEL_REVIEW
    record = {
        'generator_version': gen.GENERATOR_VERSION,
        'membership_channel': gen.MEMBERSHIP_CHANNEL,
        'adapter_version': ADAPTER_VERSION,
        'key': spec.key,
        'candidate_id': spec.candidate_id,
        'partition': spec.partition,
        'site': scene.site,
        'class': spec.example_class,
        'severity': spec.severity,
        'label': label,
        'seed': spec.seed,
        'scene_version': scene.audit.get('scene_version'),
        'query_box': scene.audit.get('query_box'),
        'query_pad_m': scene.audit.get('query_pad_m'),
        'donors': sorted(set(donors)),
        'donor_point_counts': donors_detail,
        'context_ids': list(scene.source_ids[1:]),
        'point_count': scene.point_count,
        'actual_point_source_ids': list(union),
        'actual_point_source_counts': scene.source_counts(),
        'actual_point_source_aliases': {
            source_id: list(provenance.source_aliases_of.get(source_id, ()))
            for source_id in union},
        'candidate_points': scene.candidate_points,
        'context_points': scene.context_points,
        'membership_fraction': round(float(membership.mean()), 6),
        'gt_membership_fraction': round(float(gt_membership.mean()), 6),
        'changed_membership_points': int(changed.sum()),
        'changed_membership_fraction': round(float(changed.mean()), 6),
        'claimed_points': int(detail.get('claimed_points', 0)),
        'dropped_points': int(detail.get('dropped_points', 0)),
        'source_union': list(union),
        'corruption': detail,
    }
    return GeneratedExample(spec=spec, points=scene.points, membership=membership,
                            label=label, changed_membership=changed,
                            source_ids=scene.source_ids, provenance=record)


def generate_heldout_example(spec: SamplingSpec, geometry, provenance,
                             view: HeldOutView) -> GeneratedExample:
    """:func:`corrupt_scene` on a held-out scene, with the held-out authorization gate."""
    scene = build_heldout_scene(spec.candidate_id, geometry, provenance, view)
    union = scene_module.scene_source_union(scene)
    authorised = authorize_heldout_scene(provenance, view, spec.candidate_id, union)
    if tuple(authorised) != tuple(union):
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': 'held-out source union normalization mismatch'})
    return corrupt_scene(spec, scene, provenance, authorised_union=authorised)


def generate_selftest_example(spec: SamplingSpec, scene, provenance) -> GeneratedExample:
    """The parity path: the shared body on a frozen model-selection scene."""
    union = scene_module.scene_source_union(scene)
    provenance.assert_scene_sources(spec.candidate_id, union)
    return corrupt_scene(spec, scene, provenance, authorised_union=union)


# --- held-out geometry loading ----------------------------------------------

HELDOUT_GEOMETRY_CACHE_VERSION = 'EFG_HELDOUT_GEOMETRY_CACHE_V1'


def heldout_geometry_key(provenance, view: HeldOutView, candidate_id: str,
                         max_tree_points: int) -> str:
    """Cache key for held-out geometry, mirroring the frozen key construction.

    The frozen key is unavailable here because it calls ``authorized_context``, which
    deliberately refuses a held-out tree. The components are otherwise the same, plus the
    held-out candidate count so a changed manifest cannot re-use old geometry.
    """
    payload = json.dumps({
        'version': HELDOUT_GEOMETRY_CACHE_VERSION,
        'adapter_version': ADAPTER_VERSION,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'provenance_digest': provenance.provenance_digest,
        'candidate': candidate_id,
        'context': list(view.context[candidate_id]),
        'held_out_candidates': len(view.eligible),
        'max_tree_points': int(max_tree_points),
    }, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def load_heldout_geometry(cache_root: Path | str, provenance, view: HeldOutView,
                          candidate_id: str, data_root: Path | None = None,
                          max_tree_points: int = geometry_module.MAX_TREE_POINTS
                          ) -> geometry_module.CandidateGeometry:
    """Authorise, then read, one held-out candidate and its held-out context.

    This is the held-out twin of ``gate_d_geometry.build_candidate_geometry``. It reuses
    the frozen reader, the frozen stride cap and the frozen cache reader/writer unchanged,
    and replaces only the authorization step — which for held-out data is
    :func:`authorize_heldout_scene` rather than the model-selection allowlist that
    deliberately rejects held-out identifiers.

    Geometry is cached under its own key in its own directory, so the three held-out
    evaluation passes and the robustness sweep read each candidate from disk once instead
    of once per pass. Authorization is re-checked on every call, cache hit or not.
    """
    eligible = set(view.eligible)
    if candidate_id not in eligible:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} is not a frozen held-out candidate'})
    context_ids = tuple(view.context[candidate_id])
    if not context_ids:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} has an empty frozen held-out context'})
    # Authorise every source before a single point is read, so one illegal id aborts the
    # whole batch instead of producing partially-legal geometry.
    authorize_heldout_scene(provenance, view, candidate_id,
                            (candidate_id, *context_ids))
    if provenance.partition_of[candidate_id] != HELD_OUT:
        raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
            'message': f'{candidate_id} is not held-out'})

    root = Path(cache_root)
    path = root / (heldout_geometry_key(provenance, view, candidate_id, max_tree_points)
                   + '.npz')
    if path.exists():
        try:
            return geometry_module._read_cache(path, candidate_id)
        except (OSError, KeyError, ValueError, geometry_module.GeometryError):
            path.unlink(missing_ok=True)

    data_root_path = (Path(data_root) if data_root is not None
                      else geometry_module.DATA_ROOT)
    candidate = geometry_module._read_tree(candidate_id, provenance, data_root_path,
                                           max_tree_points)
    context = tuple(geometry_module._read_tree(source_id, provenance, data_root_path,
                                               max_tree_points)
                    for source_id in context_ids)
    built = geometry_module.CandidateGeometry(candidate=candidate, context=context)
    geometry_module.write_cache(path, built)
    return built


def heldout_geometry_cache_root(layout) -> Path:
    """Held-out geometry lives in its own directory, never with model-selection files."""
    return Path(layout.path('cache', 'heldout_geometry'))


# --- dataset over the held-out view ----------------------------------------

@dataclass(frozen=True)
class HeldOutItem:
    features: np.ndarray
    label: int
    key: str
    provenance: dict


class HeldOutDataset:
    """Held-out examples in the frozen dataset's item shape.

    The held-out geometry is read only through :func:`build_heldout_scene`, which consults
    the frozen held-out context. Nothing here can select a threshold: this class has no
    threshold API at all, and ``gate_d_metrics.select_threshold`` refuses ``held_out`` by
    construction.
    """

    def __init__(self, provenance, view: HeldOutView, candidate_ids=None,
                 points: int = 4096, repeats: int = 1, seed_base: int = 0,
                 cache_root: Path | str | None = None, data_root: Path | None = None):
        self.provenance = provenance
        self.view = view
        self.points = int(points)
        self.repeats = int(repeats)
        self.seed_base = int(seed_base)
        self.cache_root = (Path(cache_root) if cache_root is not None
                           else Path('D:/TreeQCTraining/efg_heldout_cache'))
        self.data_root = data_root
        ids = list(candidate_ids if candidate_ids is not None else view.eligible)
        eligible = set(view.eligible)
        for candidate_id in ids:
            if candidate_id not in eligible:
                raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                    'message': f'{candidate_id} is not a frozen held-out candidate'})
        self.candidate_ids = tuple(ids)
        self.specs = tuple(enumerate_specs(self.candidate_ids, HELD_OUT,
                                           seed_base=self.seed_base,
                                           repeats=self.repeats))
        self._geometry: dict = {}
        # Exclusions are keyed by spec, not by index, so they survive a change in the
        # candidate subset and are written next to the held-out geometry cache.
        self.exclusions: dict[str, dict] = {}
        self._exclusions_path = self.cache_root / 'heldout_exclusions.json'
        self._load_exclusions()
        # Per-candidate reuse. One held-out candidate has 21 declared specs (7 classes x 3
        # severities) that share one scene and one authorization decision: only the
        # corruption and the sampler draw differ. Building the scene per example costs
        # ~0.5 s on a large held-out scene, so it is built once per candidate and replaced
        # when the candidate changes.
        self._scene_cache: dict = {}

    def _load_exclusions(self) -> None:
        try:
            payload = json.loads(self._exclusions_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict) and payload.get('partition') == HELD_OUT:
            self.exclusions = {str(k): dict(v)
                               for k, v in (payload.get('exclusions') or {}).items()}

    def _save_exclusions(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        tmp = self._exclusions_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'partition': HELD_OUT,
                                   'exclusions': self.exclusions},
                                  indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(self._exclusions_path)

    def exclusion(self, spec_key: str) -> dict | None:
        return self.exclusions.get(spec_key)

    def __len__(self) -> int:
        return len(self.specs)

    def _geometry_for(self, candidate_id: str):
        cached = self._geometry.get(candidate_id)
        if cached is not None:
            return cached
        # The frozen geometry loader authorises through ``partition_of_candidate``, which
        # refuses a held-out tree by design, so the held-out route loads its own geometry
        # behind the held-out authorization gate. Both the candidate and every context
        # tree must be an authorized held-out tree in the candidate's own frozen context.
        loaded = load_heldout_geometry(self.cache_root, self.provenance, self.view,
                                       candidate_id, self.data_root)
        self._geometry[candidate_id] = loaded
        return loaded

    def preload(self, candidate_ids=None) -> None:
        ids = list(candidate_ids if candidate_ids is not None else self.candidate_ids)
        for candidate_id in ids:
            self._geometry_for(candidate_id)

    def generate(self, index: int) -> GeneratedExample:
        spec = self.specs[index]
        return generate_heldout_example(spec, self._geometry_for(spec.candidate_id),
                                        self.provenance, self.view)

    def evaluated_candidates(self) -> int:
        """How many held-out candidates this instance actually scores."""
        return len(self.candidate_ids)

    def scene_for(self, candidate_id: str):
        """The candidate's held-out scene and its authorized source union, built once.

        The scene depends only on the candidate's frozen geometry, so every spec of the
        candidate shares it. Authorization is re-run whenever the candidate changes, which
        is exactly as often as a new source union appears.
        """
        cached = self._scene_cache.get(candidate_id)
        if cached is None:
            geometry = self._geometry_for(candidate_id)
            scene = build_heldout_scene(candidate_id, geometry, self.provenance, self.view)
            union = scene_module.scene_source_union(scene)
            authorised = authorize_heldout_scene(self.provenance, self.view,
                                                 candidate_id, union)
            if tuple(authorised) != tuple(union):
                raise HeldOutError('STOP_HELDOUT_MANIFEST_CHANGED', {
                    'message': 'held-out source union normalization mismatch'})
            self._scene_cache = {candidate_id: (scene, authorised)}
            cached = self._scene_cache[candidate_id]
        return cached

    def __getitem__(self, index: int) -> HeldOutItem:
        spec = self.specs[index]
        recorded = self.exclusions.get(spec.key)
        if recorded is not None:
            raise NotApplicableExample(spec.key, str(recorded.get('reason')),
                                       recorded.get('detail'))
        scene, authorised = self.scene_for(spec.candidate_id)
        try:
            example = corrupt_scene(spec, scene, self.provenance,
                                    authorised_union=authorised)
        except gen.NotApplicableError as exc:
            # A held-out defect the real geometry cannot support is excluded and counted,
            # exactly as on the model-selection side. It is never substituted, and the count
            # is bounded by the scorer's exclusion limit.
            record = {'reason': exc.reason, 'detail': _jsonable(exc.detail)}
            self.exclusions[spec.key] = record
            self._save_exclusions()
            raise NotApplicableExample(spec.key, exc.reason, record['detail']) from exc
        sample_seed = spec.sampling_seed(self.points, self.seed_base)
        sample = sampler.sample(example.points, example.membership,
                                example.changed_membership, scene.candidate_mask,
                                sample_seed, self.points, spec.repeat)
        provenance = dict(example.provenance)
        provenance['sample'] = {
            'sampler_version': sampler.SAMPLER_VERSION,
            'sampling_seed': int(sample_seed),
            'corruption_seed': int(spec.seed),
            'repeat': int(spec.repeat),
            'points_target': int(sample.diagnostics['points_target']),
            'points': sample.points,
            'source_points': sample.source_count,
            'features': list(sampler.FEATURES),
            'diagnostics': sample.diagnostics,
        }
        provenance['features_digest'] = hashlib.sha256(
            np.ascontiguousarray(sample.features).tobytes()).hexdigest()
        return HeldOutItem(features=sample.features,
                           label=dataset_module.LABEL_INDEX[example.label],
                           key=spec.key, provenance=provenance)

    def digest(self) -> str:
        payload = json.dumps({
            'version': ADAPTER_VERSION,
            'partition': HELD_OUT,
            'points': self.points,
            'repeats': self.repeats,
            'seed_base': self.seed_base,
            'candidate_ids': list(self.candidate_ids),
            'provenance_digest': self.provenance.provenance_digest,
            'identity_manifest_digest': self.provenance.identity_manifest_digest,
        }, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def class_balance(self) -> dict:
        counts: dict[str, int] = {}
        for spec in self.specs:
            key = f'{spec.example_class}|{spec.severity}'
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


class TorchHeldOutDataset:
    """Thin torch adapter, mirroring ``gate_d_dataset.TorchGateDDataset``."""

    def __init__(self, dataset: HeldOutDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        import torch
        item = self.dataset[index]
        return (torch.from_numpy(np.ascontiguousarray(item.features)),
                torch.tensor(item.label, dtype=torch.long), index)


def make_heldout_loader(dataset: HeldOutDataset, batch_size: int, shuffle: bool,
                        seed: int):
    import torch
    from torch.utils.data import DataLoader
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(TorchHeldOutDataset(dataset), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      collate_fn=dataset_module.collate, generator=generator,
                      pin_memory=False, drop_last=False)


# --- adapter parity gate ----------------------------------------------------

def parity_report(provenance, candidate_ids=None, points: int = 4096,
                  classes=gen.CLASSES, severities=gen.SEVERITIES) -> dict:
    """Prove the held-out adapter reproduces the frozen pipeline exactly.

    The same specs are generated twice on real **validation** candidates: once through the
    frozen ``gate_d_generator.generate_example`` (which itself calls the frozen
    ``gate_d_scene.build_base_scene``) via ``gate_d_dataset.GateDDataset``, and once
    through :func:`generate_selftest_example` on a scene produced by the *frozen*
    ``build_base_scene``. Held-out geometry is not involved at any point, and no held-out
    identifier is read beyond the frozen artifact itself.

    Required equalities per example: spec key, corruption seed, sampling seed, label,
    changed-membership count and fraction, sampler pre-reservation and final index digests,
    and the feature tensor bytes.
    """
    validation_ids = list(provenance.candidates(PARITY_PARTITION))
    if candidate_ids is not None:
        chosen = list(candidate_ids)
    else:
        per_candidate = max(1, len(classes) * len(severities))
        count = max(1, min(len(validation_ids),
                           -(-DEFAULT_PARITY_EXAMPLES // per_candidate)))
        chosen = validation_ids[:count]
    for candidate_id in chosen:
        if candidate_id not in validation_ids:
            raise AdapterDrift({
                'message': f'{candidate_id} is not a validation candidate'})

    frozen = dataset_module.GateDDataset(
        dataset_module.DatasetConfig(partition=PARITY_PARTITION, points=points),
        provenance, candidate_ids=chosen)
    adapter_specs = enumerate_specs(chosen, PARITY_PARTITION, classes=classes,
                                    severities=severities)
    if len(adapter_specs) != len(frozen.specs):
        raise AdapterDrift({
            'message': 'spec enumeration length differs from the frozen dataset',
            'adapter': len(adapter_specs), 'frozen': len(frozen.specs)})

    rows: list[dict] = []
    mismatches: list[dict] = []
    for index, adapter_spec in enumerate(adapter_specs):
        frozen_spec = frozen.specs[index]
        frozen_item = frozen[index]
        geometry = frozen._geometry_for(adapter_spec.candidate_id)
        scene = scene_module.build_base_scene(adapter_spec.candidate_id,
                                              PARITY_PARTITION, geometry, provenance)
        port = generate_selftest_example(adapter_spec, scene, provenance)
        sample_seed = adapter_spec.sampling_seed(points, frozen.config.seed_base)
        sample = sampler.sample(port.points, port.membership, port.changed_membership,
                                scene.candidate_mask, sample_seed, points,
                                adapter_spec.repeat)
        frozen_diagnostics = frozen_item.provenance['sample']['diagnostics']
        checks = {
            'spec_key': (adapter_spec.key, frozen_spec.key),
            'corruption_seed': (int(adapter_spec.seed), int(frozen_spec.seed)),
            'sampling_seed': (int(sample_seed),
                              int(frozen_item.provenance['sample']['sampling_seed'])),
            'label': (int(dataset_module.LABEL_INDEX[port.label]), int(frozen_item.label)),
            'provenance_label': (port.provenance['label'], frozen_item.provenance['label']),
            'corruption_class': (port.provenance['class'],
                                 frozen_item.provenance['class']),
            'corruption_severity': (port.provenance['severity'],
                                    frozen_item.provenance['severity']),
            'corruption_mechanism': (port.provenance['corruption']['mechanism'],
                                     frozen_item.provenance['corruption']['mechanism']),
            'changed_points': (int(port.changed_membership.sum()),
                               int(frozen_item.provenance['changed_membership_points'])),
            'changed_fraction': (round(float(port.changed_membership.mean()), 6),
                                 frozen_item.provenance['changed_membership_fraction']),
            'pre_reservation_indices_digest': (
                sample.diagnostics['pre_reservation_indices_digest'],
                frozen_diagnostics['pre_reservation_indices_digest']),
            'final_indices_digest': (sample.diagnostics['final_indices_digest'],
                                     frozen_diagnostics['final_indices_digest']),
            'feature_bytes': (
                hashlib.sha256(np.ascontiguousarray(sample.features).tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(
                    frozen_item.features).tobytes()).hexdigest()),
        }
        failed = {name: {'adapter': values[0], 'frozen': values[1]}
                  for name, values in checks.items() if values[0] != values[1]}
        row = {'spec_key': adapter_spec.key, 'class': adapter_spec.example_class,
               'severity': adapter_spec.severity,
               'candidate_id': adapter_spec.candidate_id, 'agreement': not failed}
        if failed:
            row['mismatches'] = failed
            mismatches.append(row)
        rows.append(row)

    result = {
        'parity_version': PARITY_VERSION,
        'adapter_version': ADAPTER_VERSION,
        'partition': PARITY_PARTITION,
        'candidates': chosen,
        'examples': len(rows),
        'points': int(points),
        'classes': list(classes),
        'severities': list(severities),
        'agreement_count': sum(1 for row in rows if row['agreement']),
        'mismatch_count': len(mismatches),
        'mismatches': mismatches[:20],
        'HELDOUT_ADAPTER_PARITY': 'PASS' if not mismatches else 'FAIL',
        'held_out_geometry_read': False,
        'checked_fields': ['spec_key', 'corruption_seed', 'sampling_seed', 'label',
                           'provenance_label', 'corruption_class', 'corruption_severity',
                           'corruption_mechanism', 'changed_points', 'changed_fraction',
                           'pre_reservation_indices_digest', 'final_indices_digest',
                           'feature_bytes'],
    }
    result['parity_digest'] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(',', ':'),
                   default=str).encode('utf-8')).hexdigest()
    return result


def require_parity(provenance, **kwargs) -> dict:
    """Run the parity gate and refuse held-out when it fails."""
    report = parity_report(provenance, **kwargs)
    if report['HELDOUT_ADAPTER_PARITY'] != 'PASS':
        raise AdapterDrift({
            'message': 'the held-out adapter does not reproduce the frozen pipeline',
            'mismatch_count': report['mismatch_count'],
            'first_mismatches': report['mismatches'][:5],
            'note': 'held-out geometry was never read; nothing was unblinded'})
    return report


def heldout_cache_key(provenance, view: HeldOutView, points: int, repeats: int,
                      seed_base: int, spec_list_digest: str,
                      generator_digest: str, sampler_version: str) -> dict:
    return {
        'cache_version': 'EFG_HELDOUT_CACHE_V1',
        'adapter_version': ADAPTER_VERSION,
        'partition': HELD_OUT,
        'points': int(points),
        'repeats': int(repeats),
        'seed_base': int(seed_base),
        'spec_list_digest': spec_list_digest,
        'generator_digest': generator_digest,
        'sampler_version': sampler_version,
        'provenance_digest': provenance.provenance_digest,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'held_out_candidates': len(view.eligible),
    }
