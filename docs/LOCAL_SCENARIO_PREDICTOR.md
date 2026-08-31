# Local scenario geometry preview

`core.future_preview` contains a small, CPU-only preview for a finalized tree
geometry.  It is an overlay contract for a viewer and does not change the
current survey result, tree membership, exports, or persisted project state.

```python
from core.future_preview import FinalizedTreeGeometry, LocalScenarioPredictor

tree = FinalizedTreeGeometry(
    persistent_tree_id="tree-17",
    span_id="span-A",
    xyz_local_m=display_points_local_m,
    height_m=12.4,
    crown_major_m=4.8,
    crown_minor_m=3.2,
    dbh_cm=None,
    anchor_xyz_local_m=full_tree_anchor_local_m,
)
prediction = LocalScenarioPredictor().predict(tree, horizon_years=3)
future_points = prediction.xyz_local_m
```

The public input is `FinalizedTreeGeometry` with persistent tree and span
identities, an XYZ point or `(N, 3)` local sample, nullable canonical height
and crown dimensions, nullable DBH in centimetres, and an optional local XYZ
anchor.  The frontend or desktop adapter owns the persisted finalized gate and
must construct this object only after that gate passes.  A bounded display
sample is suitable for the geometry input; canonical metrics remain separate
and are never estimated from that sample.  The optional anchor is normally a
full-tree XY mean and Z minimum calculated before sampling.

`LocalScenarioPredictor.predict(tree, horizon_years)` accepts only horizons 1,
3, and 5.  It returns `FutureTreeGeometryPrediction` with the parent
identities, `predicted_xyz_local_m` (also exposed as `xyz_local_m`), nullable
predicted metrics, `growth_direction_xy`, `prediction_mode="LOCAL_SCENARIO"`,
the model and version, `experimental=True`, assumptions, and safety tags.  No
`original_index` or current-membership field is produced.  Input and output
arrays are detached, immutable bytes-backed NumPy views.  The prediction
contract also leaves room for explicitly labelled `LOCAL_MODEL` or
`FORMAL_EXTERNAL` implementations; this package implements only
`LOCAL_SCENARIO`.

The preview uses deterministic NumPy operations only.  For a sample with
planar spread it obtains a sign-stabilized PCA major axis and a perpendicular
minor axis.  Crown major and minor coordinates are scaled around the supplied
anchor; height is scaled around the anchor Z or the sample Z minimum.  Point
order is preserved, and translating both the sample and anchor translates the
preview by the same amount.  A sample with insufficient planar spread uses
canonical XY axes and records that assumption explicitly.  The configured
`growth_direction_xy` is a local XY displacement in metres per year and is
zero by default.

`LocalScenarioConfig` defaults are illustrative bounded increments of 0.20 m
per year for height, 0.15 m per year for crown major, and 0.10 m per year for
crown minor, with a 1.50 maximum dimension scale.  These values are a stable
geometry-scaling scenario, not biological parameters.  If a canonical metric
is missing, the configured canonical dimension is used only to scale the
synthetic geometry; the corresponding predicted metric remains `None`.  A
supplied DBH is carried through unchanged, and missing DBH stays `None`.

Every prediction carries `SYNTHETIC_FUTURE_GEOMETRY`, `EXPERIMENTAL`,
`NOT_CURRENT_SURVEY_SCIENCE`, and
`NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST`.  The current survey geometry
must remain the authoritative viewer layer; future points should be rendered
as a separate, clearly labelled display-only layer.  This module has no
network, GPU, SoftwareB, external model, biological calibration, or long-term
forecast validation path.
