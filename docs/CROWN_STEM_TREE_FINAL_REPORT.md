# CROWN_STEM_TREE_FINAL_REPORT

> Historical read-only comparison. Owner subsequently authorized canonical promotion; see [TREE_BACKEND_PROMOTION_REPORT.md](TREE_BACKEND_PROMOTION_REPORT.md).

## Decision

`CROWN_GENERATED + STEM_VALIDATED` is implemented as a parallel refinement in `algorithms/spans/stem_guided_hybrid.py`. CHM/tree-top/saddle/watershed membership remains the sole normal tree-candidate generator. A stem does not create a normal tree; it can validate a crown, support conservative attachment, or participate in a split only when separately supplied CHM structural evidence reports multiple crown peaks.

```text
BASELINE_CROWN_CANDIDATES = 31
BASELINE_FINAL_TREES = 22
RAW_STEM_CANDIDATES = 267
DEDUPLICATED_STEM_HYPOTHESES = 37
VALIDATED_INDEPENDENT_STEMS = 26
HYBRID_CONFIRMED_TREES = 25 (per-span crown candidates; not reconciled/persistent entities)
CROWN_ONLY_REVIEW = 0
RECOVERY_REVIEW = 0
SMALL_FRAGMENTS_REJECTED = 0
NO_CROWN_REJECTED = 0
FRAGMENTS_ATTACHED = 6
MULTI_STEM_SPLIT_EVENTS = 0
FALSE_STEM_CONTROL = PASS
STEM_DEDUP = PASS
SMALL_REAL_TREE_PROTECTION = PASS (synthetic)
SMALL_FRAGMENT_REJECTION = PASS (synthetic)
MULTI_STEM_SPLIT = PASS (synthetic)
CROWN_ONLY_REVIEW_GATE = PASS (synthetic)
STEM_ONLY_REJECTION = PASS (synthetic)
ORIGINAL_INDEX = PASS (actual S005 read-only back-map: 20,838,455 assigned raw rows; 20,838,455 unique original_index values)
MEMBERSHIP_DUPLICATES = 0
DETERMINISM = PASS (synthetic repeated run)
REAL_FRAGMENT_REDUCTION = PASS at local-crown-candidate level (31 labels -> 25 confirmed + 6 attached)
REAL_OVERMERGE_REDUCTION = NOT_DEMONSTRATED
NO_SCENE_HARDCODE = PASS
RUNTIME = detection 1.77 s; dedup 0.19 s; crown validation/split 0.66 s; total 2.63 s on 588,500 fine representatives; full raw back-map 20.59 s for 52,477,380 rows
VISUALIZATION = cache/debug/crown_stem_final_s005/CROWN_STEM_FINAL_COMPARISON_S005.html
CANONICAL_PROMOTION = NO
OLD_BACKEND_PRESERVED = YES
AUTHORITATIVE_RUN_CREATED = NO
REGISTRY_CHANGED = NO
COMMIT = NONE
PUSHED = NO
FINAL_STATUS = KEEP_CURRENT_TREE_BACKEND
```

## What the read-only comparison demonstrates

The prior failure was 267 raw vertical columns becoming 262 confirmed trees. The final candidate route preserves the 267 raw columns only as diagnostic evidence, deduplicates them inside their owning crown footprints, and produces no stem-created trees. It does not split S005 because the read-only candidate call has no additional CHM multi-peak evidence per watershed crown; this is intentional safety, rather than treating stem columns alone as separability proof.

The six weak crown candidates are attached only after an actual 3-D contact test to a confirmed crown. They are no longer independent confirmed observations. Persistent identity and display ordering remain owned by the existing reconciliation stage and were not changed.

## Promotion blocker

Real S005 over-merge reduction is not demonstrated: zero split events is safer than the previous 262-tree failure, but does not establish a visually verified true multi-trunk split. Canonical promotion requires that missing evidence, plus a viewer sourced with tower geometry, before an authoritative S005 rerun can be recommended. No parameter sweep or scene-specific tuning was performed.
