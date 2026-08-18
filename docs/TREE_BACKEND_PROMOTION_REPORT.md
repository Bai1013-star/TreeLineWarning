# TREE_BACKEND_PROMOTION_REPORT

```text
DEFAULT_TREE_BACKEND = CROWN_STEM_HYBRID
LEGACY_BACKEND_PRESERVED = YES
STEM_INDEPENDENT_TREE_CREATION = NO
SMALL_FRAGMENT_REFINEMENT = ENABLED
CROWN_STEM_VALIDATION = ENABLED
MULTI_STEM_SPLIT = AVAILABLE_BUT_REAL_S005_NOT_YET_PROVEN
FULL_RES_ORIGINAL_INDEX = PASS
MEMBERSHIP_DUPLICATES = 0
TREE_SCIENCE_DEFAULT_CHANGED = YES — owner-authorized backend promotion
AUTHORITATIVE_S005_RERUN = NO
PUSHED = NO
FINAL_STATUS = READY_FOR_DESKTOP_MVP_INTEGRATION
```

`FastBaselineConfig.tree_backend` now defaults to `crown_stem_hybrid`.
`legacy_watershed` remains an explicit selectable fallback/reference. The
canonical runner retains legacy CHM/watershed artifacts under
`legacy_reference`, then uses those crown labels as the input to the promoted
crown-stem refinement before preserving the normal full-resolution
`original_index` back-map and reconciliation contracts.

This promotion is authorized despite no real S005 split event. Stem evidence
cannot create an ordinary tree; it can only refine a crown candidate. No
authoritative S005 rerun was performed.
