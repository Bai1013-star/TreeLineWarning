# STEM_GUIDED_HYBRID_TREE_REPORT

> Superseded by [CROWN_STEM_TREE_FINAL_REPORT.md](CROWN_STEM_TREE_FINAL_REPORT.md). This document records the rejected stem-created prototype only.

```text
TRUNK_SIGNAL_AVAILABLE = YES (lower/mid HAG occupancy), BUT NOT DISCRIMINATIVE ENOUGH
VALIDATED_STEM_COUNT = 267
BASELINE_TREE_COUNT = 22
HYBRID_CONFIRMED_TREE_COUNT = 262
CROWN_ONLY_REVIEW_COUNT = 5
REJECTED_SMALL_FRAGMENT_COUNT = 0
REJECTED_NO_CROWN_COUNT = 5
MULTI_STEM_SPLIT_COUNT = 22
SMALL_REAL_TREE_PROTECTION = PASS (synthetic)
MULTI_STEM_SPLIT = PASS (synthetic)
STEM_ONLY_REJECTION = PASS (synthetic)
CROWN_ONLY_REVIEW = PASS (synthetic)
FULL_RES_ORIGINAL_INDEX = PASS (synthetic raw-to-fine-to-raw regression; candidate runner reuses exact contract)
MEMBERSHIP_DUPLICATES = 0
DETERMINISM = PASS (synthetic repeated run)
REAL_FRAGMENT_REDUCTION = FAIL
SCENE_HARDCODE = NO
BASELINE_VS_HYBRID = 22 reconciled baseline trees versus 262 hybrid confirmed candidates; hybrid is grossly over-segmented
VISUALIZATION = cache/debug/stem_guided_hybrid_s005/STEM_GUIDED_TREE_COMPARISON_S005.html
TREE_CORE_PROMOTION = NO
OLD_TREE_BACKEND_PRESERVED = YES
AUTHORITATIVE_RUN_CREATED = NO
REGISTRY_CHANGED = NO
COMMIT = NONE
PUSHED = NO
FINAL_STATUS = KEEP_CURRENT_TREE_BACKEND
```

The read-only artifact report with individual candidates and stem evidence is generated beside the viewer as `STEM_GUIDED_HYBRID_TREE_REPORT.json`.

Critical comparison answers: the candidate cannot credibly confirm or demote any of the 22 reconciled trees, because its 262 confirmations are already disproportional. It reports 22 local multi-stem split attempts, but none is accepted as a real split. Likewise, it does not establish a safe fragment merge/removal on S005. These are deliberately reported as **unresolved**, not converted into an unsupported tree-by-tree verdict.

The sole blocker is robustness: the available stem proxy accepts 267 narrow, vertically persistent columns in S005, far beyond the 22 baseline trees. Promoting it would create the exact false-tree behavior this task aims to remove. No second parameter-tuning loop was performed.
