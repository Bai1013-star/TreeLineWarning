# STEM_GUIDED_DESIGN_AUDIT

Historical audit of the initial stem-created candidate experiment on the
read-only `G002 / S005 / s005-c96b79b5d7f04e46a4258c17bd3cdaa8` artifacts.
Its original non-promotion conclusion applies only to that experiment. It is
superseded by the owner-authorized crown-generated / stem-validated promotion
documented in `CROWN_STEM_TREE_FINAL_REPORT.md` and
`TREE_BACKEND_PROMOTION_REPORT.md`.

## Existing path retained as baseline

The retained legacy reference path in `algorithms/spans/tree_runner.py` performs high-recall vegetation recovery, 0.10 m fine representatives, HAG estimation, CHM construction, smoothing, tree-top/saddle evidence, marker-controlled watershed, 3-D fine membership, and exact raw-row back-mapping through `original_index`. `algorithms/spans/reconcile.py` then performs CORE ownership, cross-span reconciliation, display ordering, and persistent identity handling. The promoted crown-primary hybrid reuses these contracts and does not assign persistent IDs.

## Signal audit

| Question | Finding |
| --- | --- |
| A. Lower/mid vegetation signal | **Yes, numerically.** The fine artifact has 351,965 representatives (28,430,682 raw-point weight) in HAG 0.5--5.5 m. This is enough to attempt a stem detector, but not evidence that those points retain botanically separable trunk surfaces. |
| B. HAG height slicing | **Yes.** `fine_hag_m` is aligned one-to-one with `fine_original_index`, finite for all 588,500 fine representatives, and spans -0.339 to 12.841 m. |
| C. Classification/intensity/returns | **No useful auxiliary signal in the candidate artifact.** `classification` is 0 for all 588,500 fine representatives; intensity and return attributes are not preserved in the fine/tree artifacts. |
| D. Best representation for stems | Fine representatives are the appropriate detection representation: 0.10 m spatial support plus `raw_count`, HAG, and `original_index`. Full-resolution rows remain the membership/back-map representation. |
| E. Existing error evidence | The local baseline contains 31 watershed labels before reconciliation's 22 CORE-owned trees. Several labels have one crown cell / very low fine membership, consistent with crown fragments. The artifacts alone do not establish botanical truth for every label. |
| F. `original_index` preservation | **Yes.** The candidate runner calls the existing `map_region_rows_to_tree`, and a synthetic raw-to-fine-to-raw regression verifies exclusive source-row routing. |

## Historical candidate experiment and decision

The initial parallel `STEM_GUIDED_HYBRID` experiment used lower/mid HAG slices, per-column XY persistence, vertical extent/gaps, PCA verticality, cross-section radius, weighted support, and deterministic seed IDs. Its failure was specifically that stems could independently create ordinary candidate trees.

On S005 it produced **267 validated stem hypotheses** and **262 confirmed candidates** from a 31-label watershed baseline. This was implausible and demonstrated that fine vegetation occupancy/PCA alone cannot distinguish trunk shafts from vertically persistent foliage/scan structures in this input.

That failure was corrected by changing the science contract, not by tuning S005:
the current default backend creates normal trees only from watershed crown
candidates, conditions stem hypotheses on those crowns, and allows stems only
to validate, refine, attach/reject small fragments, or provide gated split
evidence. It does not rerun the authoritative S005 case.
