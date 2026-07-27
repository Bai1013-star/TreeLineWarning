# V2A directional-evidence and sensitivity correction review

## 1. Baseline and scope

Baseline is `feature/algorithm` at `1c6a72fe59de5139ceef82a97407c61f0f355ed8`. Only the seven approved V2A files changed. V2 remains byte-identical and V2A remains pair-only/review-only.

## 2. Directional continuation

Each cached cluster summary now stores a stable PCA major axis and reliability. Reliability is `(lambda_1-lambda_2)/lambda_1`, bounded [0,1]; fewer than four points, non-finite coordinates, degenerate covariance, or reliability below .20 make the direction unavailable. The directional channel is sign invariant and equals the weaker PCA reliability times a smooth compatibility of the minimum of A-axis/boundary alignment, B-axis/boundary alignment, and mutual-axis alignment. It is independent of vertical separation, which remains its own blocker channel.

## 3. Directional width and profile width

`directional_width` is an active dimensionless tolerance: for raw alignment `q`, transition is `clip(1-(1-q)/directional_width,0,1)`. `profile_width` remains active in production: `clip(1-abs(profile_A-profile_B)/profile_width,0,1)`. Direct production-evaluator tests prove nearby values change their respective channels.

## 4. Unavailability semantics

Unavailable/unstable PCA emits zero directional evidence plus `DIRECTIONAL_EVIDENCE_UNAVAILABLE`; it is not a blocker and does not lower mandatory decision-input completeness. Provenance marks stable A/B directions separately. Sign reversal, uniform pair rotation, and uniform scale changes preserve equivalent directional evidence within test tolerance.

## 5. Configuration-consumption table

| Field | Production consumer | Effect | Direct test/sensitivity |
|---|---|---|---|
| source_id | run/input validation | sibling identity/output name | integration |
| search_multiplier, min/max radius | `_pair_radius`, `_candidate_pairs` | candidate discovery | ±10/20% |
| per_cluster_candidate_cap | `_candidate_pairs` | bounded retained partners | ±1 multi-candidate test |
| boundary_sample_cap | `_boundary` | bounded ordered samples | ±1 sample test |
| scale_ratio_width, density_ratio_width | `_compat` | compatibility channels/blocker | ±10/20% |
| directional_width | `_directional_continuation` | PCA continuation transition | ±10/20%, direct |
| profile_width | `_evaluate_pair` | profile transition | ±10/20%, direct |
| spatial_possible_threshold | `_evaluate_pair` | POSSIBLE gate | ±10/20% |
| possible/strong thresholds | `_evaluate_pair` | status gates | ±10/20% |
| decisive_blocker_threshold | `_evaluate_pair` | explicit-conflict gate | ±10/20% |
| required independent positives | `_evaluate_pair` | STRONG rule | ±1 direct |
| low_support_requirement | `_low_support` | LOW evidence scaling | ±1 direct |
| record_memory | run/statistics | RSS collection | integration |

## 6. Continuous and discrete sensitivity

Production sensitivity covers default, -10%, +10%, -20%, +20% for search multiplier, both search bounds, scale/density widths, directional/profile widths, spatial transition, possible/strong thresholds, and blocker threshold. Discrete neighboring values cover candidate cap, boundary cap, independent-positive requirement, and LOW-support requirement. Candidate settings invoke `_candidate_pairs`; boundary settings invoke `_boundary`; evidence/status settings invoke `_evaluate_pair`. Unsafe families remain non-STRONG under every tested setting.

## 7. Smoke matrix

The exact smoke matrix remains 1,296 cases. Every family has 108 cases, all factor levels, and each of the six occlusion×noise combinations exactly 18 times.

## 8. Full 23,328-case matrix

Every case generates factor-sensitive geometry/features and invokes the production evaluator. Per-family A/P/S/B; confidence p10/median/p90; directional p10/median/p90:

| Family | A/P/S/B | Confidence | Directional |
|---|---|---|---|
| uniform crown | 0/1430/514/0 | .681/.733/.823 | .243/.700/.938 |
| density transition | 1403/529/0/12 | .394/.458/.539 | .000/.609/.817 |
| occlusion branch | 68/1876/0/0 | .512/.578/.670 | .790/.916/.949 |
| sparse tall | 1529/292/0/123 | .349/.413/.492 | .000/.000/.631 |
| compact dense | 0/0/0/1944 | .276/.282/.334 | .000/.000/.101 |
| touching crowns | 12/0/0/1932 | .215/.247/.294 | .000/.306/.753 |
| crown/wall | 0/0/0/1944 | .215/.238/.288 | .000/.000/.429 |
| crown/pole | 0/0/0/1944 | .256/.274/.325 | .000/.000/.066 |
| vertical overlap | 0/0/0/1944 | .225/.225/.275 | .000/.000/.000 |
| sloped HAG | 488/1450/0/6 | .434/.531/.625 | .000/.607/.822 |
| LOW-only bridge | 0/0/0/1944 | .236/.294/.359 | .123/.653/.878 |
| anisotropic scanner | 668/1246/0/30 | .413/.502/.599 | .000/.371/.723 |

Decision-input completeness is 1.0 in all generated rows. Direction availability is recorded through A/B provenance flags; optional unavailable HAG occurs in 1,296 generated rows per family. Touching, wall, pole, vertical overlap, and LOW-only bridge never reach STRONG. All four statuses are reachable.

## 9. Real-scene aggregate regression

Read-only V2A regeneration produced 443 pairs: 19 ABSTAINED, 1 POSSIBLE, 0 STRONG, 423 BLOCKED. Directional p10/median/p90 is .0000/.0000/.3618; stable-direction availability is 437 A / 428 B. Profile p10/median/p90 is .0000/.5562/1.0000. Explicit blockers: physical 314, vertical 85, touching 28, planar 0, slender 15, compact 0, mixed 0, reliable HAG 273, scale/density 89, contradiction 0. Lineage unavailable: 337. All 443 have decision-input completeness 1.0.

## 10. Publication, invariance, and performance

Publication remains temporary validated NPZ then JSON completion marker, with sibling hash/reload validation and cleanup. V2 NPZ/JSON remain `95d40327417bcdb6f760e97890a71bfabe3b49d85949d2104fc75518a48a9772` / `c600bf8b672a0edcf9efcbdd84c28c9109ccffd4788ce8daef88f18764f26e06`. V2A NPZ/JSON are `cc02d9c4e259427e16d8f72bb4cc2217defada7cf2a3ad1d0159c7eaa1e4b0e3` / `dd6137210091abe41dfc35fe5beab6c8c74fd2c69821e9cd887bb803d87c8f22`. Runtime is 20.762018 s; RSS start/peak/end 66,236,416 / 122,351,616 / 122,351,616 bytes. PCA caching adds summary work (.063684 s); evidence/status work is 20.302197 s and remains bounded by pair and boundary caps.

## 11. Tests, limits, and safety

The focused suite proves PCA direction, sign/rotation/scale invariance, instability behavior, configuration consumption, candidate/boundary/LOW discrete consumers, complete smoke coverage, full matrix, sensitivity, and V2 hashes. V2A has no labels and makes no biological identity claim; it has no groups, merges, manual state, C1/R1/P1 effect, or broad-generalization claim. No staging, commit, push, merge, reset, cleanup, LAS/LAZ, or A3 access occurred.
