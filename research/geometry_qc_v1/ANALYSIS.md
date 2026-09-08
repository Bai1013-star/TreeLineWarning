# Geometry feature analysis

No trustworthy explicit-good distribution is available. The provisional-control column
below retains the prior manifest's eight good claims for exploration ONLY; all eight
are UNCERTAIN_REFERENCE in definitive metrics. No production accuracy is calculated.

| Feature | Explicit good | Provisional controls min / median / max | Bad min / median / max |
|---|---|---|---|
| point_count | unavailable | 106330 / 482209 / 704065 | 13997 / 1.20791e+06 / 4.19732e+06 |
| sample_count | unavailable | 16384 / 16384 / 16384 | 13997 / 16384 / 16384 |
| sample_z_extent_m | unavailable | 5.7047 / 7.54285 / 8.2294 | 4.3642 / 7.0493 / 8.4195 |
| robust_z_extent_m | unavailable | 3.90488 / 5.1567 / 6.87614 | 3.79642 / 5.12884 / 6.01732 |
| width_major_m | unavailable | 2.54373 / 2.97161 / 3.72644 | 4.03336 / 8.20807 / 21.5742 |
| width_minor_m | unavailable | 1.92108 / 2.46981 / 3.51258 | 3.91598 / 4.72608 / 6.50272 |
| width_aspect | unavailable | 1.06088 / 1.17166 / 1.37083 | 1.02997 / 1.56188 / 4.51163 |
| width_height_ratio | unavailable | 0.372491 / 0.66303 / 0.723861 | 0.837097 / 1.4414 / 4.89362 |
| xy_minor_major_variance | unavailable | 0.400912 / 0.631233 / 0.770752 | 0.0427754 / 0.365955 / 0.774251 |
| empty_vertical_bin_fraction | unavailable | 0 / 0 / 0 | 0 / 0 / 0 |
| max_vertical_gap_fraction | unavailable | 0.00186665 / 0.022418 / 0.0587407 | 0.00274701 / 0.0124387 / 0.084472 |
| component_16_largest | unavailable | 1 / 1 / 1 | 0.988098 / 1 / 1 |
| component_16_second | unavailable | 0 / 0 / 0 | 0 / 0 / 0.0115967 |
| component_16_singleton | unavailable | 0 / 0 / 0 | 0 / 0 / 0 |
| component_32_largest | unavailable | 0.993774 / 0.999878 / 1 | 0.919983 / 0.998169 / 1 |
| component_32_second | unavailable | 0 / 6.10352e-05 / 0.00610352 | 0 / 0.00125122 / 0.0785168 |
| component_32_singleton | unavailable | 0 / 3.05176e-05 / 0.000183105 | 0 / 3.05176e-05 / 0.000427246 |
| component_64_largest | unavailable | 0.91864 / 0.987152 / 0.992798 | 0.51161 / 0.803192 / 0.993958 |
| component_64_second | unavailable | 0.00128174 / 0.00747681 / 0.0402222 | 0.00292969 / 0.186096 / 0.361023 |
| component_64_singleton | unavailable | 0 / 0.00088501 / 0.0012207 | 0.00012207 / 0.000835801 / 0.00360107 |

## Existing segmentation evidence (original field memberships)

| Case | Raw columns | Deduplicated hypotheses | Retained independent stems | CHM peak field |
|---|---:|---:|---:|---:|
| FIELD_BAD_T000 | 15 | 2 | 1 | 1 |
| FIELD_BAD_T001 | 69 | 2 | 1 | 1 |
| FIELD_BAD_T005 | 15 | 4 | 1 | 1 |
| FIELD_BAD_T009 | 7 | 1 | 1 | 1 |
| FIELD_BAD_T010 | 15 | 3 | 1 | 1 |
| FIELD_BAD_T012 | 13 | 3 | 1 | 1 |
| FIELD_BAD_T013 | 23 | 2 | 1 | 1 |
| FIELD_BAD_T014 | 9 | 1 | 1 | 1 |
| FIELD_GOOD_T002 | 5 | 1 | 1 | 1 |
| FIELD_GOOD_T003 | 6 | 1 | 1 | 1 |
| FIELD_GOOD_T004 | 2 | 1 | 1 | 1 |
| FIELD_GOOD_T006 | 3 | 1 | 1 | 1 |
| FIELD_GOOD_T007 | 8 | 1 | 1 | 1 |
| FIELD_GOOD_T008 | 6 | 1 | 1 | 1 |
| FIELD_GOOD_T011 | 7 | 1 | 1 | 1 |

## Case-by-case diagnostic results

This is the predeclared two-scale diagnostic hypothesis, not accepted product QC.

| Tree | Reference | QC result | Issue | Reason | Match / miss |
|---|---|---|---|---|---|
| FIELD_BAD_T000 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T001 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T005 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T009 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T010 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T012 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T013 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_BAD_T014 | EXPLICIT_BAD | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | MISS (abstention) |
| FIELD_GOOD_T002 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T003 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T004 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T006 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T007 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T008 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| FIELD_GOOD_T011 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| ACCEPT_GOOD_T001 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | NO_VALIDATED_PASS_CRITERION | NOT_SCORED |
| ARCHIVE_UNCERTAIN_T012 | UNCERTAIN_REFERENCE | UNCERTAIN | GEOMETRY_AMBIGUOUS | REFERENCE_GEOMETRY_UNAVAILABLE | NOT_SCORED |

## Sensitivity checks, not threshold selection

All use the same unvalidated 10% material-component assumption. No PASS follows from
a missing signal. The controls here remain provisional.

| Signal | Bad signaled / 8 | Provisional controls signaled / 8 |
|---|---:|---:|
| Second sampled component >= 10%, diagonal/16 voxels | 0 | 0 |
| Second sampled component >= 10%, diagonal/32 voxels | 0 | 0 |
| Second sampled component >= 10%, diagonal/64 voxels | 5 | 0 |
| Second sampled component >= 10% at BOTH /32 and /64 | 0 | 0 |

Fine-scale component evidence overlaps multiple deduplicated stem hypotheses for
T000/T005/T010/T013 (4/8 bad). Multiple hypotheses alone signal 6/8 bad and 0/7
field provisional controls, but are not independently validated stems. T009/T014
have only one hypothesis; T001 has almost all sampled points connected. These are
promising research signals with important misses; validation remains incomplete.

## Rechecked visual provenance

- FIELD_BAD_T000: contact-sheet SHA-256 `cc7d38596fb0c2257dbeca7ade87e15338ce61335db177225900027004e92cf4` verified;
- FIELD_BAD_T001: contact-sheet SHA-256 `f3975a4a1ab00c7b0df9230722cca26542f766151fac548c08b063cb298a6c1e` verified;
- FIELD_BAD_T005: contact-sheet SHA-256 `5f4259f82a3ceae419b87f375a0e8f75d564724c7f5a8d469a00189c0423ea8e` verified;
- FIELD_BAD_T009: contact-sheet SHA-256 `9ed007e2e264bee3e916ccbc6d42a8a32db7c8d98a85cd685dcc71a0f07da10a` verified;
- FIELD_BAD_T010: contact-sheet SHA-256 `f38add1b2dfd50c0e894406d7d2f83d81cdf35841f67a0b17fdf6a945ffe4dde` verified;
- FIELD_BAD_T012: contact-sheet SHA-256 `6b065626c85eed619f60ec6f7f849e802597f2a45906d0334a6d8681ddcddf70` verified;
- FIELD_BAD_T013: contact-sheet SHA-256 `224d2f372811db435301e3d76b656c68babd9e6e49357c275f7f952a375ff7bc` verified;
- FIELD_BAD_T014: contact-sheet SHA-256 `845f6e99a5dee7289cd53ef15af301415dcf02837b9a5dea54afa3c0b7bbf3b9` verified;
