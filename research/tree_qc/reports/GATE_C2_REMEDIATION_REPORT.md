# Gate C2 minimal remediation — source isolation and high-containment quarantine

Worker: Claude Code (DeepSeek). Worker report only; GPT-6 Supervisor decides.
Baseline `START_HEAD = 856440577ecb9dbaeafdf1a45f35c6527b6340d0`.

Canonical identity was **not** redesigned. Gate A/B were not rerun, no data were
redownloaded, no 163M-point scan was repeated, no model was trained, and the frozen
V0.1 was not touched. All numbers below are re-derived from the frozen manifest,
membership tables and context index.

## 1. Blockers as found by the Supervisor

Both were reproduced independently from raw manifest rows before any fix:

| Finding | Reproduced |
|---|---|
| canonical trees used as context on both sides | **5**, the same five (`tile_60_-20_Boom2`, `tile_60_-40_Boom20`, `tile_60_-40_Boom6`, `tile_60_-80_Boom13`, `tile_60_-80_Boom9`) |
| high-containment pairs / trees | 17 / **25** |
| previous strict pool | **749** |
| previous candidates referencing the quarantine set | **200** |

`tile_60_-20_Boom2` was context for 1 training candidate and 2 validation candidates,
matching the Supervisor's example.

## 2. Remediation R1 — complete example source sets

`EXAMPLE_SOURCE_SET = candidate ∪ candidate aliases ∪ context trees ∪ context aliases
∪ donor trees ∪ donor aliases`, evaluated over whole canonical trees.

Deterministic policy (fixed before any model result):

1. a quarantined source contributes to no partition;
2. a candidate whose surviving context is empty is excluded (fail closed);
3. a source still in both unions is removed from the side where it is not a candidate,
   or from both sides when it is a candidate on neither; it is never removed from the
   side where it is a candidate;
4. iterate to a fixpoint; non-convergence raises instead of passing;
5. donors may only come from the candidate's own surviving context.

All five conflicting sources are Litchfield **boundary-quarantine** trees — candidates
in neither partition — so rule 3 removed them from both sides. The 3 m radius and all
geometry are unchanged.

| Assertion | Value |
|---|---:|
| `TRAIN_VAL_CANDIDATE_LEAKAGE` | 0 |
| `TRAIN_VAL_CONTEXT_LEAKAGE` | 0 |
| `TRAIN_VAL_DONOR_LEAKAGE` | 0 (no generator yet; donors ⊆ context) |
| `TRAIN_VAL_ALIAS_LEAKAGE` | 0 |
| `TRAIN_VAL_ALL_SOURCE_LEAKAGE` | 0 |
| `TRAIN_HELDOUT_ALL_SOURCE_LEAKAGE` | 0 |
| `VALIDATION_HELDOUT_ALL_SOURCE_LEAKAGE` | 0 |

Source unions: train 482, validation 207, held-out 851 — pairwise disjoint.

Train 442 partition members → **440** eligible (2 excluded for base eligibility);
validation 177 → **175** (2 excluded). No candidate was lost to source isolation.

## 3. Remediation R2 — high-containment quarantine

`HIGH_CONTAINMENT_QUARANTINE_SET` = the 25 canonical trees of the 17 pairs with
containment ≥ 0.30, all Wytham. Publisher labels unchanged; no pair unioned.

Every held-out candidate's context was rebuilt without those 25 and re-evaluated.

| Assertion | Value |
|---|---:|
| `HIGH_CONTAINMENT_AS_CANDIDATE` | 0 |
| `HIGH_CONTAINMENT_AS_CONTEXT` | 0 |
| `HIGH_CONTAINMENT_AS_DONOR` | 0 |
| `HIGH_CONTAINMENT_AS_ALIAS_SOURCE` | 0 |

| Strict held-out pool | Value |
|---|---:|
| previous (withdrawn as a frozen number) | 749 |
| **re-derived from clean provenance** | **749** |
| candidates dropped for lack of clean context | 0 |
| candidates whose context provenance changed | **200** |
| 25 quarantine trees excluded as candidates | 25 |

The count is unchanged because every one of the 200 affected candidates retains a real
point-bearing context after the quarantine is removed — the minimum surviving context
size is 5, the median 21, the maximum 64, and none reaches zero. The pool is therefore
re-derived, not assumed: the candidate set is identical, but the context provenance of
exactly those 200 candidates is now quarantine-free.

Minimum support stays at **30**; the strict pool holds 749 ≥ 30, so no HOLD is
triggered on support grounds.

## 4. Independent worker check

`GATE_C2_REMEDIATION.json → independent_reconstruction` rebuilds each partition's
complete source union directly from raw manifest `neighbour_ids` rows and re-derives
the intersections:

```
train_union_size 482   validation_union_size 207   held_out_union_size 851
train∩validation 0     train∩held-out 0            validation∩held-out 0
quarantine_in_held_out_union 0                     matches_reported true
```

A separate scratch re-derivation (not the audit code) reproduced the same unions, the
same zero intersections, the absence of the five resolved sources from both sides, and
the 200-candidate context change.

## 5. Tests and lint

`python -m pytest test_canonical_identity.py` — **30 passed** (19 pre-existing +
11 new). New regression coverage, all fail-closed:

1. shared context source removed from both sides;
2. train candidate used as validation context;
3. validation candidate used as train context;
4. donor crossing partitions removed;
5. alias never crosses partitions (with a positive control);
6. shared quarantined neighbour excluded, and quarantine-only context fails closed;
7. zero-point accidental provenance fails closed;
8. held-out rebuild drops quarantined context and keeps base blocker reasons;
9. candidate with only shared context fails closed.

`python -m ruff check` on the changed files — all checks passed.

`GATE_C2_VERIFICATION.json` — **12 checks, ALL PASS** (8 identity + 4 remediation).

## 6. Artifacts

- `GATE_C2_REMEDIATION.json` — policy, quarantine set, resolutions, counts, assertions,
  independent reconstruction
- `GATE_C2_CONTEXT_PROVENANCE.json` — per-candidate context/donor provenance, blocked
  sets, exclusion reasons for train, validation and held-out
- `GATE_C2_VERIFICATION.json` — extended verification
- `GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2.md` — §9 records the isolation policy

Identity results are untouched: 1,693 occurrences, 1,579 canonical trees (703 / 876),
113 exact-alias groups, digest
`eb515773b5235f2cdb0206231493195f73f269c761d9da2d2186ba68ab3ed5aa`.

## 7. Verdict

`FINAL_WORKER_VERDICT = GATE_C2_WORKER_READY_FOR_SUPERVISOR_REVIEW`.

Both blockers are closed with deterministic, documented, fail-closed policies; the
strict held-out pool is re-derived at 749 with clean provenance; minimum support 30 is
preserved. Gate D/E training was **not** started. Acceptance remains the Supervisor's
decision.
