# FRV V1 reviewed fixed policy — small governance design amendment

**Status: DESIGN-FROZEN**

**Baseline:** `5564b005518b57be3320eb54eedaba6c1ecaf824` (`feature/algorithm`)

**Targeted independent review:**
`FRV_REVIEWED_FIXED_POLICY_TARGETED_REVIEW: PASS_READY_FOR_FREEZE_AUTHORIZATION`

## 1. Decision and non-goals

This amendment adds the fourth `FrvConfigurationPurpose` value:
`REVIEWED_FIXED_POLICY`.

It means: the exact pure FRV algorithm and membership policy have been design
reviewed, frozen, deterministic, and traceable, but have **not** been
statistically calibrated or independently held-out validated.  It is the
authority for formal FRV review/evaluation artifacts, V4 review/evaluation
handoff, and FRV-VIS Human Gate review.

This is not an FRV rewrite, FRV-B replacement, segmentation change, parameter
schema, scene-specific calibration, or Human-Gate execution.  It changes only
authority, governance, and provenance gates.  The following remain unchanged:

- FRV numerical membership algorithm;
- P-SOURCE, P-HAG, P-GEOMETRY, and P-WIRE algorithms and evidence semantics;
- V4 tree-segmentation Algorithm-V1 and V4 normal/review/excluded/unrecovered
  point semantics;
- all current production-calibration semantics of `PRODUCTION_APPROVED`.

`docs/frv-b-calibration-authority-design.md` remains unchanged and
**KEEP_FOR_FUTURE**.  It is the future stronger, data-calibrated
`PRODUCTION_APPROVED` path; this amendment coexists with it.

## 2. Authority matrix

| Purpose | Calibration required / non-null allowed | Full providers | Exhaustive scope | Complete routing | Formal FRV / artifact | V4 handoff | FRV-VIS | V4 review/evaluation | Production-calibrated claim | Production-certified V4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `TEST_ONLY` | no / yes, but no formal authority follows | no | no | no | no / no | no | no | no | no | no |
| `CALIBRATION_CANDIDATE` | no / yes, but no formal authority follows | no | no | no | no / no | no | no | no | no | no |
| `REVIEWED_FIXED_POLICY` | `None` required / **no** | yes | yes | yes | yes / review-evaluation only | yes | yes | yes, only after eligible Owner decision | no | no |
| `PRODUCTION_APPROVED` | valid non-null SHA-256 / yes | yes | yes | yes | yes / yes | yes | separately authorized | future path, subject to FRV-B verification | only after FRV-B binding | only then and other gates |

`REVIEWED_FIXED_POLICY` must have
`configuration_is_runtime_complete=True`,
`formal_completeness_allowed=True`,
`membership_artifact_publication_allowed=True`, an `EXHAUSTIVE_REGION` scope,
and non-null validated wire-provider interface/configuration binding.  It has
the same hard formal-provider closure as `PRODUCTION_APPROVED`; it differs
only in the deliberately absent calibration record and in its claim boundary.

`TEST_ONLY` and `CALIBRATION_CANDIDATE` retain their existing false closure
booleans and never enter the formal publication path.  `PRODUCTION_APPROVED`
continues to require its non-null valid `calibration_record_digest`; it is not
renamed, weakened, or made optional.

## 3. Configuration and fixed-policy authority

`FrvConfiguration.calibration_record_digest` has a purpose-specific contract:

- `REVIEWED_FIXED_POLICY`: exactly `None`; any SHA-256 or placeholder fails
  validation because the mode explicitly means *not data calibrated*.
- `PRODUCTION_APPROVED`: non-null strict SHA-256, exactly as today.

`FrvMembershipPolicyConfig` remains the sole reviewed-policy authority.  Its
existing `policy_digest` must exactly equal
`FrvConfiguration.feature_policy_digest` before formal execution.  The current
reviewed V1 policy includes `ordinary_min_hag_m=0.30`,
`upper_seed_min_hag_m=1.00`, `ground_max_abs_raw_hag_m=0.10`,
`propagation_min_hag_drop_m=0.05`, `max_structural_rounds=12`, and every
existing policy-contract field.  No second parameters schema, `parameters.json`
authority, calibration payload, or scene override table is introduced.

Every policy-field change changes `policy_digest`, then changes the bound
configuration authority.  A configuration is immutable; no runtime,
per-scene, filename, source-path, source-SHA, `scene_simple`, `scan_small`, or
environment-variable threshold override is allowed after it is constructed.
If Human Gate finds a stable quality problem, the workflow is: revise
policy/algorithm -> obtain a new digest -> review -> rerun Human Gate.  It is
never silent single-scene tuning.

`configuration_digest` continues to be canonical serialization of every
configuration field except itself.  It therefore binds purpose, the exact
`feature_policy_digest`, all provider/configuration authority fields, and the
existing production-relevant payload.  Canonical JSON encodes the reviewed
mode's `calibration_record_digest` as JSON `null`.  Consequently otherwise
identical reviewed-fixed and production-approved configurations have different
digests, and their authority cannot collide.

## 4. Formal gates retained

This amendment weakens none of the following gates: `SourceIdentity`; source
SHA-256; source size/bounds identity; origin binding; P-SOURCE authority;
P-HAG/P-GEOMETRY readiness; P-WIRE authority; provider configuration digests;
scope evidence; `feature_policy_digest`; `configuration_digest`; FRV routing
and membership-decision completeness; V4 routing completeness;
`segmentation_input_complete`; `original_index` uniqueness and conservation;
mutually exclusive route populations; artifact manifest/digest integrity;
atomic publication; strict finite-number JSON; deterministic ordering; no
NaN/Infinity; and no partial success after resource failure.

The minimal FRV membership gate becomes:

```text
configuration_purpose in {REVIEWED_FIXED_POLICY, PRODUCTION_APPROVED}
and runtime/formal/publication booleans are true
and scope is EXHAUSTIVE_REGION
and all existing provider, scope, policy-digest, coverage, conservation,
artifact, and publication checks pass
```

All other purposes fail closed before formal publication.

## 5. Authority propagation and artifact contract

Today a published FRV/V4 root carries `configuration_digest` but not a
recoverable `FrvConfiguration`; a digest alone cannot tell a downstream loader
whether it represents reviewed-fixed or production-approved authority.  The
minimum required new provenance is therefore one mandatory enum field,
`configuration_purpose`, on
`FullResolutionVegetationMembershipArtifact`.

The FRV runtime derives that field from the validated `FrvConfiguration`,
validates it against the formal gate, and includes it in both `artifact_id` and
`manifest_digest`.  The existing `FRVArtifactBindingRecord` already binds the
artifact manifest/semantic ID/configuration digest, so it needs no duplicated
purpose field.  The V4 validated view embeds that artifact and is digest-bound;
the handoff loader recovers authority from that embedded artifact.  Thus the
provenance chain is:

```text
FrvConfiguration.purpose + configuration_digest
  -> formal FRV execution and coverage
  -> membership artifact.configuration_purpose + artifact_id + manifest_digest
  -> existing binding record
  -> V4 validated view (embedded artifact) -> persisted V4 handoff
  -> FRV-VIS and formal V4 runtime
```

This is not a caller assertion: the publisher derives the field from the
validated configuration, and every downstream consumer validates the
digest-bound artifact and equality among artifact, binding, view, coverage, and
handoff identities before using it.  A missing purpose, unsupported purpose,
or a purpose contradicting the formal publisher's configuration fails closed.
Changing a reviewed artifact to `PRODUCTION_APPROVED` changes its semantic and
manifest digests and breaks these bindings unless it is republished through the
production gate, which still requires its calibration digest.  No legacy
artifact is silently interpreted as reviewed-fixed or production-approved.

The membership-artifact and any exact on-disk contract schema that gains this
field must receive a new explicit schema version.  Legacy schemas remain
legacy/read-only where currently supported, but are rejected by the new
authority-required formal V4 path rather than defaulted to either authority.
The existing V4 view need not duplicate the field because its embedded,
validated FRV artifact is the canonical propagated authority.

Reviewed-fixed formal artifacts are labelled **REVIEWED_FIXED_POLICY —
REVIEW/EVALUATION AUTHORITY**.  They may never claim
`PRODUCTION_CALIBRATED`, `DATA_CALIBRATED`, or `HELD_OUT_VALIDATED`; all
structural, digest, and conservation validation remains mandatory.

## 6. V4 and Human Gate boundary

V4 input may be built from a ready reviewed-fixed FRV artifact without changing
any V4 point routing.  The V4 input loader must reject missing/unsupported
upstream authority and bind the recovered artifact purpose to the formal
runtime decision.  `V4ConfigStatus` gains `REVIEWED_FIXED_POLICY` and the
formal `segment_v4_trees` entry guard is:

| Upstream FRV authority | V4 config status | Outcome |
|---|---|---|
| `REVIEWED_FIXED_POLICY` | `REVIEWED_FIXED_POLICY` | review/evaluation execution only, after Owner authorization boundary |
| `REVIEWED_FIXED_POLICY` | `PRODUCTION_APPROVED` | reject |
| `PRODUCTION_APPROVED` | `PRODUCTION_APPROVED` | future production-certified eligibility, still subject to FRV-B and other gates |
| any missing, legacy, test, or calibration authority | any formal status | reject |

Existing V4 `TEST_ONLY` and `CALIBRATION_CANDIDATE` behavior remains unchanged;
they do not make a reviewed-fixed input production-certified.  The guard is
provenance/authority-only: it changes neither Algorithm-V1 nor point meaning.

FRV-VIS must display the digest-bound authority in its HTML, summary JSON, and
checklist, including the explicit no-calibration claim boundary for
`REVIEWED_FIXED_POLICY`.  Its four existing decisions are retained:
`PASS_READY_FOR_V4`, `PASS_WITH_REVIEW_NOTES`,
`FRV_RULE_ADJUSTMENT_REQUIRED`, and `DATA_OR_ARTIFACT_BLOCKED`.

There is currently a checklist generator but no persisted, digest-bound Human
Gate approval artifact or runtime-verifiable Owner decision.  This amendment
does not invent an approval subsystem.  The V1 operational boundary is:

1. FRV-VIS produces the identity-checked HTML/summary/checklist from the
   reviewed-fixed artifact.
2. Owner records one of the four decisions outside the algorithm artifact,
   with the reviewed artifact/configuration digests and the checklist retained
   in the review record.
3. Only `PASS_READY_FOR_V4` or `PASS_WITH_REVIEW_NOTES` authorizes the operator
   to invoke formal V4 **review/evaluation** with matching reviewed authority.
4. The V4 runtime independently verifies artifact authority, completeness, and
   reviewed-vs-production status; it does not falsely claim cryptographic
   knowledge of the Owner decision.

Human Gate never transforms a reviewed-fixed policy into `PRODUCTION_APPROVED`.

## 7. Frozen production implementation scope

| Production file | Why it changes |
|---|---|
| `core/contracts/frv.py` | Add the enum member and reviewed-specific configuration validation; add the single artifact authority field and its digest/schema validation. |
| `algorithms/vegetation/frv_membership.py` | Admit reviewed-fixed in the formal gate and derive/bind artifact authority during publication. |
| `algorithms/vegetation/v4_inputs.py` | Check reviewed/production authority at handoff build and reload; fail closed on missing or contradictory provenance. |
| `core/contracts/v4_tree_segmentation.py` | Add `V4ConfigStatus.REVIEWED_FIXED_POLICY` and its strict status contract. |
| `algorithms/vegetation/v4_tree_segmentation.py` | Add the pre-algorithm authority/status guard for review/evaluation versus production-certified execution. |
| `algorithms/vegetation/frv_v4_visualization.py` | Read validated propagated authority and disclose its label/claim boundary in HTML, summary, and checklist. |

Exactly these six production files are frozen for the amendment.  No new
production module is required.  In particular, P-SOURCE, P-HAG, P-GEOMETRY,
P-WIRE, FRV numerical decision code, and V4 computational code are not
otherwise in scope.

The following production files are
**NOT_REQUIRED_FOR_THIS_AMENDMENT** and must not be modified unless a future
independent blocker disproves this frozen assumption and Owner explicitly
authorizes a scope revision:

| Excluded production file | Frozen reason |
|---|---|
| `core/contracts/frv_a2.py` | Existing required-role logic already handles the reviewed configuration when the required wire binding exists. |
| `algorithms/vegetation/frv_inputs.py` | Its existing non-production limitation branch applies only to `TEST_ONLY` and `CALIBRATION_CANDIDATE`. |
| `core/contracts/v4_inputs.py` | The existing validated V4 view already persists the embedded, digest-bound FRV membership artifact; no duplicate authority field is required. |

## 8. Frozen test scope

| Test file | Required focused coverage |
|---|---|
| `tests/contracts/test_frv_contracts.py` | Reviewed validation with `None`; rejection of non-null digest; unchanged production non-null requirement; purpose/configuration/policy digest separation; artifact authority serialization and legacy fail-closed behavior. |
| `tests/contracts/test_frv_serialization.py` | Membership artifact schema-version migration, authority serialization, digest binding, and legacy read-only versus formal-V4 fail-closed handling. |
| `tests/contracts/test_frv_membership_contracts.py` | `configuration_purpose` artifact contract validation, supported-authority closure, and manifest/semantic digest integrity. |
| `tests/contracts/test_frv_a2_contracts.py` | Reviewed requires all formal roles and can reach complete readiness; test/calibration retain prohibited limitations. |
| `tests/vegetation/test_frv_membership.py` | Test/calibration cannot publish; reviewed can formally execute/publish; policy mismatch and scene-specific override attempts fail; all existing conservation/digest gates remain. |
| `tests/vegetation/test_v4_inputs.py` | Reviewed handoff/reload preserves authority; missing or relabelled authority fails; point routing/conservation remains unchanged. |
| `tests/vegetation/test_frv_v4_visualization.py` | Reviewed authority and no-calibration boundary appear in FRV-VIS outputs/checklist. |
| `tests/vegetation/test_v4_tree_segmentation.py` | Reviewed input plus reviewed V4 status permits the review/evaluation boundary; reviewed cannot use production-certified status; production behavior remains distinct. |

Existing production-contract tests may retain synthetic valid calibration SHA-256
values where record content is outside their scope.  New Human-Gate and
reviewed-runtime tests must use `REVIEWED_FIXED_POLICY` with `None`, never a
fake calibration digest.  Future FRV-B tests will strengthen production record
verification.  Old artifacts are not reinterpreted.

## 9. Operational readiness observations

`scene_simple` is **RUNTIME_ARTIFACT_GENERATION_ONLY** after implementation:
no additional design or algorithm blocker is known.  Required operational work
is to run the stated P-SOURCE -> P-HAG -> P-GEOMETRY -> P-WIRE -> FRV
membership -> V4 handoff -> FRV-VIS Human Gate chain.  This status does not
claim that any of those formal artifacts already exist.

`scan_small` has a present source LAS and the request's verified
P-SOURCE-pass fact is retained.  No contrary formal record was inspected.  Its
separate known prerequisites remain **A1/V1/V2 scope evidence and W1/P-WIRE
authority pending**; they are not generation tasks in this amendment.

## 10. Review checklist

- Claims permitted for reviewed-fixed: reviewed/frozen pure policy; traceable
  policy and configuration digests; validated source/provider identities;
  complete routing and conservation; valid artifact integrity; deterministic
  runtime; Human Gate accepted/rejected; and authorized V4 review/evaluation.
- Claims prohibited: statistical optimization/calibration, independent
  calibration, held-out validation, production accuracy certification, and
  data-calibrated production approval.
- No production code, tests, frozen design, FRV-B design, `.claude/`, formal
  Human Gate result, V4 segmentation algorithm, or Git state is changed by
  this design amendment.

**Final status: DESIGN-FROZEN.**
