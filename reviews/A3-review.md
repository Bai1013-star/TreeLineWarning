# A3 Review: Rule/Manual Wire Candidate Baseline

## Scope and baseline

- Branch: `feature/algorithm`.
- Baseline: `d267762 feat: add grid ground estimation and HAG baseline`.
- No commit, push, merge, A4 work, LAS/LAZ read, or A1/A2A/A2B rerun was performed.
- Modified implementation/test files:
  - `core/contracts/candidates.py`
  - `core/contracts/__init__.py`
  - `algorithms/candidates/wire.py`
  - `algorithms/candidates/__init__.py`
  - `tests/contracts/test_candidate_contracts.py`
  - `tests/candidates/test_wire.py`

## Contracts and output

- Added fixed `CandidateReasonFlag`, `WireCandidateQualityFlag`, `CandidateLevel`, and
  `CandidateSource` enums with the approved uint16/uint8 values.
- Added `WireCandidateConfig`, `ManualSelectionConfig`, `CandidatePointSet`,
  `CandidateEvidenceSet`, `WireCandidateStatistics`, and `WireCandidateResult`.
- All A3 per-point arrays are full-length NPZ arrays aligned to A1 `xyz_local`,
  `original_index`, and `sample_index`; contracts enforce shape, dtype, finite values,
  index uniqueness, and candidate mask/level/source agreement.
- JSON keeps only provenance, hashes, configuration, statistics, formulas, and array
  metadata. Paths are safely serialized as strings and restored as `Path` objects.

## Input validation and scoring

- `run_wire_candidate_selection()` loads only the six A1/A2A/A2B NPZ/JSON artifacts.
  It validates paths, source ID, manifest source ID, origin, A1/A2A/A2B index alignment,
  A1/A2A XYZ alignment, required dtypes/shapes, finite valid HAG, and the complete A1 ->
  A2A -> A2B SHA-256 provenance chain.
- No LAS/LAZ API is imported or called. The implementation uses vectorized M-length arrays
  only and does not construct an N x N distance matrix.
- Guard: `linearity >= 0.60`, `scattering <= 0.10`, and no A2A `NUMERIC_INVALID` flag.
- Score components are the approved linearity, scattering, low-planarity, HAG/local-height,
  and density scores. The normalized weights are 0.35/0.25/0.10/0.20/0.10. Rule score is
  the weighted score times `(1 - capped quality_penalty)`.
- HAG is used normally; high-ground-variation points blend `0.35 * HAG + 0.65 * height_m`;
  invalid, extrapolated, below-ground, and extreme-HAG points fall back to `height_m`.
- `NORMAL_DIRECTION_AMBIGUOUS` has no A3 candidate penalty. Only principal-direction
  ambiguity receives the approved 0.05 penalty.
- `rule_score` is retained unchanged. Manual include can raise only `candidate_score`; exclude
  always wins. `rule`, `union`, `intersection`, and `manual_override` modes are implemented.
  Numeric-invalid points cannot be manually included unless explicitly configured.
- Empty candidate sets are legal. Default no-overwrite, cancellation, output-dir boundary, and
  temporary-file cleanup failures return explicit structured errors.

## Output and preview

- Core outputs are written to unique temporary files then atomically renamed:
  - `outputs/candidates/scene_simple_wire_candidates.npz`
  - `outputs/candidates/scene_simple_wire_candidates.json`
- The independent NPZ-only preview was successfully written to:
  - `outputs/previews/scene_simple_wire_candidates_preview.html`
- Preview samples a deterministic 40,000-point non-candidate background and all 17,784
  candidates (under the 50,000 cap). If preview generation fails after core publication, the
  core artifacts remain valid and the returned result records a warning.

## Synthetic tests

- A3 tests cover ideal high line points, high/low points, planar and scattered points, geometry
  guard, complete score components, density five-piece boundaries, normal/principal direction
  ambiguity, quality penalty cap, normal/mixed/fallback HAG handling, numeric-invalid handling,
  evidence/quality flags, level/source values, rule/manual/union/intersection/manual-override
  modes, original-index and local-box selection, exclude priority, valid empty output, output
  reload, no-overwrite, cancellation, output path restriction, preview deterministic sampling,
  preview failure isolation, input hash protection, no LAS import, and no N x N distance API.
- A3-only result: `11 passed`.

## Full scene_simple validation

- Inputs: existing A1/A2A/A2B artifacts only; `M=146126`.
- Default rule-mode output: 17,784 candidates (12.1703%). The planning estimate was about
  17,822, a difference of 38 points (0.21% of the estimate); no parameters were adjusted.
- Candidate levels: low 6,432; medium 3,126; high 8,226.
- Candidate sources: rule 17,784; manual 0; hybrid 0.
- Rule score: min 0.0000, median 0.1941, p95 0.8550, max 1.0000. Candidate score: min
  0.4500, median 0.7598, p95 1.0000, max 1.0000.
- Evidence counts: high-linearity 12,137; low-scattering 50,445; acceptable-planarity 58,185;
  sufficient-height 32,653; acceptable-density 59,902; height-fallback-used 43,315.
- Quality counts: geometry-degraded 4; ground-degraded 42,711; HAG-unavailable 0;
  below-ground 1,206; extreme-density 67,408; ambiguous-principal-direction 4; low-confidence
  6,432; invalid-input-feature 0.
- HAG handling: normal 102,811; high-gradient blend 42,109; fallback 1,206. Candidate counts
  in the mutually exclusive normal/blend/fallback groups were 11,526/6,123/135. There were
  6,171 candidates in high-ground-variation regions in total (including 48 fallback points),
  and 472 candidates were in raw negative-HAG areas, where the documented height fallback is
  intentionally used.
- Candidate XYZ local range: `[0.0020, 0.0004, 0.0033]` to
  `[12.9997, 9.9999, 10.8824]` m. Candidate local-height median is 6.8946 m; candidate HAG
  median is 4.8838 m.
- The top 100 candidate points have median linearity 0.9878, scattering 0.000251, planarity
  0.01183, density 239.09 points/m^3, local height 7.9591 m, and HAG 7.6376 m.
- Candidates occupy 370 half-metre cells; the largest contains 282 points (1.586% of all
  candidates), so there is no single-grid concentration anomaly.
- Core elapsed time was 0.794 s. RSS start/peak/end was 39.8/87.2/83.5 MB, below the 1.5 GB
  target. The output NPZ SHA-256 is
  `7bb56ed4f40c2409ef8d25b72c4e59a0227480db95546975154930288987dfac`.
- Reload validation passed: result SHA matches the NPZ; all M-length arrays are aligned;
  `xyz_local` and origin equal A1; candidate original/sample indexes are unique; all scores are
  finite. The six preexisting A1/A2A/A2B artifact hashes are unchanged.

## Quality and repository protection

- `E:\Miniforge\envs\treeline\python.exe -m pytest`: `144 passed`.
- `E:\Miniforge\envs\treeline\python.exe -m ruff check .`: passed.
- A3 `ruff format --check`: passed.
- `compileall algorithms core tests scripts`: passed.
- The generated outputs are ignored by `.gitignore`; no data, point cloud, model, cache, NPZ,
  JSON, or HTML artifact is tracked. No A1/A2A/A2B input artifact changed.

## Limitations and recommendation

- This is a transparent baseline, not labeled wire classification. It intentionally has no
  connected components, wire IDs, slices, cross-slice association, or spline reconstruction.
- Ground-quality fallback preserves candidates in uncertain terrain; A4 should use the retained
  provenance and quality flags rather than treating all candidates as confirmed conductors.
- Recommend committing A3 after GPT review. Recommend entering A4 only after that approved A3
  commit; A4 has not been started in this task.
