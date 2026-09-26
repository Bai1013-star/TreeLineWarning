# I1 cross-computer bundle review

## Baseline and scope

- Baseline: `feature/algorithm` at `b120a45bb0bdc9eb3c4a966bef26a0e747b28359`.
- Interpreter: `E:\Miniforge\envs\treeline\python.exe`.
- Scope is limited to the approved nine I1 contract, integration, test, documentation, and review files.
- W1, V2, V2A contracts, implementations, tests, reviews, artifacts, LAS/LAZ, and A3 point data were not modified or accessed.

## Contract and architecture

I1 implements `i1-bundle-v1` / `1.0.0` as a portable immutable directory. Its JSON completion marker references copied W1 and V2 NPZ/JSON siblings and optional V2A siblings; no combined NPZ, per-cluster summary duplication, or large point-array duplication is created. References are typed by artifact kind, include relative safe paths, SHA-256 and byte size, and record source schema/algorithm versions.

The deterministic bundle ID canonicalizes source/manifest identity, coordinate context, policy version, required source digests, and optional V2A state. It deliberately excludes time, username, source/destination machine paths, and copy ordering.

## Coordinate and eligibility policy

I1 only accepts `LOCAL_CARTESIAN_XYZ`, metres, and identity transformation. Export recovers W1's origin from W1's hash-verified A4 JSON and requires exact finite float64 equality with V2's origin. The recorded provenance hash is rechecked against W1 during standalone reload.

The frozen `i1-eligibility-v1` policy exposes only confirmed, ready, finite W1 segments with at least two samples. Gaps are retained and no segment is fabricated. Vegetation requires confirmed status, a valid vegetation ID, nonempty member rows, exact member-row cluster/vegetation agreement, readiness, and finite geometry. V2A remains optional audit/UI metadata and has no eligibility effect.

## Publication and reload

Export validates public W1/V2(/V2A) loaders, copies files to a same-volume temporary directory, verifies copied sizes and digests, writes/parses the manifest last, then renames the complete directory to its deterministic ID. Existing equivalent bundles are reloaded and reused. Standalone reload validates paths, files, hashes, versions, source identity, coordinate provenance, and optional V2A-to-V2 lineage.

## Verification

Focused deterministic fixture tests cover valid bundles, optional V2A presence/absence/isolation, deterministic ID reuse, relocation, multiple ready/non-ready segments, explicit gaps, confirmed/review/provisional/rejected vegetation handling, no eligible records, origin-provenance absence, corrupted/missing files, path traversal, malformed/unknown manifest fields, and expected-ID mismatch.

No real structured export was run: the task prohibits regenerating or modifying W1/V2/V2A artifacts. Consequently, no runtime/RSS or runtime artifact hashes are claimed.

Final verification: focused I1 contracts/integration tests passed (`21 passed`); the full suite passed (`298 passed`); `ruff check .` passed; `compileall core algorithms tests` passed; and the approved I1 Python files passed `ruff format --check`. I1 makes no C1/R1/P1/V3 or prediction-side adapter claim.
