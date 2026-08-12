# TreeLineWarning P2 Project Persistence Implementation Plan

Status: design-only review candidate. No P2 implementation is authorized by this document.

Baseline branch: feature/fast-research-pipeline

P1 implementation baseline: 980dc5cecf66aa974972359e49ef3872c98784dd

P1 design baseline: bd4a899ca2eaa71c854db2f772b6697489460af3

## 1. P2 Objective

P2 adds a durable, validated persistence boundary for the project aggregate defined by P1. It must allow a future TreeLineWarning application to create a project, bind it to one authoritative LAS/LAZ source identity, persist towers, spans, counters, and a lightweight run index, close safely, and reopen the same project with the same identities, topology, revisions, statuses, and source binding.

P2 also establishes two safety properties:

1. Only one writable application session can own a project at a time.
2. An interrupted write cannot cause the loader to combine files from different project states.

The default source policy is **reference the original source read-only**. Project creation does not copy the LAS/LAZ. Source relocation is an explicit, identity-verified access-record update.

P2 does not implement Potree, viewer caches, desktop UI, project actions, scientific execution, prediction execution, or any expensive point-cloud processing.

## 2. Baseline Audit

### Repository and Git state

At the audit point:

- branch: feature/fast-research-pipeline;
- HEAD: 980dc5c (feat: add project tower span contracts);
- HEAD versus origin/feature/fast-research-pipeline: 0 0 divergence;
- P1 is present locally and remotely.

The worktree already contains unrelated, uncommitted manual-span and other working files, including .claude/. They are outside this plan and must be preserved. .claude/ is not part of the audit and must not be read or modified.

Only this document is permitted to be created by the current plan-only task.

### P1 contract audit

core/contracts/project.py contains immutable, frozen P1 contracts and to_dict() serializers. It intentionally contains no filesystem I/O and currently does not provide complete from_dict() loaders. P2 must add deserialization at the persistence boundary without weakening the contracts.

Important existing rules are:

- ProjectContract uses exact schema treeline-project-v1, a canonical UUID project_id, UTC timestamps, an embedded ProjectSourceReference, an abstract ViewerReference, project-relative registry references, positive next-sequence counters, and optional PredictionCapability.
- TowerContract uses Txxx display IDs, a lifecycle state, source binding, coordinate context, geometry, refinement evidence, a positive revision, and timestamps.
- SpanContract uses Sxxx, two distinct endpoint tower IDs, endpoint revisions, source binding, derived geometry, analysis status/dependencies, and optional artifact/prediction envelopes.
- derive_span_geometry_digest() binds source binding, endpoint revisions, and geometry/configuration data.
- PredictionCapability.not_connected() is representable without a provider or ML package.
- _safe_relative_posix_path() rejects traversal, absolute paths, backslashes, colons/ADS, control characters, Windows-forbidden filename characters, and reserved Windows device basenames.
- SourceIdentityBinding and ProjectSourceReference validate the source content and source-record digests through the existing FRV contract.

P2 must preserve these semantics. It must not add filesystem concerns to the scientific contracts merely for convenience.

### Existing I/O and persistence audit

The repository has several JSON writers using a temporary sibling and os.replace()/Path.replace(). These establish useful intent but are not one complete P2 durability primitive:

- algorithms/spans/checkpoint.py writes a digest-keyed checkpoint with a fixed .tmp sibling and replaces it. It is fail-closed on provenance mismatch, but does not fsync the file or coordinate multiple project files.
- algorithms/manual_span/pipeline.py and algorithms/manual_span/server.py use fixed .tmp JSON writers for working-run state. Those files are unrelated to project authority and must remain unchanged.
- algorithms/integration/i1.py stages a complete bundle directory, validates the manifest, flushes/fsyncs the manifest, and publishes the directory with one rename. Its completion-manifest-last pattern is the closest repository precedent for a coherent multi-file publication.
- algorithms/vegetation/frv_artifacts.py has canonical JSON helpers, file digesting, and staged-directory publication. It is useful precedent for validation and artifact publication, but P2 should avoid importing scientific artifact machinery into project persistence.
- core/config/fast_baseline.py has a compact sorted-key JSON digest convention.

No project lock, project open service, project registry loader, project transaction-generation service, or general-purpose durable JSON writer exists today.

### Source identity and provider audit

The reusable authoritative identity is core.contracts.frv.SourceIdentity, which includes:

- source ID and recorded/resolved access paths;
- source file SHA-256;
- optional canonical point-content identity/spec digests;
- point count and bounds;
- coordinate frame, units, and origin context;
- relocation/equivalence audits;
- content and record/access digests.

SourceIdentity.from_dict(), validate_source_content_identity_digest(), and validate_source_identity_record_digest() are reusable. algorithms/io/headers.py reads LAS/LAZ headers without materializing points, and algorithms/io/source_provider.py verifies source identity while streaming. P2 source validation may use header-only reads and bounded hashing; it must not trigger a full scientific pipeline.

The existing algorithms/io/discovery.py::stable_source_id() is a discovery label based on relative path. It is not authoritative project identity and must not replace SourceIdentity.

## 3. Existing Reusable Infrastructure

| Concern | Reuse | P2 treatment |
| --- | --- | --- |
| P1 object validation and serialization | core/contracts/project.py | Construct exact contracts after strict decode; persist to_dict() projections. |
| Source identity | SourceIdentity and FRV validators | Reuse as the sole source identity authority. |
| Canonical JSON/digests | core.contracts.frv.canonical_frv_json_bytes() and frv_digest() | Use for logical document bytes/digests; no new identity hash scheme. |
| JSON field checks | core.contracts._json.json_safe, require_fields, reject_unknown_fields | Use for persistence envelopes and recursive decoding. |
| Relative path safety | P1 _safe_relative_posix_path() policy | Keep one implementation; if it must be made public, perform a mechanical helper extraction with compatibility tests. |
| Header-only source access | algorithms.io.headers | Use only for source availability/identity checks; no point arrays. |
| Provenance invalidation | algorithms.spans.checkpoint.py | Reuse its fail-closed principle for run-index dependency validation, not its fixed-temp writer. |
| Complete directory publication | algorithms.integration.i1.py and FRV artifact staging | Adapt the staging/validate/publish idea into a project-generation service. |
| Existing manual-span persistence | algorithms.manual_span/* | Do not change or treat pending selector journals as confirmed project topology. |

The P2 writer must not call json.dump() directly on an authoritative path, must not reuse a fixed .tmp name, and must not use a scientific output writer as a project transaction mechanism.

## 4. Persistence Architecture

P2 freezes an immutable-snapshot model with a small atomic active pointer. Each complete project state is a self-contained generation. The root pointer changes only after the generation is sealed and validated.

The authoritative object graph is:

    root project.json
            |
            +--> one sealed state/generations/<generation-id>/
                    +--> project.json
                    +--> source/source-reference.json
                    +--> towers/towers.json
                    +--> spans/spans.json
                    +--> runs/run-index.json
                    +--> generation.json  (completion/transaction manifest)

The root project.json is an active-pointer envelope whose nested project member is the exact serialized P1 ProjectContract. The generation project.json is the exact P1 project manifest. The root copy makes the project discoverable at the approved path; the generation copy is immutable and is the object validated with the other registry files.

A save is copy-on-write:

1. Load one complete active generation under the project lock.
2. Apply the in-memory mutation and allocate any IDs in the candidate state.
3. Write a new staging generation in a sibling temporary directory.
4. Parse and validate every staged file, all digests, all cross-references, and all counters.
5. Write and fsync generation.json last with sealed: true.
6. Atomically rename the complete staging directory to its final generation directory.
7. Re-read and validate the final generation.
8. Atomically replace the root project.json pointer with a pointer to that final generation.

The previous generation is never overwritten or deleted by normal P2 operation. A failure before step 8 leaves the prior root pointer authoritative. A failure after step 8 is safe because the new generation was already sealed. A malformed or manually damaged root pointer fails closed; diagnostic recovery may list complete generations, but normal loading never guesses a generation.

## 5. Physical Project Layout

The exact V1 layout is:

    <ProjectRoot>/
        project.json                         # active pointer + nested P1 manifest
        project.lock                         # OS-held single-writer lock metadata
        state/
            generations/
                g-0000000000000001-<short-id>/
                    generation.json          # written last; completion marker
                    project.json              # exact P1 ProjectContract projection
                    source/
                        source-reference.json
                    towers/
                        towers.json
                    spans/
                        spans.json
                    runs/
                        run-index.json
        logs/                                 # reserved; not authoritative in P2

The generation ID is opaque, lower-case, and filesystem-safe. A monotonically formatted sequence plus a random suffix is recommended so diagnostics remain understandable while concurrent/stale staging directories cannot collide. Generation IDs are not tower/span identities and must not be exposed as domain IDs.

There are no viewer/, cache/, prediction/, or copied source/managed/ directories in normal P2 creation. logs/ may exist but contains no input to project validity.

All project-internal references are normalized POSIX relative paths and must resolve inside the selected generation or the project root according to their declared scope. The only external absolute path is the explicitly audited source access path inside SourceIdentity/the source access envelope.

The P1 ProjectContract fields tower_registry_reference, span_registry_reference, and run_index_reference point to the selected generation’s registry files, for example state/generations/g-.../towers/towers.json. The source object remains embedded in the P1 contract because that is the approved P1 type; the sidecar is checked against it and carries source access/header metadata needed by P2.

## 6. Serialization Rules

### Canonical JSON

All authoritative JSON is UTF-8, has no BOM, uses JSON objects with string keys, uses ensure_ascii=False, sort_keys=True, compact separators (',', ':'), and allow_nan=False. The published file bytes are the canonical bytes followed by exactly one LF. There is no trailing whitespace.

Logical document digests are SHA-256 over the canonical bytes without the final LF. Generation file digests are SHA-256 over the exact published bytes including the final LF, plus the byte size. Both notions are explicit and must not be conflated.

P2 should call the existing canonical FRV JSON implementation for contract projections. If a standard-library-only persistence helper is later preferred, it must be byte-for-byte compatible and covered by parity tests; it must not introduce a second digest convention.

### Strict decode

Each envelope has an exact schema version and exact required-field set. Unknown fields are rejected, not ignored. Missing required fields, wrong JSON root types, wrong enum values, invalid numeric types, nonfinite numbers, malformed digests, invalid timestamps, and invalid nested contracts are errors.

Only the currently approved V1 schema versions are accepted. A future or unknown version produces an explicit unsupported-schema error. There is no silent migration during normal open. A future migration tool may read an old generation and write a new generation, but that is outside P2.

Nested P1 values are decoded recursively into the exact contract classes. Enums are reconstructed from their serialized values; tuples/lists are normalized by the contract constructors; SourceIdentity.from_dict() is used for the FRV source record. Every reconstructed object is validated before it enters the in-memory project aggregate.

### Deterministic ordering

Registry entries are serialized as arrays sorted by numeric tower/span sequence. Revision arrays are sorted by positive revision. Run-index entries are sorted by run_id. Reservation entries are sorted by sequence. Mapping keys are canonicalized by the JSON serializer. Semantic order inside P1 historical/status collections remains the order defined by the contract.

## 7. Project Manifest

### P1 manifest payload

The generation project.json is exactly ProjectContract.to_dict() and must preserve:

- schema_version: treeline-project-v1;
- immutable project_id;
- project_name, created_at, and updated_at;
- embedded source_reference;
- abstract viewer state/reference;
- tower, span, and run-index references;
- prediction_capability;
- next_tower_sequence and next_span_sequence.

The P1 project_id is generated once at creation and is never replaced by a load/save operation. Every envelope in a generation must carry or imply the same project ID. A save request that supplies another project ID fails before staging.

### Root active-pointer envelope

Root project.json has the exact persistence envelope:

    {
      "schema_version": "treeline-project-active-pointer-v1",
      "project_id": "<same UUID>",
      "active_generation_id": "g-...",
      "generation_manifest_reference": "state/generations/g-.../generation.json",
      "generation_manifest_sha256": "<exact-file digest>",
      "project_manifest_reference": "state/generations/g-.../project.json",
      "project_manifest_sha256": "<exact-file digest>",
      "project": { "schema_version": "treeline-project-v1", "...": "..." }
    }

The root envelope and nested project projection are validated against the selected generation. A mismatch in project ID, generation ID, references, or file digest is an integrity failure. The root envelope is not itself a second independent project identity.

### Required/optional policy

All fields emitted by ProjectContract.to_dict() are required in the P1 manifest, including prediction_capability. Optionality is represented by the contract’s explicit null values and empty collections, not by omitted keys. Persistence envelope fields are all required. Unknown fields are rejected.

### Immutable identity protection

The loader records the project ID from the first validated P1 manifest and compares it to the root pointer, generation marker, source envelope, tower envelope, span envelope, run-index envelope, and every run entry. It does not infer a project ID from a directory name. Save APIs receive the loaded project identity and cannot replace it.

## 8. Tower Registry Persistence

V1 uses one towers.json registry per generation, not one file per tower. One registry is easier to validate as a single identity set, makes counter checks atomic with the records, and avoids many small-file failure modes. It is also small compared with point-cloud/scientific artifacts.

The registry envelope contains:

    schema_version: treeline-tower-registry-v1
    generation_id
    project_id
    reserved_sequences: [{sequence, tower_id, reserved_at, reason}]
    records: [
      {
        tower_id,
        current_revision,
        revisions: [<TowerContract.to_dict()>, ...]
      }
    ]

There is exactly one registry entry per physical tower_id. A revision snapshot is stored once inside that entry; current_revision identifies the current snapshot. Revision numbers are positive, unique, and start at 1. A domain correction appends a new snapshot and advances current_revision; it does not mutate a prior historical snapshot. Every snapshot has the same tower ID and source content identity.

The current snapshot’s state is either CONFIRMED or ARCHIVED. A failed or rejected selection is never serialized as a TowerContract. If a durable ID reservation is needed before confirmation, it is represented only in reserved_sequences, not in records, and it cannot be used to form a span.

On reload the exact tower ID, current revision, historical revisions, lifecycle state, evidence, and timestamps are reconstructed. Duplicate tower IDs, duplicate revisions, gaps that violate the chosen revision policy, mismatched current_revision, malformed snapshots, and inconsistent source bindings fail closed.

## 9. Span Registry Persistence

V1 uses one spans.json registry per generation. It contains a strict envelope with schema_version: treeline-span-registry-v1, generation_id, project_id, reserved_sequences, and a sorted records array of SpanContract.to_dict() values.

Each physical span is stored once. Its endpoints are only:

    S001 -> from_tower_id T001 / from_tower_revision 1
    S001 -> to_tower_id   T002 / to_tower_revision   1

T002 is resolved from the tower registry and is not copied into the span. On reopen, the topology S001 = T001 -> T002 and S002 = T002 -> T003 is rebuilt by ID lookup. No endpoint copy is created.

The cross-reference validator requires:

- both endpoint towers exist in the registry;
- the referenced revisions exist in that tower’s retained revision history;
- a self-span is rejected;
- an active span cannot use an archived endpoint; an archived span may retain archived endpoints for historical topology;
- a span’s source content identity matches the project source content identity;
- the span’s coordinate context matches its source binding;
- the geometry digest recomputes exactly;
- analysis.run_dependencies, if present, obeys P1 stale-state rules;
- duplicate/conflicting span IDs are rejected;
- a span’s current dependency state is coherent with its endpoint revisions and geometry.

Archiving a tower is an explicit write operation. The P2 service must either archive all dependent active spans in the same generation or reject the tower archive; it must not leave a CONFIRMED span pointing at an archived endpoint. Historical run-index entries remain referentially valid.

## 10. Source Reference Persistence

source/source-reference.json is a strict P2 envelope:

    schema_version: treeline-project-source-reference-persistence-v1
    generation_id
    project_id
    project_source_reference: <ProjectSourceReference.to_dict()>
    access_metadata:
      policy: REFERENCE_ORIGINAL_SOURCE_READ_ONLY
      current_resolved_source_path
      file_size_bytes
      source_file_sha256
      point_count
      bounds_min_xyz
      bounds_max_xyz
      header: <available LAS/LAZ header summary fields>
      source_available_at_save
      recorded_at
    source_identity_history: [<prior validated SourceIdentity.to_dict()>, ...]

The embedded project_source_reference must be byte/logically identical to the P1 project manifest’s source_reference. The sidecar does not create a second source identity. It supplies the required access/header record and preserves prior approved access records for validating historical dependencies after relocation.

At minimum the sidecar preserves the source content identity digest, source identity record/access digest, current resolved path, source file size, source file SHA-256, point count, bounds, header metadata where available, coordinate frame, units, and origin context. The current SourceIdentity remains the authoritative digest-bearing object.

Creation does not copy the original LAS/LAZ. A source file is opened read-only for identity/header verification. No project-relative path field may contain the external source path.

## 11. Run Index

P2 persists only a lightweight runs/run-index.json envelope. It does not execute a span or create result artifacts. The envelope is:

    schema_version: treeline-run-index-v1
    generation_id
    project_id
    entries: [
      {
        run_id,
        span_id,
        status,
        artifact_root_reference: <safe project-relative path or null>,
        source_content_identity_digest,
        source_identity_record_digest,
        span_dependencies: <SpanDependencySnapshot.to_dict()>,
        created_at,
        completed_at: <timestamp or null>
      }
    ]

The run index is strict and identity-keyed. run_id and span_id are validated by the P1 ID rules; statuses are an explicit closed set compatible with P1 analysis status (RUNNING, COMPLETE, FAILED, INTERRUPTED, STALE, with READY allowed only if a future producer needs it). Artifact references are project-relative and are not followed during P2 load unless an explicitly supplied later validator needs them.

Each entry must refer to an existing span, carry the project ID, and preserve the source/tower-revision/geometry/configuration dependencies needed to decide whether a later run is current or stale. A historical run may retain an older source record/access digest after relocation, but its source content identity must remain compatible with the project source and its dependency snapshot must remain internally valid.

No prediction result file, model file, or ML import is part of the run index.

## 12. Monotonic ID Allocation

next_tower_sequence and next_span_sequence are the next never-before-reserved sequence values, starting at 1. Display IDs are generated only by the existing helpers:

    T{sequence:03d}
    S{sequence:03d}

The minimum width is three; IDs continue as T1000/S1000 rather than wrapping or truncating.

The exact transaction algorithm is:

1. Under the project write lock, read the active generation.
2. Validate that the counter is strictly greater than every active, archived, historical, and reserved sequence already represented.
3. Reserve the current counter value in the candidate registry and increment the candidate counter by one.
4. Validate the proposed domain record, including source binding and all cross-references.
5. If validation succeeds, convert the reservation into the confirmed Tower/Span record in the same candidate generation.
6. If confirmation fails after the sequence was durably reserved, retain a reservation-only entry with a reason and publish the incremented counter as a normal coherent generation. It is not a Tower/Span record and cannot be used by a span.
7. Return an allocated ID to the caller only after a complete generation containing either its domain record or its reservation is active.

Thus, a reserved-but-unused ID may be skipped. Skipping is intentional: provenance and crash ambiguity are safer than attempting to infer whether an identifier was ever observed. A failed pre-reservation click/refinement proposal does not allocate an ID and creates no Tower record. The P2 API must not expose a confirmed ID before its generation commit.

A crash before a reservation generation is published leaves the old counter authoritative; the ID was never returned and may be retried safely. A crash after publication leaves either a record or an explicit reservation, so the next allocation cannot reuse it.

On load, next_tower_sequence must be greater than every sequence in tower records, tower revisions, archived/tombstoned entries, and reserved sequences. The same rule applies to spans. Counters are never reduced or silently repaired during normal open.

## 13. Tombstone / Archive Semantics

P2 does not erase confirmed scientific/project identities.

- Active Tower: one confirmed physical tower available for new topology operations.
- Archived Tower: the same physical identity retained in the registry but excluded from normal active topology actions.
- Active Span: confirmed topology with non-archived endpoints and coherent current dependencies.
- Archived Span: retained topology/history that is excluded from new execution and normal active navigation.
- Reserved sequence: an allocated ID with no confirmed domain record; it is not a Tower or Span.

Archived records remain serialized, retain their IDs and revision history, and reserve their sequences permanently. Historical run-index entries continue to refer to them. The normal UI may hide archived entities later, but P2 persistence must keep them available to diagnostics and historical reports.

Removal is a state transition, not a physical deletion. A dependent active span must be archived in the same transaction as a tower archive or the archive operation must fail. No new span may use an archived tower. No future physical tower may inherit an archived tower’s ID.

## 14. Atomic Single-File Writes

The P2 primitive for any authoritative small JSON is:

    target.json
      -> create a unique temporary sibling in the same directory
      -> write canonical UTF-8 bytes plus one LF
      -> flush and fsync the temporary file
      -> parse and validate the temporary bytes
      -> atomically replace target.json
      -> fsync the containing directory where the platform permits it
      -> re-read and validate the published target

The temporary name includes a UUID/session component; a fixed .tmp path is forbidden because two writers or a stale process could overwrite one another’s staging data. On Windows, os.replace() must be used only after all handles are closed. A file fsync failure aborts publication. Directory fsync is attempted where supported; unsupported directory flush is recorded as a platform limitation, not hidden as a successful stronger guarantee.

For project state, the primitive is applied inside an isolated staging generation. No active generation file is overwritten in place. The root active pointer uses the same primitive only after the generation is complete.

## 15. Multi-File Transaction / Generation Model

P2 recommends immutable generation directories plus an atomic root pointer. This is the V1 strategy; alternatives were rejected as follows:

- Per-file atomic replacement alone can expose new towers with old spans or an old project manifest.
- A “manifest written last” approach with fixed root filenames detects some mismatches but still leaves a mixed root state and requires backup choreography.
- A database is unnecessary for five small JSON documents and would add packaging/locking/deployment complexity.

Every generated registry envelope contains the same generation_id. generation.json is the completion manifest and contains:

- exact generation schema version;
- project ID and generation ID;
- sealed: true written only after all other files are complete;
- sorted relative file records with exact byte sizes and SHA-256 digests for project, source, towers, spans, and run index;
- the logical project-manifest digest;
- creation/publication timestamps.

The generation marker itself is not included in its own digest list. Its own exact bytes are protected by the root pointer’s generation_manifest_sha256.

The loader first validates the root pointer, then the generation marker, then every listed file and every cross-reference. It never reads one file from the active generation and another file from a staging directory. Staging directories and unsealed generations are ignored by normal open. If the active pointer names an incomplete/missing/mismatched generation, normal open fails explicitly; it does not fall back to a guessed generation. A diagnostic recovery mode can enumerate sealed generations and require an explicit operator choice.

This guarantees that an interrupted save preserves the prior complete generation as the last known-good project state.

## 16. Project Lock / Single Writer

project.lock is a lock metadata file plus an OS-held byte-range lock. It is not a project identity and is not included in any generation digest.

The lock metadata schema is:

    schema_version: treeline-project-lock-v1
    session_id: canonical UUID
    process_id: integer
    hostname: current machine name
    opened_at: UTC timestamp
    application_version: opaque version string

On Windows, core/project/locking.py opens/creates the file with standard-library file APIs and obtains a non-blocking exclusive byte-range lock using msvcrt. The handle remains open for the entire writable session. A POSIX fcntl implementation is allowed for development/test portability; no third-party package is required. Metadata is written only after the OS lock is owned and is flushed/fsynced best-effort. The lock handle is released on close and process exit.

Writable open behavior:

- Instance A acquires the OS lock and may save.
- Instance B cannot acquire it and must not write. It may open the validated active generation read-only or report “project already open.”
- A lock file’s existence alone is not proof of ownership; the OS lock is authoritative.
- The project lock is never used as a domain ID, source ID, generation ID, tower ID, or span ID.

The implementation must not assume C:, a particular username, or a fixed machine path. The lock is next to the project root regardless of its drive.

## 17. Stale-Lock Recovery

P2 must never delete a lock solely because opened_at is old.

When a writer cannot acquire the OS lock, it must treat the project as owned. It may read lock metadata for diagnostics, but it must not override the owner based on PID age or hostname alone. If the platform safely permits checking the owner process, that information is advisory only while the OS lock remains held.

A stale lock is recovered by attempting the same OS-level exclusive lock acquisition after the prior handle has actually disappeared. If acquisition succeeds, the old metadata file is reused and overwritten with a new session_id; no blind delete is required. If acquisition fails, the lock is live/unknown and the caller remains read-only or stops.

Malformed or partially written stale metadata does not prevent recovery after the OS lock is acquired, because the lock metadata is diagnostic and the new owner rewrites it. Malformed metadata while the OS lock is held is reported as an explicit lock-diagnostic problem and never used to justify concurrent writing.

The lock implementation should include a best-effort Windows process-existence diagnostic using standard-library/OS APIs, but PID existence, timestamps, and machine identity must never replace the kernel lock as the safety decision. This handles PID reuse and removable/network storage more safely.

## 18. Read-Only / Degraded Open

P2 includes read-only loading in the persistence API, even if a later UI exposes it in a small follow-up.

Read-only mode is allowed when:

- another writer owns the project lock;
- the source is missing;
- the source path exists but fails identity validation, in explicit degraded/diagnostic mode;
- the user only wants historical topology/report metadata.

Read-only mode may parse and expose the last complete generation, tower/span topology, archived identities, statuses, run metadata, and prediction capability. It must not:

- change counters, IDs, revisions, archive state, or statuses;
- relocate or replace the source reference;
- publish a generation;
- create a run or scientific artifact;
- repair malformed authoritative files silently.

Missing source produces a defined SOURCE_MISSING availability state: project topology and reports remain readable, but refinement, new analysis, source relocation approval, and other source-dependent actions are disabled. A present path that has only passed fast metadata checks is SOURCE_PRESENT_UNVERIFIED, not scientifically available. An identity mismatch produces SOURCE_IDENTITY_MISMATCH; the project is not treated as source-available. Normal writable scientific operations reject unverified or mismatched source state. A diagnostic read-only loader may still show historical metadata.

Implementing this mode directly in P2 is recommended because lock contention and missing-source handling are persistence concerns, not UI concerns.

## 19. Project Open Validation Sequence

The normal open sequence is fixed. The writable and read-only lock orders are intentionally different:

1. Validate that the requested project root is a directory and that the root path itself is not a project-relative artifact reference.
2. For writable mode, acquire the project writer lock before reading the authoritative root pointer. If any pre-lock read was made for diagnostics or UI, discard it completely after lock acquisition.
3. While holding the writable lock, read and strictly decode root project.json as the active-pointer envelope. For read-only mode, do not take the writer lock; its pointer/generation is immutable once selected.
4. Locate exactly the referenced final generation directory; reject staging names and path escapes.
5. Read generation.json; validate schema, sealed, project ID, generation ID, file list, sizes, and digests.
6. Read the generation project.json; reconstruct and validate the exact ProjectContract.
7. Validate the source sidecar, decode SourceIdentity, verify its persisted digests, and compare the sidecar source projection to the project manifest source reference.
8. Validate the tower registry, all revision snapshots, reservations, unique IDs, lifecycle states, and source-content bindings.
9. Validate the span registry, endpoint existence/revisions, archived-endpoint rules, self-span rule, geometry digest, analysis/dependency consistency, and source bindings.
10. Validate all cross-file references and all project-relative paths using the P1 hardened path policy; ensure they resolve within the declared generation scope.
11. Validate the run index, run IDs, span IDs, status/timestamp rules, artifact references, and historical dependency identity.
12. Validate next_tower_sequence and next_span_sequence against every represented or reserved identity. Never reduce a counter.
13. Perform the fast source availability check: inspect the audited external path, stat it, and perform safe header/metadata checks without a full source hash or point materialization. Set SOURCE_MISSING, SOURCE_PRESENT_UNVERIFIED, or an explicit mismatch state.
14. Expose one immutable in-memory aggregate plus the explicit availability/open-mode state. Before any source-dependent mutation or scientific execution, run strong SourceIdentity verification and require success; a fast check never authorizes science.

Any integrity failure before step 14 returns a typed failure and no partial topology. The loader must never return a half-populated project as if it were valid.

## 20. Source Relocation

The default source policy is REFERENCE_ORIGINAL_SOURCE_READ_ONLY.

### Frozen P1 binding distinction

The audited P1 contracts make the distinction important:

- SourceIdentity.to_content_digest_dict() covers the scientific source-content identity: source identity schema, source ID, source file SHA-256, canonical point-content/spec identity when present, point count, bounds, coordinate frame, units, and origin context. The resulting source_content_identity_digest is immutable for the same exact source content.
- SourceIdentity.to_record_digest_dict() adds recorded/resolved access paths, relocation/equivalence audit references, and the source content digest. The resulting source_identity_record_digest is an audited access-record identity and changes when the approved access record changes.
- SourceIdentityBinding contains both digests and a CoordinateContext containing both digests. Therefore a TowerContract, SpanContract, and SpanDependencySnapshot capture the binding that was valid when that scientific record or run was created; they are not silently rewritten whenever the current access path changes.
- derive_span_geometry_digest() includes the complete serialized SourceIdentityBinding. A new binding would therefore produce a different geometry digest, but an exact-content relocation does not replace the binding inside an existing historical record.

P2 validates that every current or historical tower/span binding has the same immutable source content identity as the current project source and that its source_identity_record_digest is either the current audited record or a validated record in source_identity_history. This preserves strict provenance without treating a path change as a scientific geometry change.

P1 does not require a TowerContract or SpanContract source_identity_record_digest to equal the current ProjectSourceReference record digest; it requires each contract’s own SourceIdentityBinding and CoordinateContext to be internally consistent. Therefore an exact-content access relocation needs no new Tower revision or Span definition transition. A new revision/definition transition is reserved for an actual domain correction or a changed scientific content binding, not for a path-only move.

### Open-time source states

- **Source exists:** run the fast availability check first. A matching path/stat/header result is SOURCE_PRESENT_UNVERIFIED until strong SourceIdentity verification succeeds. Ordinary project launch does not perform a full multi-gigabyte hash merely to display persisted topology and historical metadata.
- **Source missing:** open topology in SOURCE_MISSING degraded mode; do not change the source reference.
- **Source exists but fast metadata or strong identity mismatches:** do not adopt it. Return SOURCE_IDENTITY_MISMATCH for source-dependent actions; a diagnostic read-only view may show the project.
- **Source moved:** do not search-and-adopt by filename. Require an explicit candidate path from the user/application.

The fast availability check requires only path existence, safe file stat information, extension/type checks, and header/metadata checks that do not materialize the point cloud. It provides availability for navigation and metadata display, never scientific authorization. Strong verification performs the required full SourceIdentity checks, including the source-file digest and the existing provider’s TOCTOU/finalization safeguards, before refinement, new scientific execution, or any source-dependent authoritative mutation.

Strong verification may be cached only in memory for the current ProjectSession. The cache key must include the normalized resolved path, expected source content identity, file size, modification signature, and platform file identity where available; any changed signature invalidates it. The cache is not persisted across sessions and does not replace the final source-provider verification required by a scientific operation.

### Approved relocation workflow

1. Require a writable project lock and a validated active generation.
2. Read the candidate path read-only and perform strong verification of the stored source file identity, not just the filename. P2 V1 accepts exact stored source-file identity by default; canonical/equivalent-content adoption requires a separately approved equivalence audit and is not implicit.
3. Create a new audited SourceIdentity access record with the new resolved path and append the prior source identity to source_identity_history.
4. Preserve the stable source content identity. Change only the current audited source access record and its source_identity_record_digest through the approved FRV relocation semantics.
5. Do not rewrite any existing serialized TowerContract revision, SpanContract, or SpanDependencySnapshot in place. Their historical source binding, coordinate context, geometry digest, and analysis state remain byte-for-byte/domain-for-domain unchanged; exact-content relocation alone does not increment a tower revision, change a span geometry digest, or mark an existing analysis STALE.
6. Preserve historical run dependencies and prior source records so historical runs remain auditable. Existing spans continue to use their captured binding while strong verification proves that the current path supplies the same immutable source content. A future new tower revision, new span, or explicit domain correction may use the current access binding and receives a new binding/geometry digest as required by P1.
7. Validate and publish one new complete generation. If any step fails, the old generation and old access record remain active.

A same-named file with a different identity is rejected, never silently adopted. P2 does not copy the source into the project as part of relocation.

## 21. Integrity / Failure Semantics

Normal loading is fail-closed. Representative outcomes are:

| Condition | Normal behavior |
| --- | --- |
| Malformed root project.json | Explicit integrity failure; diagnostic recovery only. |
| Missing/corrupt generation.json | Active generation rejected; no fallback guess. |
| Unsealed/incomplete generation | Ignored when unreferenced; rejected when active. |
| Mismatched generation ID/digest | Explicit integrity failure. |
| Malformed towers.json, spans.json, or run-index.json | Explicit schema/integrity failure; no partial load. |
| Interrupted temporary write | Temporary sibling is not authoritative; prior active generation remains. |
| Corrupt active pointer | Normal open fails; diagnostic mode may enumerate complete generations for explicit recovery. |
| Missing source | Degraded read-only topology/report view. |
| Source digest mismatch | Candidate/source-dependent action rejected; never adopted. |
| Live project lock | No writable open; read-only fallback or user-visible already-open result. |
| Unlocked old lock metadata | Acquire OS lock safely and replace metadata; never delete by age. |
| Invalid tower/span cross-reference | Explicit integrity failure. |
| Counter less than represented/reserved sequence | Explicit integrity failure; no automatic reduction/fix. |
| Unknown schema version/field | Explicit unsupported-schema failure. |

The loader may expose a diagnostic report containing the failure code, file, generation ID, and safe field/path context. It must not conceal malformed JSON, select a different generation without explicit recovery, or silently repair authoritative data during normal open.

## 22. Security and Path Safety

Every project-relative reference is validated before joining it to a root. The implementation must reuse the P1 hardened relative-POSIX path policy, including rejection of:

- .., absolute paths, UNC paths, drive prefixes, and backslashes;
- colons/NTFS alternate data streams;
- ASCII control characters;
- Windows forbidden filename characters;
- trailing dots/spaces;
- reserved device names such as CON, NUL, COM1, and LPT1;
- symlink/junction escapes when resolving an artifact path for validation;
- references escaping the project root or selected generation.

Generation IDs, lock metadata fields, run IDs, and file names have their own strict bounded validators. External source access paths are explicitly separated from project-relative references. They may be absolute and machine-local because they are audited source access records, but they are never used to construct a project-internal artifact path.

The implementation must not assume a particular drive, username, current working directory, or hostname. Project references are relative and portable; only the audited source access path and lock diagnostics may contain machine-specific values.

## 23. Prediction Persistence Boundary

Prediction remains NOT_CONNECTED by default. The P1 PredictionCapability envelope is serialized/deserialized as ordinary JSON and preserved through round trips, including provider metadata when a future producer supplies it.

P2 does not import ML libraries, discover model files, create prediction result files, or require a prediction provider to load a project. Missing prediction software is not a project-integrity failure. The prediction capability may be shown as unavailable/not connected while all topology, status, and historical metadata remains loadable.

Any future prediction artifact path must use the same safe project-relative reference and generation validation rules, but no prediction directory is created by P2.

## 24. Testing Strategy

Tests must use tmp_path/temporary directories, synthetic contract objects, tiny fixture files, and byte-sized fake source files where source hashing is required. They must not use the 105M-point LAS and must not execute any expensive scientific stage.

Recommended repository-consistent location: tests/project/.

Focused tests:

1. Create a project in a temporary directory and verify the exact physical layout.
2. Save and reopen an exact P1 project round trip.
3. Preserve the same canonical UUID project ID across close/reopen.
4. Preserve T001/T002/T003 and S001/S002 IDs across close/reopen.
5. Reconstruct shared T002 in S001 = T001 -> T002 and S002 = T002 -> T003 without an endpoint copy.
6. Allocate tower IDs monotonically and never reuse an archived T003; the next ID is T004.
7. Allocate span IDs monotonically and never reuse an archived S003.
8. Preserve archived records and reservation-only sequences.
9. Confirm that a failed selection proposal creates no TowerContract record.
10. Confirm that a durable allocation failure can leave a reservation but never a confirmed Tower.
11. Simulate an interrupted temporary file write and verify the old active generation remains valid.
12. Leave an incomplete staging generation and verify normal open ignores it when the old pointer is active.
13. Point the active pointer at an incomplete generation and verify explicit rejection rather than mixed loading.
14. Corrupt generation IDs, file sizes, or digests and verify fail-closed behavior.
15. Corrupt/malformed project, tower, span, source, and run-index JSON and verify no partial aggregate is returned.
16. Verify mismatched generation IDs across files are rejected.
17. Verify a corrupted root active pointer requires explicit diagnostic recovery.
18. Start two writer sessions against one temporary project and verify the second cannot write.
19. Verify stale lock metadata is recoverable only after the OS lock is actually free.
20. Verify valid lock ownership is never overridden by an old timestamp or a misleading PID.
21. Verify missing source opens in the defined degraded read-only state.
22. Verify an identity-mismatched source cannot be adopted for new source-dependent actions.
23. Verify explicit source relocation preserves content identity, topology, IDs, and revisions while changing only the audited access binding and preserving the existing analyses as NON_STALE.
24. Verify unsafe relative paths, traversal, ADS, device names, control characters, and root escapes are rejected.
25. Verify PredictionCapability.not_connected() round trips without importing an ML package.
26. Verify unknown schema versions and unknown fields fail closed.
27. Verify counter values lower than any active, archived, historical, or reserved sequence are rejected and never silently corrected.
28. Verify no test invokes Potree, LAS point materialization, a scientific stage, or a 105M-point fixture.
29. Exercise the writable-open stale-pointer race: B reads an old pointer, A publishes and releases a newer generation, B acquires the writer lock, and B must reload the new pointer/generation before any mutation or save.
30. Verify exact-content source relocation does not mutate historical TowerContract revision content in place.
31. Verify exact-content relocation follows the frozen P1 policy: content identity remains stable, the audited access record changes, existing tower/span bindings and geometry digests do not change, and existing analyses do not become STALE solely because of the move.
32. Verify no public authoritative mutation or ID reservation succeeds without a held writer lock.
33. Verify fast metadata project open does not claim strong source verification.
34. Verify strong SourceIdentity verification is required before a future source-dependent refinement or scientific operation, including invalidation of the session cache after a file signature change.

The interruption tests should exercise the publication boundaries, not rely only on mocks: a test writer may raise after staging, after sealing, and before/after active-pointer replacement, then reopen and assert that exactly one complete generation is selected.

## 25. Proposed File Changes

P2 implementation should remain separate from scientific algorithms:

    core/project/
        __init__.py
        persistence.py       # schemas, strict decode, generations, registries, counters, validation
        locking.py           # Windows msvcrt/POSIX fcntl lock handle and metadata
    tests/project/
        __init__.py
        conftest.py          # tiny valid P1/source fixtures; no large LAS
        test_persistence.py  # creation, round trip, registries, generations, counters, failure cases
        test_locking.py      # single writer, read-only fallback, stale-lock behavior
        test_source_reference.py  # source states and explicit relocation validation

Potential mechanical contract-helper change, only if required during implementation:

    core/contracts/_json.py or core/contracts/project.py
        expose the already-reviewed safe relative-path validator without changing its behavior

That helper extraction is not authorized in this design-only task. No P1 contract field shape should change. No changes are proposed to algorithms/manual_span/, scientific output writers, or existing checkpoint files.

The implementation should use the Python standard library (json, hashlib, os, pathlib, tempfile, uuid, socket, datetime, and platform lock APIs). No database or external locking package is proposed.

## 26. Implementation Subphases

The smallest independently reviewable sequence is:

### P2A — Serialization and coherent generations

- implement strict P1 recursive deserialization;
- freeze canonical JSON bytes and file/logical digest rules;
- implement staging directories, completion manifests, atomic root pointer, and generation validation;
- define project/source/tower/span/run envelope schemas without exposing a mutable ProjectSession or any public authoritative mutation API;
- land round-trip, malformed-input, digest, interruption, and mixed-generation tests.

### P2B — Locking and authoritative project sessions

- implement Windows-safe single-writer lock ownership and diagnostics;
- implement writable/read-only ProjectSession creation, open, close, and reopen;
- acquire the writer lock before every authoritative writable root read and reload the active generation while holding it;
- implement fast source availability states and strong-verification gating;
- land two-writer, stale-pointer race, stale-lock, degraded-open, and source-verification tests.

### P2C — Registry mutation, identities, and relocation

- implement one-file Tower and Span registries;
- implement revision history, archive/tombstone state, reservation entries, and cross-reference validation;
- implement exact monotonic counter reservation/allocation and no-reuse behavior;
- implement lightweight run-index mutation and validation;
- implement explicit strong-verified source relocation without historical Tower/Span mutation;
- land topology, shared-tower, revision, archive, counter, reservation, relocation, and path-safety tests.

Each subphase must preserve the prior complete generation on failure and must be independently reviewable. No subphase starts Potree, creates a viewer cache, imports prediction code, or runs a scientific stage.

The controlling rule is absolute: no public authoritative project mutation, ID reservation, archive/tombstone operation, run-index mutation, or source relocation is permitted without a successfully held project writer lock.

## 27. Acceptance Criteria

The eventual P2 implementation is acceptable only when it proves:

    Create Project P001
      -> allocate T001
      -> allocate T002
      -> create S001 (T001 -> T002)
      -> allocate T003
      -> create S002 (T002 -> T003)
    Close.
    Reopen.

Reopen must recover the same project ID, T001/T002/T003, S001/S002, endpoint revisions, statuses, source binding, and shared T002 topology. After archiving T003, the next new tower must be T004. No old ID may be reused.

The implementation must also demonstrate:

- a second writer is prevented by the project lock;
- a read-only open is possible when the project is locked or the source is missing;
- an interrupted save does not corrupt or mix the last coherent generation;
- malformed schemas and cross-references fail closed;
- source relocation requires verified identity and never silently adopts a same-named mismatch;
- prediction remains optional and NOT_CONNECTED without ML imports;
- no Potree/viewer code and no scientific pipeline behavior changes occur.

## 28. Open Questions

These are follow-up decisions, not blockers for the frozen P2 storage model:

1. Whether a later release should offer an operator-facing generation repair screen, rather than the P2 diagnostic API that requires explicit generation selection.
2. Whether a future managed-source import mode should copy the original LAS/LAZ after an explicit space/portability decision. P2 keeps external read-only reference as the only normal policy.
3. Whether removable/network filesystems need a product-level warning about weaker directory-flush and lock guarantees. The P2 implementation should report platform limitations rather than pretending they do not exist.
4. Whether to expose the safe relative-path helper publicly or keep the compatibility wrapper after parity tests. The validation policy itself is not open.

No open question permits changing the project ID, counter no-reuse rule, generation coherence model, lock ownership rule, source mismatch behavior, or prediction boundary without a new reviewed design decision.

## 29. Explicit Non-Goals

P2 must not implement or alter:

- Potree, PotreeConverter, COPC, viewer cache generation, viewer HTML/JavaScript, pywebview, Qt, or EXE packaging;
- tower clicking, refinement UX, Add First Tower, Add Next Tower, or span corridor visualization;
- scientific span execution or ProjectSpanRunAdapter;
- Ground, HAG, Wire, Vegetation, FRV, tree segmentation, clearance, risk, prediction models, prediction execution, or cross-computer prediction transfer;
- expensive point-cloud processing or scientific pipeline reruns;
- existing manual-span code, working-tree manual-span files, or existing scientific outputs;
- automatic repair of malformed authoritative state;
- copying the original LAS/LAZ during normal project creation;
- a database or third-party locking dependency without a separately approved need.

This document is a design/implementation plan only. It does not authorize P2 code, staging, commit, push, UI work, or scientific execution.
