# I1 cross-computer bundle interface

I1 is the immutable, portable boundary between the completed main-computer W1/V2 work and later prediction-computer work. It contains references to copied, verified source artifacts; it does not combine or duplicate point arrays and it does not implement C1, R1, P1, V3, or a prediction adapter.

## Layout and versions

```text
<bundle-id>/
  i1_manifest.json
  artifacts/w1/<source>.npz
  artifacts/w1/<source>.json
  artifacts/v2/<source>.npz
  artifacts/v2/<source>.json
  artifacts/v2a/<source>.npz     # optional
  artifacts/v2a/<source>.json    # optional
```

`i1_manifest.json` is written last and is the completion marker. I1 v1 uses schema `i1-bundle-v1`, interface version `1.0.0`, and eligibility-policy version `i1-eligibility-v1`. It accepts exactly W1 `wire-models-v2` / `w1-confirmed-wire-centreline-v3`, V2 `vegetation-instances-v2` / `v2-generalized-morphology-safe-vegetation-instances-v2`, and optional V2A `vegetation-associations-v1` / `v2a-review-level-vegetation-association-v1`.

## Coordinates and source provenance

Only local Cartesian XYZ coordinates in metres are supported:

```text
frame: LOCAL_CARTESIAN_XYZ
unit: m
transform: identity
```

During export, I1 recovers W1's origin from the W1-declared, hash-verified A4 JSON metadata. It compares that finite float64 XYZ origin exactly to V2's authoritative origin. The manifest records the origin, source kind/version/hash, and successful exact-match verification. Global coordinates, mismatched origins, non-identity transforms, unknown units, and conversion attempts are rejected.

## Integrity and bundle ID

Every source sibling has a normalized bundle-relative path, SHA-256 digest, byte size, schema version, and algorithm version. Paths must remain inside `artifacts/<kind>/`; absolute paths, Windows separators, and traversal are invalid.

The deterministic `bundle_id` is SHA-256 over canonical JSON containing I1 versions, eligibility policy, source/manifest identity, coordinate context, required W1/V2 references, and optional V2A state/reference. It excludes destination, timestamps, user names, and machine paths. The completed directory is named with that ID.

## Eligibility

Formal wire segments are exposed only when their parent is `CONFIRMED_MODEL`, `segment_distance_ready` is true, and the sampled segment has at least two finite points. Published segment boundaries and gap arrays remain authoritative; nothing may bridge a gap. A non-continuous wire can therefore expose ready segments, but a later adapter must report partial coverage.

Formal vegetation is exposed only when its cluster is `CONFIRMED`, has a non-negative vegetation ID, has non-empty member rows, and every member row agrees with the cluster/vegetation metadata, is distance-ready, and has finite local XYZ. Review, provisional, rejected, unassigned, empty, or inconsistent records are excluded rather than repaired.

V2A is optional audit/UI metadata. Its absence never blocks I1, and it cannot affect eligibility, merge vegetation, or create identity.

## Publication, reload, and transfer

`export_i1_bundle` validates original sibling pairs through their public loaders, validates identity/origin compatibility, copies actual files into a same-volume temporary directory, re-hashes every copy, writes and validates the manifest, then atomically renames the directory. Existing equivalent bundles are reloaded and reused; they are never overwritten. Failure or cancellation removes only that exporter invocation's temporary directory.

`load_i1_bundle` is standalone: it requires the completion manifest, validates the ID, paths, hashes, sizes, source loaders, source identity, coordinate context, and optional V2A lineage. It does not rely on the original project path and continues to work after moving the complete bundle to another computer.

Transfer the immutable directory outside Git (runtime bundles remain untracked). Transfer I1 code through reviewed Git commits. The future `feature/prediction` adapter must consume only validated I1 source views and eligibility helpers; it must not import W1/V2/V2A internals or parse their raw filenames directly.

## Failure semantics

I1 raises typed `I1Error` values with stable categories for cancellation, contract/path/version errors, source validation, coordinate mismatch, integrity failure, publication failure, and immutable-bundle conflict. No fallback coordinate conversion, stale-file selection, or partial publication recovery occurs.
