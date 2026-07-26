"""Unit tests for the I1 manifest-only contracts."""

import pytest
from core.contracts import (
    I1_ELIGIBILITY_POLICY_VERSION,
    I1_INTERFACE_VERSION,
    I1_SCHEMA_VERSION,
    I1ArtifactKind,
    I1ArtifactReference,
    I1BundleManifest,
    I1CoordinateContext,
    I1CoordinateFrame,
    I1CoordinateUnit,
    I1OptionalArtifactState,
    calculate_bundle_id,
)


def _reference(kind: I1ArtifactKind) -> I1ArtifactReference:
    return I1ArtifactReference(
        kind,
        f"artifacts/{kind.value}/{kind.value}.npz",
        f"artifacts/{kind.value}/{kind.value}.json",
        "a" * 64,
        1,
        "b" * 64,
        2,
        f"{kind.value}-schema",
        f"{kind.value}-algorithm",
    )


def _context() -> I1CoordinateContext:
    return I1CoordinateContext(
        (1.0, 2.0, 3.0),
        I1CoordinateFrame.LOCAL_CARTESIAN_XYZ,
        I1CoordinateUnit.METRE,
        "identity",
        "w1_a4_json",
        "wire-separation-result-v1",
        "a4-test",
        "c" * 64,
        "verified_exact_match",
    )


def _manifest() -> I1BundleManifest:
    context, w1, v2 = _context(), _reference(I1ArtifactKind.W1), _reference(I1ArtifactKind.V2)
    bundle_id = calculate_bundle_id(
        source_id="scene",
        manifest_source_id="manifest",
        coordinate_context=context,
        w1=w1,
        v2=v2,
        v2a_state=I1OptionalArtifactState.ABSENT,
        v2a=None,
    )
    return I1BundleManifest(
        I1_SCHEMA_VERSION,
        I1_INTERFACE_VERSION,
        bundle_id,
        "scene",
        "manifest",
        I1_ELIGIBILITY_POLICY_VERSION,
        context,
        w1,
        v2,
        I1OptionalArtifactState.ABSENT,
        None,
    )


def test_manifest_roundtrip_and_deterministic_identifier() -> None:
    manifest = _manifest()
    assert I1BundleManifest.from_dict(manifest.to_dict()) == manifest
    assert _manifest().bundle_id == manifest.bundle_id


@pytest.mark.parametrize(
    "path",
    ("C:/bad.npz", "../bad.npz", "artifacts/w1/../bad.npz", "artifacts\\w1\\bad.npz"),
)
def test_artifact_reference_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        I1ArtifactReference(
            I1ArtifactKind.W1, path, "artifacts/w1/a.json", "a" * 64, 1, "b" * 64, 1, "s", "a"
        )


def test_manifest_rejects_unknown_field_and_noncanonical_id() -> None:
    raw = _manifest().to_dict() | {"unexpected": True}
    with pytest.raises(ValueError):
        I1BundleManifest.from_dict(raw)
    raw = _manifest().to_dict() | {"bundle_id": "d" * 64}
    with pytest.raises(ValueError):
        I1BundleManifest.from_dict(raw)
