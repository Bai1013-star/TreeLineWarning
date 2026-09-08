"""Focused research checks; synthetic clouds are not biological ground truth."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "geometry_audit", Path(__file__).with_name("audit.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_order_independence_and_no_mutation():
    rng = np.random.default_rng(9)
    xyz = rng.normal(size=(20000, 3))
    idx = np.arange(len(xyz))
    saved = xyz.copy()
    a = audit.features(xyz, idx)
    order = rng.permutation(len(xyz))
    assert audit.features(xyz[order], idx[order]) == a
    assert np.array_equal(xyz, saved)
    assert audit.membership_digest(idx[order]) == audit.membership_digest(idx)


def test_disconnected_signal_and_no_fake_confidence():
    rng = np.random.default_rng(4)
    xyz = np.concatenate((rng.normal(0, 0.01, (1000, 3)), rng.normal(10, 0.01, (1000, 3))))
    result = audit.diagnostic(audit.features(xyz, np.arange(len(xyz))))
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["issue"] == "GEOMETRY_AMBIGUOUS"
    assert "confidence" not in result


def test_missing_dbh_does_not_imply_bad_or_pass():
    xyz = np.column_stack((np.zeros(1000), np.zeros(1000), np.linspace(0, 10, 1000)))
    assert audit.diagnostic(audit.features(xyz, np.arange(1000)))["status"] == "UNCERTAIN"


def test_relative_geometry_scale_and_translation():
    rng = np.random.default_rng(11)
    xyz = rng.normal(size=(1000, 3))
    a = audit.features(xyz, np.arange(1000))
    b = audit.features(xyz * 7 + 100, np.arange(1000))
    for key in a:
        if "fraction" in key or "ratio" in key or "component" in key or "aspect" in key:
            assert a[key] == pytest.approx(b[key])


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="INVALID_GEOMETRY"):
        audit.features(np.full((3, 3), np.nan), np.arange(3))


def test_reference_labels_preserve_uncertainty():
    manifest = audit.read(Path(__file__).with_name("reference_manifest.json"))
    cases = manifest["cases"]
    assert sum(c["reference_label"] == "EXPLICIT_BAD" for c in cases) == 8
    assert all(
        c["reference_label"] == "UNCERTAIN_REFERENCE"
        for c in cases
        if c["prior_reference_category"] == "EXPLICIT_GOOD"
    )
    assert all(
        c["pre_edit_revision"] and c["post_edit_revision"] and c["evidence_source"] for c in cases
    )


def test_real_benchmark_does_not_hide_misses():
    result = audit.read(Path(__file__).with_name("results.json"))
    bad = [c for c in result["cases"] if c["reference_label"] == "EXPLICIT_BAD"]
    assert len(bad) == result["metrics"]["BAD_TOTAL"] == 8
    assert all(c["prediction"] == audit.diagnostic(c["features"]) for c in bad)
    assert result["metrics"]["BAD_UNCERTAIN"] == 8
    assert result["product_accepted"] is False
