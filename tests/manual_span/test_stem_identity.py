import numpy as np
import pytest
from algorithms.manual_span import stem_identity as identity


def cylinder(tilt=0, offset=(0, 0, 0)):
    angle = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    unit = np.array([np.sin(np.radians(tilt)), 0, np.cos(np.radians(tilt))])
    u = np.array([unit[2], 0, -unit[0]])
    xyz = np.vstack(
        [
            unit * h / unit[2]
            + 0.15 * np.cos(angle)[:, None] * u
            + 0.15 * np.sin(angle)[:, None] * np.array([0, 1, 0])
            for h in np.arange(0.85, 3.51, 0.025)
        ]
    )
    hag = xyz[:, 2].copy()
    xyz += offset
    axis = {
        "bh_reference_xyz": (unit * 1.3 / unit[2] + offset).tolist(),
        "direction_xyz": unit.tolist(),
        "slope_xyz_per_hag": (unit / unit[2]).tolist(),
        "radius_m": 0.15,
    }
    return xyz, hag, axis


@pytest.mark.parametrize("tilt", [0, 12])
def test_crown_to_bh_track_and_unchanged_normal_fit(tilt):
    xyz, hag, axis = cylinder(tilt, (30, 50, 12))
    track = identity.track_downward(xyz, hag, np.arange(len(xyz)), axis)
    assert track["status"] == "PASS"
    assert [r["hag_target_m"] for r in track["layers"]] == sorted(identity.LAYERS, reverse=True)
    assert track["refined_axis"]["anchor_status"] == "INTERPOLATED_FROM_TRACK_SPANNING_TRUE_BH"
    result = identity.fit_identified_stem(xyz, hag, track["refined_axis"])
    assert result["dbh_cm"] == pytest.approx(30, abs=0.15)
    assert result["dbh_method"] == "RECOVERED_STEM"


def test_missing_intermediate_layer_cannot_bridge_crown_to_bh():
    xyz, hag, axis = cylinder()
    keep = np.abs(hag - 1.8) > 0.15
    track = identity.track_downward(xyz[keep], hag[keep], np.flatnonzero(keep), axis)
    assert track["status"] == "CROWN_TO_BH_TRACK_INCOMPLETE"
    assert track["refined_axis"] is None


def test_upper_proposal_must_track_down_before_using_bh():
    xyz, hag, _ = cylinder(8)
    axes, proof = identity.upper_axes(xyz, hag)
    assert proof["status"] == "PASS"
    assert axes[0]["anchor_status"] == "EXTRAPOLATED_PROPOSAL_ONLY_REQUIRES_DOWNWARD_TRACK"
    with pytest.raises(ValueError, match="MUST_ESTABLISH_TRUE_BH"):
        identity.fit_identified_stem(xyz, hag, axes[0])
    track = identity.track_downward(xyz, hag, np.arange(len(xyz)), axes[0])
    assert track["status"] == "PASS"


def test_neighboring_crown_cannot_pass_from_bh_distance_alone():
    xyz, hag, axis = cylinder()
    track = identity.track_downward(xyz, hag, np.arange(len(xyz)), axis)
    crown = identity.crown_anchor(xyz[hag >= 2], hag[hag >= 2])
    own = identity.score_crown(track, axis, crown, {"status": "PASS"}, {"status": "PASS"})
    assert own["absolute_pass"] and own["score"] >= 0.75
    other = identity.crown_anchor(xyz[hag >= 2] + [4, 0, 0], hag[hag >= 2])
    foreign = identity.score_crown(track, axis, other, {"status": "PASS"}, {"status": "PASS"})
    assert not foreign["absolute_pass"]
    assert not foreign["footprint_intersection"]


def test_no_upper_structure_has_no_axis():
    axes, proof = identity.upper_axes(np.empty((0, 3)), np.empty(0))
    assert axes == [] and proof["status"] == "NO_UPPER_AXIS"


def test_downward_following_reanchors_upper_proposal_at_actual_bh():
    xyz, hag, axis = cylinder()
    tilt = np.radians(4)
    axis["direction_xyz"] = [np.sin(tilt), 0, np.cos(tilt)]
    axis["slope_xyz_per_hag"] = [np.tan(tilt), 0, 1]
    axis["bh_reference_xyz"] = [-2 * np.tan(tilt), 0, 1.3]
    axis["anchor_status"] = "EXTRAPOLATED_PROPOSAL_ONLY_REQUIRES_DOWNWARD_TRACK"
    # Correct at upper HAG 3.30; the unmeasured lower extrapolation is wrong.
    # Consecutive measured centers need only a 0.0175 m tracking correction.
    tracked = identity.track_downward(xyz, hag, np.arange(len(xyz)), axis)
    assert tracked["status"] == "PASS"
    assert tracked["refined_axis"]["bh_reference_xyz"][:2] == pytest.approx([0, 0], abs=0.002)
