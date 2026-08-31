"""Focused tests for the deterministic V3 stem ownership resolver."""

from __future__ import annotations

import json

import numpy as np
import pytest
from algorithms.manual_span.stem_ownership import (
    AXIS_SHARED_DISTANCE_M,
    MIN_IDENTITY_SCORE,
    MIN_SCORE_MARGIN,
    resolve_stem_ownership,
)


def _candidate(
    tree_id: int,
    candidate_id: str,
    *,
    score: float = 0.90,
    absolute_pass: bool = True,
    origin: tuple[float, float, float] = (0.0, 0.0, 1.30),
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    indices: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    return {
        "tree_id": tree_id,
        "candidate_id": candidate_id,
        "score": score,
        "absolute_pass": absolute_pass,
        "axis_origin": list(origin),
        "axis_direction": list(direction),
        "original_index": np.asarray(indices, dtype=np.int64),
    }


def test_nonconflicting_axes_are_selected_jointly() -> None:
    result = resolve_stem_ownership(
        [
            _candidate(1, "a", origin=(0.0, 0.0, 1.30), indices=(1, 2)),
            _candidate(2, "b", origin=(1.0, 0.0, 1.30), indices=(3, 4)),
        ]
    )
    assert result["selected"] == {1: "a", 2: "b"}
    assert result["tree_status"] == {1: "SELECTED", 2: "SELECTED"}
    assert result["decisions"]["a"]["status"] == "SELECTED"
    assert result["decisions"]["b"]["conflicts"] == []
    json.dumps(result)


@pytest.mark.parametrize(
    ("origin", "direction", "indices"),
    [
        ((0.10, 0.0, 1.30), (0.0, 0.0, 1.0), (9, 10)),
        ((2.0, 0.0, 1.30), (0.0, 0.0, 1.0), (1, 2)),
    ],
)
def test_shared_axis_or_identity_indices_make_both_trees_unavailable(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    indices: tuple[int, ...],
) -> None:
    result = resolve_stem_ownership(
        [
            _candidate(1, "a", indices=(1, 2)),
            _candidate(2, "b", origin=origin, direction=direction, indices=indices),
        ]
    )
    assert result["selected"] == {}
    assert result["tree_status"] == {1: "TOP_CANDIDATE_CONFLICT", 2: "TOP_CANDIDATE_CONFLICT"}
    assert result["decisions"]["a"]["status"] == "INELIGIBLE_CONFLICT"
    assert result["decisions"]["b"]["status"] == "INELIGIBLE_CONFLICT"
    assert result["decisions"]["a"]["conflicts"] == ["b"]
    assert result["decisions"]["b"]["conflicts"] == ["a"]


def test_score_gate_and_exact_margin_boundary() -> None:
    assert MIN_IDENTITY_SCORE == pytest.approx(0.75)
    assert MIN_SCORE_MARGIN == pytest.approx(0.10)
    boundary = resolve_stem_ownership(
        [
            _candidate(1, "high", score=0.85, indices=(1,)),
            _candidate(1, "low", score=0.75, origin=(1.0, 0.0, 1.30), indices=(2,)),
            _candidate(
                2,
                "threshold",
                score=MIN_IDENTITY_SCORE,
                origin=(3.0, 0.0, 1.30),
                indices=(3,),
            ),
            _candidate(
                3,
                "below",
                score=MIN_IDENTITY_SCORE - 0.001,
                origin=(4.0, 0.0, 1.30),
                indices=(4,),
            ),
        ]
    )
    assert boundary["selected"][1] == "high"
    assert boundary["selected"][2] == "threshold"
    assert 3 not in boundary["selected"]
    assert boundary["tree_status"][3] == "IDENTITY_SCORE_BELOW_MINIMUM"

    margin_failure = resolve_stem_ownership(
        [
            _candidate(4, "top", score=0.84, indices=(10,)),
            _candidate(
                4,
                "runner",
                score=MIN_IDENTITY_SCORE,
                origin=(1.0, 0.0, 1.30),
                indices=(11,),
            ),
        ]
    )
    assert margin_failure["selected"] == {}
    assert margin_failure["tree_status"][4] == "SCORE_MARGIN_BELOW_MINIMUM"

    below_gate_runner = resolve_stem_ownership(
        [
            _candidate(5, "winner", score=0.81, indices=(12,)),
            _candidate(
                5,
                "close-runner",
                score=0.74,
                origin=(1.0, 0.0, 1.30),
                indices=(13,),
            ),
        ]
    )
    assert below_gate_runner["selected"] == {}
    assert below_gate_runner["tree_status"][5] == "SCORE_MARGIN_BELOW_MINIMUM"


def test_fixed_claim_reserves_an_axis_without_reassignment() -> None:
    fixed = _candidate(10, "fixed", origin=(0.0, 0.0, 1.30), indices=(40,))
    result = resolve_stem_ownership(
        [
            _candidate(
                11,
                "blocked",
                origin=(AXIS_SHARED_DISTANCE_M / 2.0, 0.0, 1.30),
                indices=(41,),
            ),
            _candidate(12, "free", origin=(2.0, 0.0, 1.30), indices=(42,)),
        ],
        fixed=(fixed,),
    )
    assert result["selected"] == {10: "fixed", 12: "free"}
    assert result["tree_status"][10] == "FIXED_RESERVATION"
    assert result["decisions"]["fixed"]["status"] == "FIXED"
    assert result["decisions"]["blocked"]["status"] == "INELIGIBLE_CONFLICT"
    assert result["decisions"]["blocked"]["conflicts"] == ["fixed"]


def test_global_ordering_is_deterministic() -> None:
    candidates = [
        _candidate(1, "z", score=0.82, origin=(0.0, 0.0, 1.30), indices=(1,)),
        _candidate(1, "a", score=0.82, origin=(0.5, 0.0, 1.30), indices=(2,)),
        _candidate(2, "b", score=0.91, origin=(2.0, 0.0, 1.30), indices=(3,)),
    ]
    fixed = (
        _candidate(3, "fixed-z", origin=(4.0, 0.0, 1.30), indices=(4,)),
    )
    first = resolve_stem_ownership(candidates, fixed=fixed)
    reordered = resolve_stem_ownership(
        list(reversed(candidates)), fixed=tuple(reversed(fixed))
    )
    assert reordered == first
    assert first["tree_status"][1] == "SCORE_MARGIN_BELOW_MINIMUM"
    assert first["selected"][2] == "b"


def test_top_conflict_does_not_fall_through_to_lower_score_candidate() -> None:
    result = resolve_stem_ownership(
        [
            _candidate(1, "top", score=0.90, origin=(0.0, 0.0, 1.30), indices=(1,)),
            _candidate(1, "fallback", score=0.79, origin=(1.0, 0.0, 1.30), indices=(2,)),
            _candidate(2, "neighbor", score=0.95, origin=(0.10, 0.0, 1.30), indices=(3,)),
        ]
    )
    assert result["selected"] == {}
    assert result["tree_status"][1] == "TOP_CANDIDATE_CONFLICT"
    assert result["decisions"]["fallback"]["status"] == "INELIGIBLE_NOT_SELECTED"


@pytest.mark.parametrize(
    "bad",
    [
        {"score": 1.1},
        {"axis_direction": [0.0, 0.0, 2.0]},
        {"axis_origin": [0.0, 0.0]},
        {"original_index": np.asarray([[1]], dtype=np.int64)},
        {"original_index": np.asarray([np.nan])},
        {"original_index": np.asarray([-1, 2], dtype=np.int64)},
        {"original_index": np.asarray([2, 2], dtype=np.int64)},
    ],
)
def test_malformed_candidates_fail_closed(bad: dict[str, object]) -> None:
    candidate = _candidate(1, "bad")
    candidate.update(bad)
    result = resolve_stem_ownership([candidate])
    assert result["selected"] == {}
    assert result["decisions"]["bad"]["status"] == "INVALID"
    assert result["tree_status"][1] == "INVALID_INPUT"
