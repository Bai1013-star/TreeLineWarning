from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from algorithms.spans.reconcile import _exclusive_membership


def test_vectorized_ownership_preserves_core_and_no_core_tie_rules():
    outcomes = [
        SimpleNamespace(span=SimpleNamespace(line_id=2, span_id=5)),
        SimpleNamespace(span=SimpleNamespace(line_id=1, span_id=9)),
    ]
    original = np.asarray((10, 10, 20, 20, 30, 40, 40), dtype=np.int64)
    xyz = np.column_stack((original, np.zeros(7), np.ones(7)))
    span = np.asarray((0, 1, 0, 1, 0, 0, 1), dtype=np.int32)
    global_tree = np.asarray((5, 2, 3, 4, 7, 9, 8), dtype=np.int32)
    in_core = np.asarray((False, True, True, True, True, False, False))

    final_original, final_xyz, final_tree, conflicts = _exclusive_membership(
        original, xyz, span, global_tree, in_core, outcomes
    )

    np.testing.assert_array_equal(final_original, (10, 20, 30, 40))
    np.testing.assert_array_equal(final_xyz, xyz[[0, 2, 4, 5]])
    # 10: only span 1 is core. 20: both are core and line 1 wins. 40: neither
    # is core, so the canonical smallest global id wins.
    np.testing.assert_array_equal(final_tree, (2, 4, 7, 8))
    assert conflicts == 3


def test_vectorized_ownership_handles_an_empty_membership():
    original, xyz, tree, conflicts = _exclusive_membership(
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=bool),
        [],
    )

    assert original.shape == (0,)
    assert xyz.shape == (0, 3)
    assert tree.shape == (0,)
    assert conflicts == 0
