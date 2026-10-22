import numpy as np
import pytest

from algorithms.spans.wire import _parallel_family_interior_rescue
from algorithms.spans.wire_v2 import CONFIRMED, reconstruct_slice_tracks
from core.config.fast_baseline import FastBaselineConfig


@pytest.mark.parametrize('count', [3, 4])
def test_three_or_four_conductors_with_short_occlusion_are_not_capped_at_two(count):
    arrays = []
    for wire in range(count):
        s = np.arange(.1, 30, .2)
        if wire == count - 1:
            s = s[(s < 13) | (s > 15)]
        arrays.append(np.column_stack((s, np.full(len(s), wire * 1.2), 8 + .001 * (s - 15)**2)))
    xyz = np.concatenate(arrays)
    tracks = reconstruct_slice_tracks(np.arange(len(xyz)), xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                      np.ones(len(xyz), dtype=np.int64), 30, FastBaselineConfig())
    confirmed = [t for t in tracks if t.status == CONFIRMED]
    assert len(confirmed) == count
    assert all(t.maximum_gap_m <= 5 for t in confirmed)
    assert len(np.unique(np.concatenate([t.inlier_rows for t in confirmed]))) == sum(len(t.inlier_rows) for t in confirmed)


def test_long_but_unstable_clutter_does_not_become_a_confirmed_conductor():
    s = np.arange(.1, 30, .2)
    u = 3 * np.sin(s)
    z = 8 + 2 * np.sin(2 * s)
    tracks = reconstruct_slice_tracks(np.arange(len(s)), s, u, z, np.ones(len(s), dtype=np.int64),
                                      30, FastBaselineConfig())
    assert not any(t.status == CONFIRMED for t in tracks)


def test_third_long_conductor_can_use_strong_parallel_family_interior_evidence():
    from types import SimpleNamespace

    config = FastBaselineConfig()
    strong = SimpleNamespace(
        status=CONFIRMED,
        coverage_ratio=.94,
        supported_slice_count=38,
        supported_slice_ratio=.90,
        bounded_supported_slice_ratio=.90,
        maximum_gap_m=2.94,
        inlier_ratio=.92,
        median_residual_m=.08,
        p95_residual_m=.22,
        supported_s_min_m=0., supported_s_max_m=30., model_type='ROBUST_QUADRATIC',
        model_u_coefficients=np.array([15.,30.,0.,0.]),
        model_z_coefficients=np.array([15.,30.,0.,0.,11.]),
    )
    left = SimpleNamespace(**{**strong.__dict__, 'model_u_coefficients':np.array([15.,30.,0.,-1.])})
    right = SimpleNamespace(**{**strong.__dict__, 'model_u_coefficients':np.array([15.,30.,0.,1.])})
    assert _parallel_family_interior_rescue(strong, [strong, left, right], config)
    assert not _parallel_family_interior_rescue(strong, [strong, left], config)
    weak = SimpleNamespace(**{**strong.__dict__, 'p95_residual_m': .9})
    assert not _parallel_family_interior_rescue(weak, [left, right, weak], config)
    lower = SimpleNamespace(**{**strong.__dict__, 'model_z_coefficients':np.array([15.,30.,0.,0.,5.8])})
    assert not _parallel_family_interior_rescue(lower, [left, right, lower], config)


def test_robust_conductor_ignores_sparse_attachment_outliers_without_joining_lower_cable():
    s=np.arange(.1,44,.2)
    clouds=[]
    for lateral,height in [(-1.,11.4),(0.,11.45),(1.,11.5),(.1,5.8)]:
        u=np.full(len(s),lateral)
        if lateral==-1.:
            u[(s>19)&(s<22)]-=.6
        clouds.append(np.column_stack((s,u,height+.001*(s-22)**2)))
    xyz=np.concatenate(clouds)
    config=FastBaselineConfig()
    tracks=reconstruct_slice_tracks(np.arange(len(xyz)),xyz[:,0],xyz[:,1],xyz[:,2],np.ones(len(xyz),dtype=np.int64),44,config)
    upper=[track for track in tracks if track.status==CONFIRMED and np.median(xyz[track.inlier_rows,2])>10]
    assert len(upper)==3
    assert all(track.p95_residual_m<.1 for track in upper)
    lower=next(track for track in tracks if track.status==CONFIRMED and np.median(xyz[track.inlier_rows,2])<7)
    assert not _parallel_family_interior_rescue(lower,tracks,config)
