# W1 confirmed-wire reconstruction review — support-relative revision

## Baseline and scope

- Branch: `feature/algorithm`; baseline: `7070fee feat: add wire instance separation and confirmation`.
- Interpreter: `E:\Miniforge\envs\treeline\python.exe`; NumPy 2.3.5; SciPy 1.17.1.
- Algorithm/schema versions: `w1-confirmed-wire-centreline-v3` / `wire-models-v2`.
- No commit, staging, push, merge, branch change, dependency installation, LAS/LAZ read, or A3-output read occurred. A4 files and project dependency files were not changed.

Changed tracked files are `core/contracts/wire_models.py`, `core/contracts/__init__.py`, `algorithms/wires/reconstruct.py`, `algorithms/wires/__init__.py`, `tests/contracts/test_wire_model_contracts.py`, `tests/wires/test_reconstruct.py`, and this review. The two `__init__.py` changes are existing W1 public exports; this revision did not add an export.

## Corrected terminology and formulas

The prior diagnostic named `curvature` was actually a local consecutive-step turning angle, not physical geometric curvature. W1 now uses `turn_angle_p95_rad`, `turn_angle_max_rad`, `maximum_turn_angle_rad`, and `maximum_turn_angle_spike_multiplier`.

For an ordered observation \(x_i\), local validated unit direction \(d\), and anchor \(a\), projected support parameter is \(p_i=(x_i-a)\cdot d\) metres. Within each deterministic representative bin, weights are \(w_i=\max(assignment\_score_i,0.05)\), representative XYZ is \(r_b=\sum w_i x_i/\sum w_i\), and representative parameter is \(t_b=\sum w_i p_i/\sum w_i\). Cubic fitting uses \(u_b=(t_b-t_0)/(t_{last}-t_0)\).

For adjacent published curve steps \(e_j\), the local shape diagnostic is:

\[
\theta_j=\arccos(\operatorname{clip}(\hat e_j\cdot\hat e_{j+1},-1,1))\quad[\mathrm{rad}].
\]

It is sample-spacing dependent: for small steps it is approximately physical curvature times local step length. It is retained only as a local shape diagnostic, not interpreted as inverse-metre curvature.

Representative support polyline length is \(L_r=\sum\lVert r_{b+1}-r_b\rVert\). Candidate length is \(L_c=\sum\lVert c_{j+1}-c_j\rVert\). The dimensionless support-relative inflation is \(I=L_c/\max(L_r,10^{-12})\). Projected observed span remains separate and is not used as an arc-length baseline.

## Support-parameter residual and acceptance

For a cubic candidate \(C(u)\), W1 evaluates it directly at representative parameters and at each original observation's normalized projected parameter. It computes XYZ distances \(\lVert C(u)-r_b\rVert\) and \(\lVert C(u_i)-x_i\rVert\), combines both populations, and reports RMSE, median, p95, and maximum in metres. This combined parameter-matched support residual controls acceptance. No dense point-by-sample matrix is built.

Nearest distance to published samples remains a separately named preview/reload diagnostic only: `nearest_sample_residual_stats_m`. It no longer controls fitting acceptance.

A candidate is accepted when finite geometry, strictly increasing in-domain parameter samples, no longitudinal reversal, support residual p95 <= `maximum_residual_m`, and no support-relative oscillation rejection are all true. Warnings are retained for `turn_angle_spike_detected`, `sample_step_spike_detected`, and `unsupported_arc_inflation_detected`. Rejection is:

\[
(\text{turn-angle spike}\ \lor\ \text{sample-step spike})\ \land\ \text{unsupported arc inflation}.
\]

Non-finite geometry, invalid parameterization, longitudinal reversal, and excessive physical support residual remain independent failures. A selected candidate with shape warnings but no unsupported inflation is exposed as `REVIEW_REQUIRED`, rather than silently presented as high confidence.

Configuration defaults: `maximum_residual_m=0.15` m (physical tolerance), `maximum_turn_angle_rad=1.25` rad, `maximum_turn_angle_spike_multiplier=3.0`, `maximum_sample_step_multiplier=2.0`, and `maximum_arc_length_inflation_ratio=1.02`. The 1.02 ratio is dimensionless: ordinary synthetic and real supported curves were <= 1.0043, while deterministic unsupported interpolation was > 1.0235 and wire 4's rejected candidate was 1.0328.

## Synthetic and scale checks

All runs use deterministic analytic/support data and the production W1 path. Straight 5/15/40 m, shallow/moderate/strong sags, uneven density, isolated outliers, and row-reordered equivalents retained the least-smoothed valid cubic. The output-spacing test at 0.05 m and 0.10 m selected `s=0.0` both times; parameter-matched support residual was unchanged while nearest-published-sample residual changed, as expected.

Scale-equivalent noisy sag tests at 0.5x/1x/2x, with spatial configuration and noise scaled consistently, all selected `s=0.0`, remained ready, and had inflation ratios 1.00048/1.00046/1.00047. A smooth tight supported bend generated a turn-angle warning at `s=0.0`, inflation 1.00185, and remained accepted; it was not rejected solely for turning angle. The deterministic unsupported interpolation case generated spike warnings plus inflation > 1.02, rejected `s=0.0`, and selected `s=0.002`. Alternating support noise produced a turn-angle warning without unsupported inflation and is marked `REVIEW_REQUIRED`, rather than being attributed only to spline oscillation. Existing large-gap coverage continues to verify distinct segments and no across-gap samples.

## Sensitivity

Synthetic sensitivity used straight, sag, uneven-density, isolated-oscillation, alternating-noise, and tight-bend cases. Varying turn-angle ratio from 2.5 to 3.5 and sample-step multiplier from 1.6 to 2.4 changed no selected synthetic or real candidates. Arc-inflation ratios 1.01 and 1.015 changed none; 1.025–1.035 changed only the deliberately unsupported synthetic case. For real `scene_simple`, 1.01–1.03 changed none; 1.035 changed only wire 4 to `s=0.0` with `REVIEW_REQUIRED` warnings. Thus wire 4 is near, but does not uniquely define, the default: the independent synthetic unsupported case also changes only after the protective ratio is relaxed.

## Real `scene_simple` rerun

Input was confirmed-only A4 from `outputs/wires/scene_simple_wire_instances.npz` SHA-256 `86bbd912ee281ab938963c153467e75e986afec1c37dc0d7d92cbbb26a922583` and JSON SHA-256 `63b08e9bd5326b83ecafae2b00b14be7da6411773ddfd82f63511b4892636c81`. It contained exactly preserved IDs 0–7: eight independent wires, eight ready observed segments, zero gaps, and no provisional/review A4 tracks.

| wire | s | support RMSE / p95 / max (m) | rep arc / candidate arc / ratio | turn p95 / max (rad) | step median / p95 / max (m) | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.000 | .02616 / .04877 / .06695 | 14.264 / 14.276 / 1.00081 | .28364 / .44956 | .04968 / .05117 / .05310 | CONFIRMED_MODEL |
| 1 | 0.000 | .02831 / .05220 / .12853 | 14.457 / 14.470 / 1.00091 | .24930 / .46617 | .04968 / .05102 / .05508 | CONFIRMED_MODEL |
| 2 | 0.000 | .02711 / .04977 / .12952 | 15.319 / 15.331 / 1.00077 | .21419 / .56714 | .04972 / .05104 / .05511 | CONFIRMED_MODEL |
| 3 | 0.000 | .02545 / .04724 / .08032 | 15.745 / 15.756 / 1.00067 | .24341 / .46223 | .04984 / .05085 / .05295 | CONFIRMED_MODEL |
| 4 | .002 | .05588 / .11096 / .24345 | 14.687 / 13.474 / .91743 | .00945 / .01191 | .04573 / .04626 / .04807 | CONFIRMED_MODEL |
| 5 | 0.000 | .02679 / .04815 / .12967 | 16.087 / 16.099 / 1.00075 | .27258 / .39124 | .04979 / .05116 / .05429 | CONFIRMED_MODEL |
| 6 | 0.000 | .02652 / .04833 / .12817 | 15.857 / 15.867 / 1.00063 | .23860 / .38964 | .04970 / .05091 / .05277 | CONFIRMED_MODEL |
| 7 | 0.000 | .02502 / .04538 / .07129 | 15.779 / 15.791 / 1.00071 | .22067 / .48313 | .04981 / .05065 / .05880 | CONFIRMED_MODEL |

Wire 4 comparison: its original `s=0.0` candidate has support p95 .09160 m, turn p95/max .75542/1.74132 rad, maximum step .14918 m, representative/candidate arc 14.6868/15.1683 m, ratio 1.03278, and is rejected as `unsupported_oscillation`. Its prior and current least accepted `s=.002` candidate has support p95 .11096 m, turn p95/max .00945/.01191 rad, maximum step .04807 m, arc ratio .91743, no warnings, and `CONFIRMED_MODEL`. Wires 0–3 and 5–7 retain `s=0.0`, cubic methods, status, and readiness.

Runtime: 0.170 s; RSS start/peak/end: 77,074,432 / 88,842,240 / 88,850,432 bytes. Standalone reload passed. Formal NPZ is `outputs/wire_models/scene_simple_wire_models.npz`, 217,097 bytes, SHA-256 `060257f378c4b6598f06fe7c9c558e97e8dc73f0fa57ea2a4861e5519c7fb870`; JSON and the standalone preview are published under `outputs/wire_models/` and `outputs/previews/`. Outputs remain untracked. Human visual inspection of the regenerated preview remains required.

## Validation and limitations

Validation results: targeted W1 contract/reconstruction tests `25 passed`; full `pytest` `178 passed` in 4.68 s; `ruff check .` passed; `compileall core algorithms tests` passed; and `git diff --check` passed. `ruff format --check .` reports only the known unrelated formatting differences in `algorithms/io/samples.py`, `algorithms/preprocessing/preview.py`, and `core/contracts/pointcloud.py`; they are not changed. `git ls-files data outputs` and the prohibited-artifact pattern check found no tracked data/output artifacts.

The algorithm is designed and synthetically tested for scale-aware general behavior, but multi-scene real-world generalization is not proven because the project still has only one primary real development scene. Endpoint uncertainty remains diagnostic-only; representative support may itself contain real or noisy local variation, so `REVIEW_REQUIRED` warnings should be reviewed before production use.
