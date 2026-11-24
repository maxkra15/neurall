# Moving cable control with a differentiable impulse model

A 165,456-parameter state-conditioned model predicts a 0.48 s trajectory from one observed moving state of a root-clamped cable. The model receives 17 resampled centerline positions and velocities, cable properties, the prescribed nominal tip load, and two force locations. A GRU predicts nominal motion. A causal response head filters the added loads through four fixed time kernels and learns how those signals move the cable. Its response is linear in the six pulse amplitudes, so one model encoding supplies the exact Jacobian for a bounded least-squares solve.

Training used 600 paired Newton VBD scenes, with 1,200 recordings at eight substeps and 16 sweeps per 20 ms output interval. Training loads used even segment indices. Twelve fresh seeds used odd indices across 10–15-segment cables. Each Newton replay reproduced the full pre-intervention simulation and shared bitwise-identical observed state with its target. The planner receives a desired trajectory. The force amplitudes that generated it stay hidden.

| Planner | Mean Newton replay error as % of target response | Mean absolute tracking error | Median planner time | Median model input evaluations |
| --- | ---: | ---: | ---: | ---: |
| No added force | 100.00% | 3.320 mm | — | 0 |
| Sobol, matched iterative query count | 53.49% | 1.404 mm | 5.50 ms | 24 |
| Sobol, matched iterative wall budget | 40.15% | 1.079 mm | 16.68 ms accepted scoring | 72.5 |
| Bounded quasi-Newton (L-BFGS-B) | 26.48% | 0.792 mm | 16.85 ms | 24 |
| Exact bounded least squares | 26.50% | 0.789 mm | 0.778 ms | 1 |

The one-encoding solve ran 21.7× faster than iterative optimization through the same model, with similar Newton tracking error. It improved all 12 Newton replays over zero added force and beat both sampling controls in 9 of 12. Paired 95% trajectory-bootstrap intervals put its mean difference at −27.0 percentage points [−43.3, −11.9] against count-matched Sobol and −13.6 [−24.3, −3.9] against wall-budget Sobol. The wall budget is set by the iterative planner. Median actual elapsed sampling time was 16.98 ms with late batches discarded.

The nominal 0.48 s forecast averaged 7.39 mm RMS error, compared with 97.76 mm for hold and 95.35 mm for damped velocity. Under the same applied forces as Newton, the learned added-motion response retained 42.81% mean relative RMS error. The learned objective gradient had 0.992 mean cosine alignment with the finite-differenced Newton objective gradient. Its own double-precision autograd and central differences agreed to within 1.64 × 10⁻¹¹. Dataset generation took 141.4 s on eight CPU workers and the recovered fit took 7.17 s over 2,000 updates on one RTX PRO 6000 Blackwell. The run's cached-data loading time is recorded separately from original data generation. Planner timings include search and candidate scoring on one Intel Core Ultra 9 285K CPU thread. Newton replay and offline gradient checks ran separately.

The earlier stock-trained holdout and its fine replay subset used seeds from a separate range. The finer-solver results therefore establish the performance of this matched training and evaluation setup. Solver convergence at yet finer settings, calibrated force-response magnitude, contact, longer free rollouts, deformation beyond cables and any NeRD speed comparison remain open.

[Machine-readable receipt](../../evidence/cable-state-refined-v13.json) · [Planner comparison](../../media/cable-state-refined-v13-planning-comparison.png) · [Replay GIF](../../media/cable-state-refined-v13-moving-cable-replay.gif)

## Reproduce

[Paired synthetic training data](pairs.npz) · [Trained state dictionary](model.pt)

Place `cable_state_control.py`, `plot_cable_state_control.py`, `teachers.py` and `dataset.py` in `experiments/neurall/` of Newton revision `0ced8f8420d4ee1eab9ecfaee59e1354ef670348`. Use Python 3.12, Newton 1.9.0.dev0, Warp 1.19.0.dev20261020, PyTorch 2.15.0+cu132, NumPy 2.5.3 and SciPy 1.18.1 in the existing environment. Figure generation also uses Matplotlib and Pillow.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync \
  -m experiments.neurall.cable_state_control \
  --output cable-state-refined-v13 --model impulse --train-cases 600 \
  --eval-cases 12 --refined-cases 0 --eval-seed-offset 200000 \
  --teacher-refined --workers 8 --steps 2000 --batch 128 --budget 24
uv run --no-sync -m experiments.neurall.plot_cable_state_control cable-state-refined-v13
```
