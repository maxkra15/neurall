# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn cable response from an observed moving state and plan forces through it."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch  # noqa: TID253 -- standalone learning experiment
from scipy.optimize import lsq_linear, minimize  # noqa: TID253 -- standalone learning experiment
from scipy.stats import qmc  # noqa: TID253 -- standalone learning experiment
from torch import nn  # noqa: TID253 -- standalone learning experiment

from .dataset import digest

_DT = 0.02
_HORIZON = 24
_NODES = 17
_BOUND = 7.0
_SEED = 9_300_000
_PREDICTION_TIME = _DT * np.arange(1, _HORIZON + 1)
_ARC = np.linspace(0, 1, _NODES)


def _setup(seed, segments, *, held_out):
    rng = np.random.default_rng(seed + 1_000_000)
    start = int(rng.integers(12, 42))
    parity = int(held_out)
    left = [index for index in range(2, segments // 2 + 1) if index % 2 == parity]
    right = [index for index in range(segments // 2 + 1, segments + 1) if index % 2 == parity]
    sites = (int(rng.choice(left)), int(rng.choice(right)))
    pulses = []
    clock = _DT * np.arange(_HORIZON)
    for onset, duration in (
        (rng.uniform(0, 0.05), rng.uniform(0.14, 0.22)),
        (rng.uniform(0.1, 0.16), rng.uniform(0.17, 0.24)),
    ):
        phase = np.clip((clock - onset) / duration, 0, 1)
        pulse = np.where((phase > 0) & (phase < 1), np.sin(np.pi * phase) ** 2, 0)
        pulses.append(pulse / pulse.max())
    return start, sites, np.stack(pulses, axis=1).astype(np.float32), rng.uniform(-_BOUND, _BOUND, (2, 3))


def _teacher(seed, *, held_out, amplitude=None, refined=False, nominal=False):
    import warp as wp  # noqa: PLC0415 -- initialize the Newton runtime in each worker

    from .teachers import _cable, _record  # noqa: PLC0415 -- initialize the Newton runtime in each worker

    # The stock generator draws segment count first. Keep its RNG independent of the control.
    segments = int(np.random.default_rng(seed).integers(10, 16))
    start, sites, pulses, sampled_amplitude = _setup(seed, segments, held_out=held_out)
    amplitude = sampled_amplitude if amplitude is None else amplitude
    if nominal:
        amplitude = np.zeros((2, 3))
    applied = (pulses[..., None] * amplitude).astype(np.float32)
    nominal_tip = None

    def recorder(model, solver, dt, values, force_array, decode, *, force_field, substeps):
        nonlocal nominal_tip
        if model.body_count != segments:
            raise ValueError("The stock cable builder must register consecutive segment bodies")
        mass = model.body_mass.numpy()
        nominal_tip = (values[start:-1, -1, :3] / mass[-1]).copy()
        values = values.copy()
        for slot, site in enumerate(sites):
            values[start:-1, site - 1, :3] += applied[:, slot] * mass[site - 1]
        if refined:
            solver.iterations = 16
        return _record(
            model, solver, dt, values, force_array, decode, force_field=force_field, substeps=8 if refined else substeps
        )

    begin = perf_counter()
    clock = _DT * np.arange(start + _HORIZON + 1, dtype=np.float64)
    with wp.ScopedDevice("cpu"):
        result = _cable(seed, np.random.default_rng(seed), clock, _DT, "cpu", recorder=recorder)
    return result, (start, sites, pulses, applied, nominal_tip, perf_counter() - begin)


def _sample_arc(values):
    native_arc = np.linspace(0, 1, values.shape[-2])
    left = np.searchsorted(native_arc, _ARC, side="right").clip(1, len(native_arc) - 1) - 1
    fraction = ((_ARC - native_arc[left]) / (native_arc[left + 1] - native_arc[left]))[:, None]
    return (values[..., left, :] * (1 - fraction) + values[..., left + 1, :] * fraction).astype(np.float32)


def _features(episode, setup):
    start, sites, _, _, nominal_tip, _ = setup
    meta = episode["metadata"]
    length = meta["length_m"]
    segments = meta["segment_count"]
    root = episode["position"][start, 0]
    position = _sample_arc(episode["position"][start:] - root) / length
    velocity = _sample_arc(episode["velocity"][start]) / length
    static = np.array(
        [
            length,
            segments / 10,
            meta["radius_m"] * 100,
            meta["density_kg_m3"] / 1000,
            meta["stretch_stiffness_n_m"] / 40000,
            meta["bend_stiffness_nm_rad"] / 15,
            *((np.asarray(sites) - 0.5) / segments),
        ],
        dtype=np.float32,
    )
    state = np.concatenate((position[0].ravel(), velocity.ravel(), static))
    return state, position, nominal_tip, root


def _check_common_observation(nominal, controlled, start):
    for key in ("position", "velocity", "segment_orientation", "segment_angular_velocity"):
        np.testing.assert_array_equal(nominal[key][: start + 1], controlled[key][: start + 1])


def _generate(task):
    seed, refined = task
    nominal, setup = _teacher(seed, held_out=False, nominal=True, refined=refined)
    controlled, control_setup = _teacher(seed, held_out=False, refined=refined)
    _check_common_observation(nominal, controlled, setup[0])
    state, baseline, tip, _ = _features(nominal, setup)
    _, target, _, _ = _features(controlled, control_setup)
    return state, baseline, tip, control_setup[3], target


class CableStateModel(nn.Module):
    """Causal trajectory residual initialized from observed centerline position and velocity."""

    def __init__(self, width=96):
        super().__init__()
        self.width = width
        self.state_encoder = nn.Sequential(nn.Linear(6 * _NODES + 8, width), nn.SiLU(), nn.Linear(width, width))
        self.control_encoder = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width))
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3 * (_NODES - 1)))
        self.register_buffer("time", torch.tensor(_PREDICTION_TIME, dtype=torch.float32)[None, :, None])

    def forward(self, state, tip, applied):
        batch = len(state)
        position, velocity = state[:, : 6 * _NODES].reshape(batch, 2, _NODES, 3).unbind(1)
        initial_hidden = self.state_encoder(state).unsqueeze(0)
        force = torch.cat((tip / 10, applied.flatten(2) / 10, self.time.expand(batch, -1, -1)), dim=-1)
        latent, _ = self.recurrent(self.control_encoder(force), initial_hidden)
        residual = self.decoder(latent).reshape(batch, _HORIZON, _NODES - 1, 3)
        timestep = self.time[..., None]
        drift = velocity[:, None, 1:] * timestep / (1 + 8 * timestep)
        dynamic = position[:, None, 1:] + drift + timestep * residual
        pinned = position[:, None, :1].expand(-1, _HORIZON, -1, -1)
        return torch.cat((pinned, dynamic), dim=2)


class CableImpulseModel(CableStateModel):
    """Predict short-horizon force response with a causal linear impulse readout."""

    def __init__(self, width=96):
        super().__init__(width)
        self.force_response = nn.Sequential(
            nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, 4 * 3 * 3 * (_NODES - 1))
        )
        nn.init.zeros_(self.force_response[-1].weight)
        nn.init.zeros_(self.force_response[-1].bias)
        clock = torch.tensor(_PREDICTION_TIME, dtype=torch.float32)
        lag = clock[:, None] - clock[None] + _DT
        positive = (clock[:, None] >= clock[None]).to(lag.dtype)
        kernels = torch.stack((lag, *(torch.sin(omega * lag) / omega for omega in (12, 22, 35))))
        self.register_buffer("causal_kernels", _DT * positive * kernels)

    def _response_context(self, state, tip):
        batch = len(state)
        position, velocity = state[:, : 6 * _NODES].reshape(batch, 2, _NODES, 3).unbind(1)
        hidden = self.state_encoder(state).unsqueeze(0)
        zeros = tip.new_zeros((batch, _HORIZON, 6))
        nominal = torch.cat((tip / 10, zeros, self.time.expand(batch, -1, -1)), -1)
        latent, _ = self.recurrent(self.control_encoder(nominal), hidden)
        residual = self.decoder(latent).reshape(batch, _HORIZON, _NODES - 1, 3)
        timestep = self.time[..., None]
        drift = velocity[:, None, 1:] * timestep / (1 + 8 * timestep)
        dynamic = position[:, None, 1:] + drift + timestep * residual

        site = state[:, -2:][:, None, :, None].expand(-1, _HORIZON, -1, -1)
        context = torch.cat((latent[:, :, None, :].expand(-1, -1, 2, -1), site), dim=-1)
        gain = self.force_response(context).reshape(batch, _HORIZON, 2, 4, 3, -1)
        pinned = position[:, None, :1].expand(-1, _HORIZON, -1, -1)
        return pinned, dynamic, gain

    def forward(self, state, tip, applied):
        pinned, dynamic, gain = self._response_context(state, tip)
        filtered_force = torch.einsum("ktu,busf->btskf", self.causal_kernels, applied)
        response = (gain * filtered_force[..., None]).sum(dim=(2, 3, 4)).reshape_as(dynamic)
        return torch.cat((pinned, dynamic + response), dim=2)

    def pulse_jacobian(self, state, tip, pulse):
        """Encode the observed state once and return the exact linear pulse response."""
        pinned, dynamic, gain = self._response_context(state, tip)
        filtered_pulse = torch.einsum("ktu,bus->btsk", self.causal_kernels, pulse)
        response = torch.einsum("btskfd,btsk->btdsf", gain, filtered_pulse)
        mobile = response.reshape(*dynamic.shape, 6)
        fixed = mobile.new_zeros((*pinned.shape, 6))
        return torch.cat((fixed, mobile), dim=2)


def _dataset(path, count, workers, *, refined):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if len(data["state"]) != count or ("refined" in data and bool(data["refined"]) != refined):
                raise ValueError("The cached dataset has a different collection recipe")
            if refined and "refined" not in data:
                raise ValueError("The legacy dataset uses the stock teacher")
            return [data[key] for key in ("state", "baseline", "tip", "applied", "target")]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        data = []
        tasks = ((seed, refined) for seed in range(_SEED, _SEED + count))
        for pair in pool.map(_generate, tasks, chunksize=4):
            data.append(pair)
            if len(data) % 100 == 0:
                print(f"Generated {len(data)} paired Newton scenes", flush=True)
    arrays = [np.stack(column) for column in zip(*data, strict=True)]
    np.savez_compressed(
        path, refined=refined, **dict(zip(("state", "baseline", "tip", "applied", "target"), arrays, strict=True))
    )
    return arrays


def _train(arrays, args):
    model_type = CableImpulseModel if args.model == "impulse" else CableStateModel
    model = model_type().to(args.device)
    data = [torch.as_tensor(value, device=args.device) for value in arrays]
    state, baseline, tip, applied, target = data
    response = target[:, 1:] - baseline[:, 1:]
    response_scale = response.square().mean(dim=(1, 2, 3)).clamp_min(2e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=0.00012)
    rng = np.random.default_rng(104)
    history = []
    for iteration in range(1, args.steps + 1):
        index = rng.integers(len(state), size=args.batch)
        force = applied[index]
        features = state[index].repeat(2, 1)
        nominal_tip = tip[index].repeat(2, 1, 1)
        nominal_prediction, control_prediction = model(
            features, nominal_tip, torch.cat((torch.zeros_like(force), force))
        ).chunk(2)
        delta = control_prediction - nominal_prediction
        base_mse = (nominal_prediction - baseline[index, 1:]).square().mean()
        delta_mse = (delta - response[index]).square().mean(dim=(1, 2, 3))
        loss = base_mse / 0.006 + 0.30 * (delta_mse / response_scale[index]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if iteration == 1 or iteration % 200 == 0 or iteration == args.steps:
            row = {
                "step": iteration,
                "train_base_component_rmse_m_per_m": float(base_mse.detach().sqrt()),
                "train_response_component_rmse_m_per_m": float(delta_mse.detach().mean().sqrt()),
                "train_loss": float(loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return model.eval().cpu().requires_grad_(False), history


def _plan(model, seed, features, setup, target, budget):
    state, baseline, tip, _ = features
    _, _, pulses, _, _, _ = setup
    state, nominal_tip, pulse = [torch.as_tensor(value, dtype=torch.float64) for value in (state, tip, pulses)]
    target = torch.as_tensor(target, dtype=torch.float64)
    precision = float(target.square().sum())

    def predict(amplitudes):
        amplitude = amplitudes.reshape(-1, 2, 3)
        return model(
            state.expand(len(amplitude), -1),
            nominal_tip.expand(len(amplitude), -1, -1),
            pulse[None, :, :, None] * amplitude[:, None],
        )

    def objective(amplitudes):
        return (predict(amplitudes) - reference - target).square().sum(dim=(1, 2, 3)) / precision

    def gradient_objective(amplitude):
        value = torch.tensor(amplitude, requires_grad=True)
        loss = objective(value)[0]
        gradient = torch.autograd.grad(loss, value)[0]
        return loss.item(), gradient.numpy()

    initial = np.zeros(6)
    reference = predict(torch.from_numpy(initial))[0]
    _, gradient = gradient_objective(initial)
    epsilon = 1e-4
    directions = epsilon * np.eye(6)
    f_plus = objective(torch.from_numpy(initial + directions)).numpy()
    f_minus = objective(torch.from_numpy(initial - directions)).numpy()
    finite = (f_plus - f_minus) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite, rtol=5e-5, atol=2e-7)
    best = None
    calls = 0

    def bounded_objective(amplitude):
        nonlocal calls, best
        if calls >= budget:
            raise StopIteration
        calls += 1
        result = gradient_objective(amplitude)
        if best is None or result[0] < best[0]:
            best = (result[0], amplitude.copy())
        return result

    start = perf_counter()
    try:
        minimize(
            bounded_objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-_BOUND, _BOUND)] * 6,
            options={"maxfun": budget, "maxiter": budget, "gtol": 1e-9, "ftol": 1e-10},
        )
    except StopIteration:
        pass
    gradient_time = perf_counter() - start
    gradient_cost, optim = best

    generator = qmc.Sobol(6, scramble=True, seed=seed)
    max_candidates = 4096
    candidate_bank = 2 * _BOUND * generator.random_base2(12) - _BOUND
    candidate_bank[0] = initial
    start = perf_counter()
    sample_values = objective(torch.from_numpy(candidate_bank[:calls])).numpy()
    sample_time = perf_counter() - start
    sample_index = int(sample_values.argmin())
    sample = candidate_bank[sample_index]

    sample_wall_time = 0.0
    wall_index = 0
    wall_cost = 1.0
    count = 1
    if sample_time < gradient_time:
        sample_wall_time = sample_time
        wall_index = sample_index
        wall_cost = float(sample_values[sample_index])
        count = calls
    accepted_time = sample_wall_time
    seconds_per_candidate = sample_time / calls
    start = perf_counter() - sample_wall_time
    # Adapt batch width to the remaining deadline. A batch finishing late contributes no candidates.
    while count < max_candidates:
        available = gradient_time - (perf_counter() - start)
        width = min(128, max_candidates - count, int(0.8 * available / seconds_per_candidate))
        if width < 1:
            break
        begin = perf_counter()
        costs = objective(torch.from_numpy(candidate_bank[count : count + width])).numpy()
        end = perf_counter()
        sample_wall_time = end - start
        if sample_wall_time > gradient_time:
            break
        best_batch = int(costs.argmin())
        if costs[best_batch] < wall_cost:
            wall_cost = float(costs[best_batch])
            wall_index = count + best_batch
        count += width
        accepted_time = sample_wall_time
        seconds_per_candidate = 0.6 * seconds_per_candidate + 0.4 * (end - begin) / width

    case = {
        "seed": seed,
        "first_observed_frame": setup[0],
        "force_segment_indices_1based": setup[1],
        "moving_rms_speed_m_s": float(
            np.sqrt(np.mean(np.sum(state[3 * _NODES : 6 * _NODES].numpy().reshape(-1, 3) ** 2, axis=-1))) * state[-8]
        ),
        "absolute_nominal_rms_error_m": float(
            (reference - torch.from_numpy(baseline[1:])).square().sum(-1).mean().sqrt() * state[-8]
        ),
        "model_gradient": gradient.tolist(),
        "max_autograd_fd_abs_difference": float(np.max(np.abs(gradient - finite))),
    }
    case["initial"] = {
        "amplitudes": initial.tolist(),
        "model_evaluations": 0,
        "model_ms": 0.0,
        "model_response_error_pct": 100.0,
    }
    case["gradient"] = {
        "amplitudes": optim.tolist(),
        "model_evaluations": calls,
        "model_ms": 1000 * gradient_time,
        "model_response_error_pct": 100 * np.sqrt(gradient_cost),
    }
    case["sampling"] = {
        "amplitudes": sample.tolist(),
        "model_evaluations": calls,
        "model_ms": 1000 * sample_time,
        "model_response_error_pct": 100 * np.sqrt(sample_values[sample_index]),
    }
    case["wall_matched_sampling"] = {
        "amplitudes": candidate_bank[wall_index].tolist(),
        "model_evaluations": count,
        "model_ms": 1000 * sample_wall_time,
        "accepted_model_ms": 1000 * accepted_time,
        "deadline_overrun_ms": 1000 * max(0.0, sample_wall_time - gradient_time),
        "model_response_error_pct": 100 * np.sqrt(wall_cost),
    }
    if isinstance(model, CableImpulseModel):
        start = perf_counter()
        jacobian = model.pulse_jacobian(state[None], nominal_tip[None], pulse[None])
        matrix = jacobian.reshape(-1, 6).numpy()
        target_flat = target.flatten().numpy()
        solution = lsq_linear(matrix, target_flat, bounds=(-_BOUND, _BOUND))
        direct_time = perf_counter() - start
        direct_gradient = -2 * matrix.T @ target_flat / precision
        np.testing.assert_allclose(direct_gradient, gradient, rtol=1e-8, atol=1e-10)
        direct_cost = objective(torch.from_numpy(solution.x))[0].item()
        np.testing.assert_allclose(direct_cost, 2 * solution.cost / precision, rtol=1e-8, atol=1e-10)
        if not solution.success or direct_cost > gradient_cost + 1e-8:
            raise RuntimeError("The exact linear response solve did not recover the bounded model optimum")
        case["exact_response_solve"] = {
            "amplitudes": solution.x.tolist(),
            "model_evaluations": 1,
            "model_ms": 1000 * direct_time,
            "model_response_error_pct": 100 * np.sqrt(direct_cost),
        }
    return case, reference.numpy(), predict(torch.from_numpy(optim))[0].numpy()


def _replay_case(task):
    case, refine, run_fd = task
    seed = case["seed"]
    nominal, setup = _teacher(seed, held_out=True, nominal=True, refined=refine)
    target, target_setup = _teacher(seed, held_out=True, refined=refine)
    start = setup[0]
    _check_common_observation(nominal, target, start)
    nominal_position = nominal["position"][start:]
    target_position = target["position"][start:]
    target_delta = target_position[1:].astype(np.float64) - nominal_position[1:]
    denominator = np.square(target_delta).sum()
    replay_states = {}
    replay_runtimes = []
    methods = ["initial", "gradient", "sampling", "wall_matched_sampling"]
    if "exact_response_solve" in case:
        methods.append("exact_response_solve")
    for name in methods:
        entry = case[name]
        amplitude = np.asarray(entry["amplitudes"]).reshape(2, 3)
        result, settings = _teacher(seed, held_out=True, amplitude=amplitude, refined=refine)
        _check_common_observation(nominal, result, start)
        positions = result["position"][start:]
        error = positions[1:].astype(np.float64) - target_position[1:]
        entry["newton_response_error_pct"] = float(100 * np.sqrt(np.square(error).sum() / denominator))
        entry["newton_trajectory_rms_error_m"] = float(np.sqrt(np.square(error).sum(-1).mean()))
        replay_states[name] = positions
        replay_runtimes.append(settings[-1])
    if run_fd:
        epsilon = 0.12
        f_differences = []
        for direction in np.eye(6):
            loss_values = []
            for sign in (1, -1):
                result, _ = _teacher(
                    seed, held_out=True, amplitude=(sign * epsilon * direction).reshape(2, 3), refined=refine
                )
                difference = result["position"][start + 1 :].astype(np.float64) - target_position[1:]
                loss_values.append(np.square(difference).sum() / denominator)
            f_differences.append((loss_values[0] - loss_values[1]) / (2 * epsilon))
        expected = np.asarray(f_differences)
        actual = np.asarray(case["model_gradient"])
        case["newton_finite_difference_gradient"] = expected.tolist()
        case["newton_gradient_cosine"] = float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    case["target_control_amplitudes"] = _setup(seed, nominal["metadata"]["segment_count"], held_out=True)[3].tolist()
    case["mean_newton_replay_ms"] = 1000 * float(np.mean(replay_runtimes))
    case["target_intervention_rms_m"] = float(np.sqrt(np.square(target_delta).sum(-1).mean()))
    case["teacher_metadata"] = {
        **nominal["metadata"],
        "substeps": 8 if refine else 4,
        "iterations": 16 if refine else 8,
    }
    return case, {"nominal": nominal_position, "target": target_position, **replay_states}, target_setup


def _aggregate(cases):
    report = {"cases": len(cases)}
    initial = np.array([case["initial"]["newton_response_error_pct"] for case in cases])
    methods = ["initial", "gradient", "sampling", "wall_matched_sampling"]
    if "exact_response_solve" in cases[0]:
        methods.append("exact_response_solve")
    for name in methods:
        rows = [case[name] for case in cases]
        response = np.array([row["newton_response_error_pct"] for row in rows])
        report[name] = {
            "mean_newton_response_error_pct": float(response.mean()),
            "median_newton_response_error_pct": float(np.median(response)),
            "improved_cases": int((response < initial - 1e-6).sum()),
            "median_model_evaluations": float(np.median([row["model_evaluations"] for row in rows])),
            "median_model_ms": float(np.median([row["model_ms"] for row in rows])),
            "mean_newton_trajectory_rms_error_m": float(
                np.mean([row["newton_trajectory_rms_error_m"] for row in rows])
            ),
        }
    report["gradient_better_than_forward_matched_sampling"] = sum(
        case["gradient"]["newton_response_error_pct"] < case["sampling"]["newton_response_error_pct"] for case in cases
    )
    report["gradient_better_than_wall_matched_sampling"] = sum(
        case["gradient"]["newton_response_error_pct"] < case["wall_matched_sampling"]["newton_response_error_pct"]
        for case in cases
    )
    return report


def main():
    """Generate, fit and score a fixed-horizon differentiable cable controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-cases", type=int, default=800)
    parser.add_argument("--eval-cases", type=int, default=16)
    parser.add_argument("--refined-cases", type=int, default=6)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-checkpoint", action="store_true", help="Reevaluate a previously fitted model")
    parser.add_argument("--model", choices=("recurrent", "impulse"), default="recurrent")
    parser.add_argument("--eval-seed-offset", type=int, default=100_000)
    parser.add_argument("--teacher-refined", action="store_true", help="Train and score at 8 substeps and 16 sweeps")
    args = parser.parse_args()
    if min(args.train_cases, args.eval_cases, args.budget, args.steps, args.batch, args.workers) < 1:
        parser.error("Use positive case counts, budget, steps, batch size, and workers")
    if args.eval_seed_offset < args.train_cases:
        parser.error("Training and evaluation seed ranges must be separate")
    if args.teacher_refined and args.refined_cases:
        parser.error("Use --refined-cases 0 when the main teacher already uses the finer solver")
    if not 0 <= args.refined_cases <= args.eval_cases or args.budget > 4096:
        parser.error("Refined cases must be a subset of evaluation cases and budget must be at most 4096")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise FileExistsError(args.output / "results.json")

    torch.set_num_threads(1)
    torch.manual_seed(661)
    started = perf_counter()
    data_path = args.output / "pairs.npz"
    arrays = _dataset(data_path, args.train_cases, args.workers, refined=args.teacher_refined)
    dataset_time = perf_counter() - started
    print(f"Paired data ready after {dataset_time:.1f} s", flush=True)
    trained = perf_counter()
    checkpoint = args.output / "model.pt"
    if args.reuse_checkpoint:
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_type = CableImpulseModel if bundle.get("model_type") == "impulse" else CableStateModel
        model = model_type(bundle["width"])
        model.load_state_dict(bundle["model"])
        model.eval().requires_grad_(False)
        history = bundle["history"]
        fit_time = bundle.get("fit_seconds")
    else:
        model, history = _train(arrays, args)
        fit_time = perf_counter() - trained
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": args.model,
                "width": model.width,
                "history": history,
                "fit_seconds": fit_time,
            },
            checkpoint,
        )
    model.double()
    eval_seeds = range(_SEED + args.eval_seed_offset, _SEED + args.eval_seed_offset + args.eval_cases)
    plans = []
    traces = []
    print("Model ready, planning on observed Newton states", flush=True)
    for seed in eval_seeds:
        nominal, setup = _teacher(seed, held_out=True, nominal=True, refined=args.teacher_refined)
        target, target_setup = _teacher(seed, held_out=True, refined=args.teacher_refined)
        _check_common_observation(nominal, target, setup[0])
        features = _features(nominal, setup)
        target_features = _features(target, target_setup)
        response = target_features[1][1:] - features[1][1:]
        case, model_nominal, model_optimized = _plan(model, seed, features, setup, response, args.budget)
        state, baseline, tip, root = features
        held_position = baseline[0]
        held_velocity = state[3 * _NODES : 6 * _NODES].reshape(_NODES, 3)
        standard_drift = held_position + _PREDICTION_TIME[:, None, None] * held_velocity / (
            1 + 8 * _PREDICTION_TIME[:, None, None]
        )
        standard_drift[:, 0] = held_position[0]
        length = nominal["metadata"]["length_m"]
        with torch.no_grad():
            model_target = model(
                torch.from_numpy(state[None]).double(),
                torch.from_numpy(tip[None]).double(),
                torch.from_numpy(target_setup[3][None]).double(),
            )[0].numpy()
        label = target_features[1][1:]
        case["hold_nominal_rms_error_m"] = length * float(
            np.sqrt(np.square(held_position - baseline[1:]).sum(-1).mean())
        )
        case["damped_velocity_nominal_rms_error_m"] = length * float(
            np.sqrt(np.square(standard_drift - baseline[1:]).sum(-1).mean())
        )
        case["absolute_target_rms_error_m"] = length * float(np.sqrt(np.square(model_target - label).sum(-1).mean()))
        case["target_control_model_response_error_pct"] = 100 * float(
            np.linalg.norm(model_target - model_nominal - response) / np.linalg.norm(response)
        )
        traces.append(
            {
                "model_nominal": length * np.concatenate((baseline[:1], model_nominal)) + root,
                "model_optimized": length * np.concatenate((baseline[:1], model_optimized)) + root,
                "model_target": length * np.concatenate((baseline[:1], model_target)) + root,
            }
        )
        if "exact_response_solve" in case:
            amplitude = np.asarray(case["exact_response_solve"]["amplitudes"]).reshape(2, 3)
            applied = setup[2][..., None] * amplitude
            with torch.no_grad():
                model_exact = model(
                    torch.from_numpy(state[None]).double(),
                    torch.from_numpy(tip[None]).double(),
                    torch.from_numpy(applied[None]).double(),
                )[0].numpy()
            traces[-1]["model_exact"] = length * np.concatenate((baseline[:1], model_exact)) + root
        plans.append(case)

    print("Replaying optimized and sampled controls in Newton", flush=True)
    coarse = [(case, args.teacher_refined, True) for case in plans]
    refined = [(case, True, False) for case in plans[: args.refined_cases]]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = []
        for index, (case, states, _) in enumerate(pool.map(_replay_case, coarse + refined)):
            if index < len(coarse):
                np.savez_compressed(
                    args.output / f"case-{case['seed']}.npz",
                    time=_DT * np.arange(_HORIZON + 1),
                    **traces[index],
                    **states,
                )
            print(
                json.dumps(
                    {
                        "seed": case["seed"],
                        "refined": index >= len(coarse),
                        "gradient_error_pct": case["gradient"]["newton_response_error_pct"],
                        "sampling_error_pct": case["sampling"]["newton_response_error_pct"],
                    }
                ),
                flush=True,
            )
            results.append(case)

    cases = results[: len(coarse)]
    refined_cases = results[len(coarse) :]
    receipt = {
        "task": "Root-clamped, contact-free Newton cable from one observed moving state; match added motion "
        "over a known nominal trajectory by optimizing two bounded xyz force pulses.",
        "teacher_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "response_metric": "Native Euclidean RMS tracking error divided by nominal-to-target RMS response",
        "prediction_interval_s": _DT,
        "prediction_steps": _HORIZON,
        "prediction_arc_samples": _NODES,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_type": type(model).__name__,
        "force_bounds_m_s2": [-_BOUND, _BOUND],
        "train_seed_range_inclusive": [_SEED, _SEED + args.train_cases - 1],
        "train_cases": args.train_cases,
        "train_recordings": 2 * args.train_cases,
        "primary_solver_substeps": 8 if args.teacher_refined else 4,
        "primary_solver_sweeps": 16 if args.teacher_refined else 8,
        "train_site_protocol": "even native segment indices, test uses odd indices within each topology",
        "dataset_sha256": digest(data_path),
        "model_sha256": digest(checkpoint),
        "torch_version": str(torch.__version__),
        "train_history": history,
        "fit_device": args.device,
        "dataset_seconds": dataset_time,
        "fit_seconds": fit_time,
        "planner_timing": "One CPU thread, warm f64, includes SciPy and candidate scoring. Excludes precomputing "
        "the Sobol candidate bank, teacher replay and gradient checks. Late Sobol batches are excluded "
        "from the accepted candidates and any elapsed deadline overrun is recorded.",
        "results": _aggregate(cases),
        "refined_results": _aggregate(refined_cases) if refined_cases else None,
        "cases": cases,
        "refined_cases": refined_cases,
    }
    payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    with (args.output / "results.json").open("x") as stream:
        stream.write(payload)
    print(json.dumps(receipt["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
