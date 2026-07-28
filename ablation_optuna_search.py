#!/usr/bin/env python3
"""Optuna orchestration for low-code ablation search on run_ablation_v2.

This script treats ``run_ablation_v2.py`` as the execution backend and performs
trial-level search over a compact set of meaningful ablation knobs:

- experiment preset (captures loss family, hard negatives, DMS on/off)
- pooling mode
- learning rate
- warmup steps
- map row cap
- time budget
- optional DMS row cap (only when DMS is enabled)

Each trial runs one experiment preset end-to-end, reads the generated
``delta_vs_baseline.csv`` and ``summary.csv``, and computes a reviewer-friendly
cross-task objective based on per-task percentage deltas with metric priority:
AUC > Accuracy > Spearman > F1.

Example:
python ablation_optuna_search.py --study-name ablation_search_v1 --n-trials 24 --experiments mnrl_cosent_multi gist_cosent_multi cached_mnrl_cosent_multi cached_gist_cosent_multi cached_mnrl_no_dms --pooling-modes mean contextual_attention  --max-map-rows 600000 --max-minutes 20
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast


DEFAULT_EXPERIMENTS = [
    "mnrl_cosent_multi",
    "gist_cosent_multi",
    "cached_mnrl_cosent_multi",
    "cached_gist_cosent_multi",
    "cached_mnrl_no_dms",
]
DEFAULT_POOLING_MODES = ["mean", "contextual_attention"]
DEFAULT_METRIC_PRIORITY = ["AUC", "Accuracy", "Spearman", "F1"]
DEFAULT_MAX_MAP_ROWS = [600_000, 1_500_000]
DEFAULT_MAX_MINUTES = [15, 45]
DEFAULT_DMS_MAX_ROWS = [0, 500_000]


@dataclass(frozen=True)
class TaskDelta:
    """Selected primary delta for a task after metric-priority resolution."""

    task: str
    metric: str
    delta: float
    baseline: float
    pct_delta: float


@dataclass(frozen=True)
class TrialScore:
    """Aggregated score and diagnostics for a completed trial."""

    objective: float
    mean_pct_delta: float
    median_pct_delta: float
    std_pct_delta: float
    summary_mean_pct_delta: float
    wins: int
    losses: int
    ties: int
    n_tasks: int


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated integer list from CLI text."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in values]


def _try_float(raw: str | None) -> float | None:
    """Parse float-or-none from CSV cell text."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def experiment_uses_dms(experiment_id: str) -> bool:
    """Return whether the preset includes DMS CoSENT signal."""
    return experiment_id not in {"baseline_eval_only", "cached_mnrl_no_dms"}


def select_primary_task_delta(
    row: dict[str, str],
    metric_priority: list[str],
) -> TaskDelta | None:
    """Pick one metric delta for a row using priority order.

    The row must contain paired fields: ``delta_<metric>`` and
    ``baseline_<metric>``. If baseline is near-zero, percent delta is scaled by
    100 from raw delta to avoid division by zero.
    """
    task = row.get("Task", "")
    for metric in metric_priority:
        delta = _try_float(row.get(f"delta_{metric}"))
        baseline = _try_float(row.get(f"baseline_{metric}"))
        if delta is None or baseline is None:
            continue
        if abs(baseline) > 1e-12:
            pct = (delta / abs(baseline)) * 100.0
        else:
            pct = delta * 100.0
        return TaskDelta(
            task=task,
            metric=metric,
            delta=delta,
            baseline=baseline,
            pct_delta=pct,
        )
    return None


def _summary_mean_percent_delta(summary_csv: Path, experiment_id: str) -> float:
    """Compute mean % delta of summary means vs baseline for 4 primary metrics."""
    if not summary_csv.exists():
        return 0.0

    baseline_row: dict[str, str] | None = None
    target_row: dict[str, str] | None = None

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("experiment_id") == "baseline_eval_only":
                baseline_row = row
            if row.get("experiment_id") == experiment_id:
                target_row = row

    if baseline_row is None or target_row is None:
        return 0.0

    metrics = ["spearman", "accuracy", "f1", "auc"]
    pct_deltas: list[float] = []
    for metric in metrics:
        base_value = _try_float(baseline_row.get(f"knn_{metric}_mean"))
        target_value = _try_float(target_row.get(f"knn_{metric}_mean"))
        if base_value is None or target_value is None:
            continue
        if abs(base_value) > 1e-12:
            pct_deltas.append(((target_value - base_value) / abs(base_value)) * 100.0)
    if not pct_deltas:
        return 0.0
    return statistics.fmean(pct_deltas)


def compute_trial_score(
    *,
    delta_csv: Path,
    summary_csv: Path,
    experiment_id: str,
    metric_priority: list[str],
    tie_epsilon: float,
    noise_penalty: float,
) -> TrialScore:
    """Compute objective and diagnostics from run artifacts."""
    if not delta_csv.exists():
        raise FileNotFoundError(f"Missing delta CSV: {delta_csv}")

    selected: list[TaskDelta] = []
    with delta_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("stage") != experiment_id:
                continue
            chosen = select_primary_task_delta(row, metric_priority)
            if chosen is not None:
                selected.append(chosen)

    if not selected:
        raise ValueError(f"No comparable task deltas found for stage={experiment_id}")

    pct_values = [item.pct_delta for item in selected]
    mean_pct = statistics.fmean(pct_values)
    median_pct = statistics.median(pct_values)
    std_pct = statistics.pstdev(pct_values) if len(pct_values) > 1 else 0.0

    wins = sum(1 for item in selected if item.delta > tie_epsilon)
    losses = sum(1 for item in selected if item.delta < -tie_epsilon)
    ties = len(selected) - wins - losses

    summary_mean_pct = _summary_mean_percent_delta(summary_csv, experiment_id)
    objective = mean_pct - (noise_penalty * std_pct)

    return TrialScore(
        objective=objective,
        mean_pct_delta=mean_pct,
        median_pct_delta=median_pct,
        std_pct_delta=std_pct,
        summary_mean_pct_delta=summary_mean_pct,
        wins=wins,
        losses=losses,
        ties=ties,
        n_tasks=len(selected),
    )


def _trial_run_prefix(base_prefix: str, session_id: str, trial_number: int) -> str:
    """Build deterministic, collision-safe run prefix for one trial."""
    return f"{base_prefix}_{session_id}_t{trial_number:04d}"


def _run_ablation_trial(
    *,
    repo_root: Path,
    python_bin: str,
    experiment_id: str,
    calibrate_smoke: bool,
    smoke_max_steps: int,
    env_overrides: dict[str, str],
) -> None:
    """Execute one ablation trial through run_ablation_v2.py."""
    cmd = [
        python_bin,
        "run_ablation_v2.py",
        "--experiments",
        experiment_id,
        "--reset-summary",
        "--smoke-max-steps",
        str(smoke_max_steps),
    ]
    if calibrate_smoke:
        cmd.append("--calibrate-batch-smoke")

    trial_env = os.environ.copy()
    trial_env.update(env_overrides)

    subprocess.run(
        cmd,
        cwd=repo_root,
        env=trial_env,
        check=True,
    )


def _write_top_trials_csv(
    *,
    study: Any,
    out_csv: Path,
    top_k: int,
) -> None:
    """Write completed trial leaderboard with params and diagnostics."""
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    complete_trials.sort(key=lambda trial: trial.value, reverse=True)

    fieldnames = [
        "rank",
        "trial_number",
        "objective",
        "mean_pct_delta",
        "summary_mean_pct_delta",
        "median_pct_delta",
        "std_pct_delta",
        "wins",
        "losses",
        "ties",
        "n_tasks",
        "run_prefix",
        "params_json",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, trial in enumerate(complete_trials[: max(1, top_k)], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "trial_number": trial.number,
                    "objective": f"{trial.value:.6f}",
                    "mean_pct_delta": trial.user_attrs.get("mean_pct_delta", ""),
                    "summary_mean_pct_delta": trial.user_attrs.get(
                        "summary_mean_pct_delta", ""
                    ),
                    "median_pct_delta": trial.user_attrs.get("median_pct_delta", ""),
                    "std_pct_delta": trial.user_attrs.get("std_pct_delta", ""),
                    "wins": trial.user_attrs.get("wins", ""),
                    "losses": trial.user_attrs.get("losses", ""),
                    "ties": trial.user_attrs.get("ties", ""),
                    "n_tasks": trial.user_attrs.get("n_tasks", ""),
                    "run_prefix": trial.user_attrs.get("run_prefix", ""),
                    "params_json": str(trial.params),
                }
            )


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for Optuna ablation search."""
    parser = argparse.ArgumentParser(
        description="Run Optuna search on run_ablation_v2 experiment presets"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--sampler-seed", type=int, default=41)
    parser.add_argument("--study-name", type=str, default="ablation_optuna")
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("results") / "optuna_ablation",
        help="Directory for Optuna sqlite DB and reports",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optional Optuna storage URL. Empty uses sqlite under --study-dir.",
    )
    parser.add_argument(
        "--base-run-prefix",
        type=str,
        default="ablation_optuna",
        help="Prefix for trial RUN_PREFIX values",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to launch run_ablation_v2.py",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment presets to sample from",
    )
    parser.add_argument(
        "--pooling-modes",
        nargs="+",
        default=DEFAULT_POOLING_MODES,
        help="Pooling modes to sample",
    )
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--lr-max", type=float, default=2e-4)
    parser.add_argument("--warmup-min", type=int, default=20)
    parser.add_argument("--warmup-max", type=int, default=120)
    parser.add_argument(
        "--max-map-rows",
        type=_parse_int_list,
        default=DEFAULT_MAX_MAP_ROWS,
        help="Comma-separated choices, e.g. 800000,1500000,2500000",
    )
    parser.add_argument(
        "--max-minutes",
        type=_parse_int_list,
        default=DEFAULT_MAX_MINUTES,
        help="Comma-separated choices, e.g. 20,30,45",
    )
    parser.add_argument(
        "--dms-max-rows",
        type=_parse_int_list,
        default=DEFAULT_DMS_MAX_ROWS,
        help="Comma-separated choices for DMS-enabled presets",
    )

    parser.add_argument(
        "--metric-priority",
        nargs="+",
        default=DEFAULT_METRIC_PRIORITY,
        help="Priority order for selecting per-task metric deltas",
    )
    parser.add_argument("--tie-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--noise-penalty",
        type=float,
        default=0.10,
        help="Objective = mean_pct_delta - noise_penalty * std_pct_delta",
    )

    parser.add_argument(
        "--calibrate-batch-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable run_ablation_v2 smoke calibration per trial",
    )
    parser.add_argument("--smoke-max-steps", type=int, default=2)

    parser.add_argument("--train-cuda-device", type=str, default="")
    parser.add_argument("--bench-cuda-device", type=str, default="")

    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-tags", nargs="*", default=[])

    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Entry point for Optuna ablation search."""
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent

    if args.warmup_min > args.warmup_max:
        raise ValueError("--warmup-min must be <= --warmup-max")
    if args.lr_min <= 0 or args.lr_max <= 0 or args.lr_min >= args.lr_max:
        raise ValueError("Invalid learning rate range.")

    args.study_dir.mkdir(parents=True, exist_ok=True)
    storage_url = (
        args.storage or f"sqlite:///{(args.study_dir / 'optuna.db').resolve()}"
    )

    optuna_mod: Any = importlib.import_module("optuna")

    sampler = optuna_mod.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna_mod.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna_mod.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def objective(trial: Any) -> float:
        experiment_id = trial.suggest_categorical("experiment_id", args.experiments)
        pooling_mode = trial.suggest_categorical("pooling_mode", args.pooling_modes)
        learning_rate = trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        )
        warmup_steps = trial.suggest_int(
            "warmup_steps", args.warmup_min, args.warmup_max
        )
        max_map_rows = trial.suggest_categorical("max_map_rows", args.max_map_rows)
        max_minutes = trial.suggest_categorical("max_minutes", args.max_minutes)

        if experiment_uses_dms(experiment_id):
            dms_max_rows = trial.suggest_categorical("dms_max_rows", args.dms_max_rows)
        else:
            dms_max_rows = 0

        run_prefix = _trial_run_prefix(args.base_run_prefix, session_id, trial.number)

        env_overrides = {
            "RUN_PREFIX": run_prefix,
            "POOLING_MODE": pooling_mode,
            "LEARNING_RATE": str(learning_rate),
            "WARMUP_STEPS": str(warmup_steps),
            "MAX_MAP_ROWS": str(max_map_rows),
            "MAX_MINUTES": str(max_minutes),
            "DMS_MAX_ROWS": str(dms_max_rows),
            "FORCE_BASELINE_BENCHMARK_RERUN": "1",
            "SKIP_EXISTING_BENCHMARKS": "1",
        }
        if args.train_cuda_device:
            env_overrides["TRAIN_CUDA_DEVICE"] = args.train_cuda_device
        if args.bench_cuda_device:
            env_overrides["BENCH_CUDA_DEVICE"] = args.bench_cuda_device

        wandb_run: Any = None
        wandb_mod: Any = None
        if args.wandb_project:
            wandb_mod = cast(Any, importlib.import_module("wandb"))

            wandb_run = wandb_mod.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                tags=args.wandb_tags,
                reinit=True,
                name=f"{args.study_name}-trial-{trial.number}",
                config={
                    "study_name": args.study_name,
                    "run_prefix": run_prefix,
                    "experiment_id": experiment_id,
                    "pooling_mode": pooling_mode,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "max_map_rows": max_map_rows,
                    "max_minutes": max_minutes,
                    "dms_max_rows": dms_max_rows,
                    "noise_penalty": args.noise_penalty,
                    "metric_priority": args.metric_priority,
                },
            )

        try:
            _run_ablation_trial(
                repo_root=repo_root,
                python_bin=args.python_bin,
                experiment_id=experiment_id,
                calibrate_smoke=args.calibrate_batch_smoke,
                smoke_max_steps=max(1, args.smoke_max_steps),
                env_overrides=env_overrides,
            )

            trial_result_dir = repo_root / "results" / run_prefix
            score = compute_trial_score(
                delta_csv=trial_result_dir / "delta_vs_baseline.csv",
                summary_csv=trial_result_dir / "summary.csv",
                experiment_id=experiment_id,
                metric_priority=args.metric_priority,
                tie_epsilon=args.tie_epsilon,
                noise_penalty=args.noise_penalty,
            )

            trial.set_user_attr("run_prefix", run_prefix)
            trial.set_user_attr("mean_pct_delta", round(score.mean_pct_delta, 6))
            trial.set_user_attr(
                "summary_mean_pct_delta", round(score.summary_mean_pct_delta, 6)
            )
            trial.set_user_attr("median_pct_delta", round(score.median_pct_delta, 6))
            trial.set_user_attr("std_pct_delta", round(score.std_pct_delta, 6))
            trial.set_user_attr("wins", score.wins)
            trial.set_user_attr("losses", score.losses)
            trial.set_user_attr("ties", score.ties)
            trial.set_user_attr("n_tasks", score.n_tasks)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "objective": score.objective,
                        "mean_pct_delta": score.mean_pct_delta,
                        "summary_mean_pct_delta": score.summary_mean_pct_delta,
                        "median_pct_delta": score.median_pct_delta,
                        "std_pct_delta": score.std_pct_delta,
                        "wins": score.wins,
                        "losses": score.losses,
                        "ties": score.ties,
                        "n_tasks": score.n_tasks,
                    }
                )

            return score.objective
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
            trial.set_user_attr("run_prefix", run_prefix)
            trial.set_user_attr("failure", str(exc))
            raise optuna_mod.exceptions.TrialPruned(f"Trial failed: {exc}") from exc
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    study.optimize(
        objective,
        n_trials=max(1, args.n_trials),
        timeout=args.timeout if args.timeout > 0 else None,
        n_jobs=max(1, args.n_jobs),
    )

    top_csv = args.study_dir / f"{args.study_name}_top_trials.csv"
    _write_top_trials_csv(study=study, out_csv=top_csv, top_k=args.top_k)

    best = study.best_trial
    print("Best trial summary:")
    print(f"  number={best.number}")
    print(f"  objective={best.value:.6f}")
    print(f"  params={best.params}")
    print(f"  attrs={best.user_attrs}")
    print(f"Top-trial table: {top_csv}")


if __name__ == "__main__":
    main()
