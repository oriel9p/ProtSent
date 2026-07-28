#!/usr/bin/env python3
"""Maintainable stage-wise ablation runner.

This runner preserves the important behavior from the previous shell script while
moving orchestration logic into Python:
- baseline KNN benchmark (forced rerun by default)
- stage1 triplet training
- stage2a cached_mnrl training (pfam hard negatives + afdb + stringdb)
- stage2b multi training (+ dms cosent)
- per-stage KNN benchmarks
- per-task deltas versus baseline evluation results (fast benchmarks be default)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import argparse
import csv
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import logging

import polars as pl

from benchmark_tasks import TASKS

logger = logging.getLogger(__name__)

OOM_LOG_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cudnn_status_alloc_failed",
    "failed to allocate memory",
)
MAX_LOG_TAIL_BYTES = 1_000_000
REPORT_METRICS = (
    "Accuracy",
    "F1",
    "AUC",
    "AP",
    "F1_Weighted",
    "F1_Macro",
    "F1_Micro",
    "Spearman",
    "MSE",
)
PROBE_SUPPORTED_PROBLEM_TYPES = {"binary", "multiclass", "regression"}
PROTEINGYM_FAST_PROBE_TASKS = (
    "proteingym_clinical_substitutions_supervised",
    "proteingym_clinical_indels_supervised",
)
PIPELINE_STAGES = ("baseline", "stage1", "stage2a", "stage2b", "report")
CLI_STAGE_CHOICES = (*PIPELINE_STAGES, "all")
EXPERIMENT_IDS = (
    "baseline_eval_only",
    "triplet_cosent_multi",
    "mnrl_cosent_multi",
    "cached_mnrl_cosent_multi",
    "cached_gist_cosent_multi",
    "gist_cosent_multi",
    "cached_mnrl_no_dms",
    "staged_cached_mnrl",
)
CLI_EXPERIMENT_CHOICES = (*EXPERIMENT_IDS, "all")
# Fixed fallback batches used when smoke calibration is disabled.
# Non-cached approaches start at 64 (OOM retry may step down to 32).
# Cached approaches start at 512 with the pipeline's default mini-batch size.
DEFAULT_NON_CACHED_BATCH = 64
DEFAULT_CACHED_BATCH = 512
PIPELINE_DEFAULT_MNRL_MINI_BATCH = 256
# Calibration ladders are sized for single-GPU (ESM2-35M, ~80 GiB VRAM).
# cached_mnrl/cached_gist/staged start low because they carry 4-dataset batches.
CALIBRATION_LADDERS: dict[str, tuple[int, ...]] = {
    "triplet_cosent_multi": (32, 64, 80),
    "mnrl_cosent_multi": (256, 320, 512),
    "cached_mnrl_cosent_multi": (64, 96, 128),
    "cached_gist_cosent_multi": (64, 96, 128),
    "gist_cosent_multi": (128, 256),
    "cached_mnrl_no_dms": (64, 96, 128),
    "staged_cached_mnrl": (64, 96, 128),
}

SUMMARY_COLUMNS = [
    "run_prefix",
    "experiment_id",
    "stage",
    "model_path",
    "loss_mode",
    "multi_primary_loss",
    "train_files",
    "dms_enabled",
    "batch_size",
    "grad_accum",
    "mnrl_mini_batch",
    "max_minutes",
    "knn_spearman_mean",
    "knn_accuracy_mean",
    "knn_f1_mean",
    "knn_auc_mean",
    "knn_csv",
    "Desc",
]

MANIFEST_COLUMNS = [
    "run_prefix",
    "experiment_id",
    "stage",
    "status",
    "model_path",
    "run_name",
    "loss_mode",
    "multi_primary_loss",
    "train_files",
    "batch_size",
    "grad_accum",
    "mnrl_mini_batch",
    "max_minutes",
    "knn_csv",
    "retries",
    "params_json",
    "Desc",
]

CALIBRATION_COLUMNS = [
    "experiment_id",
    "tested_batch",
    "success",
    "retries",
    "chosen_batch",
    "Desc",
]


@dataclass(frozen=True)
class ExperimentPreset:
    """Configuration preset for one independent single-stage experiment.

    When ``chain_from_experiment`` is set the runner first ensures the named
    experiment's model exists (training it for ``max_minutes // 2`` if needed),
    then uses that checkpoint as the init model for this experiment, also
    capped at ``max_minutes // 2``.  This implements two-stage warm-start
    experiments within the single-GPU overnight budget.
    """

    experiment_id: str
    run_name_suffix: str
    stage_name: str
    loss_mode: str
    multi_primary_loss: str | None
    files: tuple[Path, ...]
    hard_negatives: bool
    dms_enabled: bool
    chain_from_experiment: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """Captured metadata for one executed train+benchmark experiment."""

    experiment_id: str
    stage: str
    run_name: str
    model_path: Path
    knn_csv: Path
    loss_mode: str
    multi_primary_loss: str
    files: tuple[Path, ...]
    batch_size: int
    grad_accum: int
    mnrl_mini_batch: int
    max_minutes: int
    retries: int
    status: str


def _full_probe_task_keys() -> list[str]:
    """Return the supported task keys for integrated KNN-style probe runs."""
    task_keys = [
        key
        for key, cfg in TASKS.items()
        if cfg.problem_type in PROBE_SUPPORTED_PROBLEM_TYPES
        and cfg.eval_mode == "standard"
    ]
    task_keys.extend(
        task_key
        for task_key in PROTEINGYM_FAST_PROBE_TASKS
        if task_key not in task_keys
    )
    return task_keys


def _parse_bool(raw: str) -> bool:
    """Parse a truthy/falsy string value from an environment variable."""
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_values(
    specs: dict[str, tuple[Callable[[str], object], object]],
) -> dict[str, object]:
    """Load and parse environment values from a declarative spec table.

    Args:
        specs: Mapping of env var name to `(parser, default)`.

    Returns:
        Parsed values keyed by env var name.

    Raises:
        ValueError: If parsing fails for any configured variable.
    """
    values: dict[str, object] = {}
    for name, (parser, default) in specs.items():
        raw = os.environ.get(name)
        if raw is None:
            values[name] = default
            continue
        try:
            values[name] = parser(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {name}: {raw!r}") from exc
    return values


def _as_int(value: object) -> int:
    """Convert an env value object into an integer."""
    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    """Convert an env value object into a float."""
    return float(cast(float | int | str, value))


@dataclass(frozen=True)
class DataPaths:
    """Filesystem paths for train/eval datasets used by the ablation runner."""

    pfam_file: Path
    hard_neg_parquet: Path
    afdb_file: Path
    stringdb_file: Path
    dms_cosent_file: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution/runtime settings for process launch and progress behavior.

    ``train_cuda_device`` and ``bench_cuda_device`` are passed as
    ``CUDA_VISIBLE_DEVICES`` to the training and benchmark subprocesses
    respectively.  Leave empty to inherit the parent environment.
    """

    python_bin: str
    train_distributed: bool
    train_num_processes: int
    train_mixed_precision: str
    progress_min_interval: float
    train_cuda_device: str
    bench_cuda_device: str


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters shared by Stage 1/2 jobs."""

    pooling_mode: str
    optimizer: str
    compile_training: bool
    save_steps: int
    max_minutes: int
    max_map_rows: int
    dms_max_rows: int
    learning_rate: float
    warmup_steps: int
    batch_size: int
    grad_accum: int
    mnrl_mini_batch: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark execution and cache reuse policy."""

    bench_fast: bool
    cache_embeddings: bool
    skip_existing_benchmarks: bool
    force_baseline_benchmark_rerun: bool


@dataclass(frozen=True)
class OomRetryConfig:
    """CUDA OOM retry behavior for training commands."""

    oom_retry_enabled: bool
    oom_retry_max_attempts: int
    oom_retry_min_batch: int


@dataclass(frozen=True)
class StageFlowConfig:
    """Stage-selection flags and stage artifact naming."""

    run_stage1: bool
    evaluate_stage1: bool
    stage2_use_stage1_init: bool
    stage2_init_model: str
    stage1_run_name: str
    stage2a_run_name: str
    stage2b_run_name: str
    stage1_model_path: Path
    stage2a_model_path: Path
    stage2b_model_path: Path


ENV_SPECS: dict[str, tuple[Callable[[str], object], object]] = {
    "PYTHON_BIN": (str, sys.executable),
    "BASE_MODEL": (str, "facebook/esm2_t12_35M_UR50D"),
    "RUN_PREFIX": (str, "ablation_v2_stagewise"),
    "TRAIN_DISTRIBUTED": (_parse_bool, True),
    "TRAIN_NUM_PROCESSES": (int, 0),
    "TRAIN_MIXED_PRECISION": (str, "bf16"),
    "PFAM_FILE": (str, "data/pfam_sorted.parquet"),
    "HARD_NEG_PARQUET": (str, "data/pfam_hard_negatives.parquet"),
    "AFDB_FILE": (str, "data/afdb_sorted.parquet"),
    "STRINGDB_FILE": (str, "data/stringdb/stringdb_train.parquet"),
    "DMS_COSENT_FILE": (str, "data/dms_cosent.parquet"),
    "POOLING_MODE": (str, "mean"),
    "OPTIMIZER": (str, "adamw_torch_fused"),
    "COMPILE_TRAINING": (_parse_bool, True),
    "SAVE_STEPS": (int, 9999),
    "MAX_MINUTES": (int, 30),
    "MAX_MAP_ROWS": (int, 1_500_000),
    "DMS_MAX_ROWS": (int, 0),
    "LEARNING_RATE": (float, 8e-5),
    "WARMUP_STEPS": (int, 50),
    "BATCH_SIZE": (int, 64),
    "GRAD_ACCUM": (int, 2),
    "MNRL_MINI_BATCH": (int, 0),
    "BENCH_FAST": (_parse_bool, True),
    "CACHE_EMBEDDINGS": (_parse_bool, True),
    "PROTEIN_PROGRESS_MIN_INTERVAL": (float, 4.0),
    "OOM_RETRY_ENABLED": (_parse_bool, True),
    "OOM_RETRY_MAX_ATTEMPTS": (int, 4),
    "OOM_RETRY_MIN_BATCH": (int, 16),
    "SKIP_EXISTING_BENCHMARKS": (_parse_bool, True),
    "FORCE_BASELINE_BENCHMARK_RERUN": (_parse_bool, True),
    "RUN_STAGE1": (_parse_bool, True),
    "STAGE2_USE_STAGE1_INIT": (_parse_bool, False),
    "STAGE2_INIT_MODEL": (str, ""),
    "TRAIN_CUDA_DEVICE": (str, ""),
    "BENCH_CUDA_DEVICE": (str, ""),
}


@dataclass(frozen=True)
class RunConfig:
    """Top-level grouped configuration for staged ablation orchestration."""

    repo_root: Path
    base_model: str
    run_prefix: str
    result_root: Path
    log_root: Path
    main_log: Path
    summary_csv: Path
    delta_csv: Path
    manifest_csv: Path
    batch_calibration_csv: Path
    paths: DataPaths
    runtime: RuntimeConfig
    train: TrainConfig
    benchmark: BenchmarkConfig
    oom_retry: OomRetryConfig
    stages: StageFlowConfig

    @staticmethod
    def from_env(repo_root: Path) -> "RunConfig":
        """Build a compact RunConfig from environment variables.

        Args:
            repo_root: Absolute path to the repository root.

        Returns:
            Fully populated, immutable RunConfig.
        """
        env = _load_env_values(ENV_SPECS)

        prefix = str(env["RUN_PREFIX"])
        run_stage1 = bool(env["RUN_STAGE1"])
        evaluate_stage1_raw = os.environ.get("EVALUATE_STAGE1")
        evaluate_stage1 = (
            _parse_bool(evaluate_stage1_raw)
            if evaluate_stage1_raw is not None
            else run_stage1
        )

        distributed = bool(env["TRAIN_DISTRIBUTED"])
        visible_devices = [
            dev.strip()
            for dev in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if dev.strip()
        ]
        auto_processes = len(visible_devices) if visible_devices else 1
        configured_processes = _as_int(env["TRAIN_NUM_PROCESSES"])
        train_processes = (
            max(1, configured_processes or auto_processes) if distributed else 1
        )

        batch_size = _as_int(env["BATCH_SIZE"])
        mnrl_default = max(1, batch_size // 2)
        mnrl_mini_batch = max(1, _as_int(env["MNRL_MINI_BATCH"]) or mnrl_default)

        s1_name = f"{prefix}_stage1_triplet"
        s2a_name = f"{prefix}_stage2a_cached_mnrl"
        s2b_name = f"{prefix}_stage2b_multi_dms"
        models_root = repo_root / "models"
        result_root = repo_root / "results" / prefix
        log_root = repo_root / "logs" / prefix

        return RunConfig(
            repo_root=repo_root,
            base_model=str(env["BASE_MODEL"]),
            run_prefix=prefix,
            result_root=result_root,
            log_root=log_root,
            main_log=log_root / "main.log",
            summary_csv=result_root / "summary.csv",
            delta_csv=result_root / "delta_vs_baseline.csv",
            manifest_csv=result_root / "experiment_manifest.csv",
            batch_calibration_csv=result_root / "batch_calibration.csv",
            paths=DataPaths(
                pfam_file=repo_root / str(env["PFAM_FILE"]),
                hard_neg_parquet=repo_root / str(env["HARD_NEG_PARQUET"]),
                afdb_file=repo_root / str(env["AFDB_FILE"]),
                stringdb_file=repo_root / str(env["STRINGDB_FILE"]),
                dms_cosent_file=repo_root / str(env["DMS_COSENT_FILE"]),
            ),
            runtime=RuntimeConfig(
                python_bin=str(env["PYTHON_BIN"]),
                train_distributed=distributed,
                train_num_processes=train_processes,
                train_mixed_precision=str(env["TRAIN_MIXED_PRECISION"]),
                progress_min_interval=max(
                    0.5,
                    _as_float(env["PROTEIN_PROGRESS_MIN_INTERVAL"]),
                ),
                train_cuda_device=str(env["TRAIN_CUDA_DEVICE"]).strip(),
                bench_cuda_device=str(env["BENCH_CUDA_DEVICE"]).strip(),
            ),
            train=TrainConfig(
                pooling_mode=str(env["POOLING_MODE"]),
                optimizer=str(env["OPTIMIZER"]),
                compile_training=bool(env["COMPILE_TRAINING"]),
                save_steps=_as_int(env["SAVE_STEPS"]),
                max_minutes=_as_int(env["MAX_MINUTES"]),
                max_map_rows=_as_int(env["MAX_MAP_ROWS"]),
                dms_max_rows=max(0, _as_int(env["DMS_MAX_ROWS"])),
                learning_rate=_as_float(env["LEARNING_RATE"]),
                warmup_steps=_as_int(env["WARMUP_STEPS"]),
                batch_size=batch_size,
                grad_accum=_as_int(env["GRAD_ACCUM"]),
                mnrl_mini_batch=mnrl_mini_batch,
            ),
            benchmark=BenchmarkConfig(
                bench_fast=bool(env["BENCH_FAST"]),
                cache_embeddings=bool(env["CACHE_EMBEDDINGS"]),
                skip_existing_benchmarks=bool(env["SKIP_EXISTING_BENCHMARKS"]),
                force_baseline_benchmark_rerun=bool(
                    env["FORCE_BASELINE_BENCHMARK_RERUN"]
                ),
            ),
            oom_retry=OomRetryConfig(
                oom_retry_enabled=bool(env["OOM_RETRY_ENABLED"]),
                oom_retry_max_attempts=max(0, _as_int(env["OOM_RETRY_MAX_ATTEMPTS"])),
                oom_retry_min_batch=max(1, _as_int(env["OOM_RETRY_MIN_BATCH"])),
            ),
            stages=StageFlowConfig(
                run_stage1=run_stage1,
                evaluate_stage1=evaluate_stage1,
                stage2_use_stage1_init=bool(env["STAGE2_USE_STAGE1_INIT"]),
                stage2_init_model=str(env["STAGE2_INIT_MODEL"]).strip(),
                stage1_run_name=s1_name,
                stage2a_run_name=s2a_name,
                stage2b_run_name=s2b_name,
                stage1_model_path=models_root / s1_name / "final",
                stage2a_model_path=models_root / s2a_name / "final",
                stage2b_model_path=models_root / s2b_name / "final",
            ),
        )


class Runner:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.cfg.result_root.mkdir(parents=True, exist_ok=True)
        self.cfg.log_root.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.cfg.main_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def require_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    def require_command(self, name: str) -> None:
        if shutil.which(name) is None:
            raise RuntimeError(f"Missing required command: {name}")

    def require_python_module(self, module_name: str) -> None:
        """Validate that the configured Python can import a required module."""
        result = subprocess.run(
            [self.cfg.runtime.python_bin, "-c", f"import {module_name}"],
            cwd=self.cfg.repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Missing required Python module in {self.cfg.runtime.python_bin}: "
                f"{module_name}"
            )

    def resolve_stage2_init_model(self) -> str:
        if self.cfg.stages.stage2_init_model:
            return self.cfg.stages.stage2_init_model
        if self.cfg.stages.stage2_use_stage1_init:
            if not self.cfg.stages.stage1_model_path.exists():
                raise FileNotFoundError(
                    "STAGE2_USE_STAGE1_INIT=1 but Stage1 model is missing: "
                    f"{self.cfg.stages.stage1_model_path}"
                )
            return str(self.cfg.stages.stage1_model_path)
        return self.cfg.base_model

    def run_cmd(
        self, cmd: list[str], log_file: Path, env: dict[str, str] | None = None
    ) -> None:
        merged_env = os.environ.copy()
        merged_env.setdefault("PROTEIN_PROGRESS_BARS", "auto")
        merged_env.setdefault(
            "PROTEIN_PROGRESS_MIN_INTERVAL", f"{self.cfg.runtime.progress_min_interval}"
        )
        merged_env.setdefault(
            "TQDM_MININTERVAL", f"{self.cfg.runtime.progress_min_interval}"
        )
        merged_env.setdefault("TQDM_MINITERS", "1")
        merged_env.setdefault("TQDM_DYNAMIC_NCOLS", "1")
        merged_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        merged_env.setdefault("NCCL_TIMEOUT", "6200")
        merged_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        merged_env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
        if env:
            merged_env.update(env)

        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                cwd=self.cfg.repo_root,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                handle.write(line)
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")

    @staticmethod
    def _get_int_flag(args: list[str], flag: str, default: int) -> int:
        if flag not in args:
            return default
        idx = args.index(flag)
        if idx + 1 >= len(args):
            return default
        try:
            return int(args[idx + 1])
        except ValueError:
            return default

    @staticmethod
    def _set_int_flag(args: list[str], flag: str, value: int) -> list[str]:
        updated = list(args)
        if flag in updated:
            idx = updated.index(flag)
            if idx + 1 < len(updated):
                updated[idx + 1] = str(value)
                return updated
        updated.extend([flag, str(value)])
        return updated

    @staticmethod
    def _log_has_oom(log_file: Path, start_offset: int = 0) -> bool:
        if not log_file.exists():
            return False
        with log_file.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            min_offset = min(max(start_offset, 0), size)
            tail_start = max(0, size - MAX_LOG_TAIL_BYTES)
            read_start = max(min_offset, tail_start)
            handle.seek(read_start, os.SEEK_SET)
            tail = handle.read().decode("utf-8", errors="ignore").lower()

        return any(pattern in tail for pattern in OOM_LOG_PATTERNS)

    @staticmethod
    def _bench_csv_is_valid(csv_file: Path) -> bool:
        if not csv_file.exists():
            return False
        try:
            df = pl.read_csv(csv_file)
        except Exception as exc:
            logger.warning("Failed to read benchmark CSV %s: %s", csv_file, exc)
            return False
        return "Task" in df.columns and df.height > 0

    def _common_train_args(
        self,
        *,
        model: str,
        files: Iterable[Path],
        loss_mode: str,
        max_minutes: int,
        learning_rate: float,
        warmup_steps: int,
        run_name: str,
        max_map_rows: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        hard_negatives: bool = False,
    ) -> list[str]:
        args = [
            "--model",
            model,
            "--files",
            *(str(path) for path in files),
            "--loss_mode",
            loss_mode,
        ]
        if hard_negatives:
            args.append("--hard_negatives")

        args.extend(
            [
                "--max_minutes",
                str(max_minutes),
                "--learning_rate",
                str(learning_rate),
                "--warmup_steps",
                str(warmup_steps),
                "--optim",
                self.cfg.train.optimizer,
                "--run_name",
                run_name,
                "--save_steps",
                str(self.cfg.train.save_steps),
                "--no_resume",
                "--max_map_rows",
                str(max_map_rows),
                "--batch_size",
                str(batch_size),
                "--gradient_accumulation_steps",
                str(gradient_accumulation_steps),
                "--multi_dataset_sampler",
                "round_robin",
                "--pooling_mode",
                self.cfg.train.pooling_mode,
            ]
        )
        return args

    @staticmethod
    def _normalize_metrics(
        df: pl.DataFrame, metrics: Iterable[str], add_missing: bool
    ) -> pl.DataFrame:
        for metric in metrics:
            if metric in df.columns:
                df = df.with_columns(pl.col(metric).cast(pl.Float64, strict=False))
            elif add_missing:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(metric))
        return df

    def find_bench_csv(self, search_dir: Path) -> Path | None:
        files = list(search_dir.rglob("bench_*.csv"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def clear_bench_csvs(self, search_dir: Path) -> None:
        for file in search_dir.rglob("bench_*.csv"):
            file.unlink(missing_ok=True)

    def extract_benchmark_means(self, csv_file: Path) -> tuple[str, str, str, str]:
        if not csv_file.exists():
            return ("N/A", "N/A", "N/A", "N/A")
        try:
            df = pl.read_csv(csv_file)
        except Exception:
            return ("N/A", "N/A", "N/A", "N/A")

        def _fmt_mean(col: str) -> str:
            if col not in df.columns:
                return "N/A"
            value = df.select(pl.col(col).drop_nans().mean()).item()
            if value is None:
                return "N/A"
            value = float(value)
            if math.isnan(value):
                return "N/A"
            return f"{value:.5f}"

        return (
            _fmt_mean("Spearman"),
            _fmt_mean("Accuracy"),
            _fmt_mean("F1"),
            _fmt_mean("AUC"),
        )

    @staticmethod
    def _latest_row(df: pl.DataFrame, stage: str) -> dict[str, Any] | None:
        """Return the most recently appended row for a stage, if present."""
        stage_rows = df.filter(pl.col("stage") == stage)
        if stage_rows.height == 0:
            return None
        return stage_rows.tail(1).row(0, named=True)

    def _append_csv_row(
        self,
        out_path: Path,
        fieldnames: list[str],
        row: dict[str, Any],
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        has_header = out_path.exists() and out_path.stat().st_size > 0
        with out_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not has_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    @staticmethod
    def _describe_run(
        experiment_id: str,
        stage: str,
        loss_mode: str,
        multi_primary_loss: str,
    ) -> str:
        if stage == "baseline":
            return "Baseline KNN benchmark for comparison anchor"
        if loss_mode == "multi":
            return (
                f"{experiment_id}: multi-task training with primary={multi_primary_loss} "
                "and DMS CoSENT"
            )
        return f"{experiment_id}: {loss_mode} training"

    def init_summary(self) -> None:
        self.cfg.summary_csv.write_text(
            ",".join(SUMMARY_COLUMNS) + "\n",
            encoding="utf-8",
        )

    def init_manifest(self) -> None:
        self.cfg.manifest_csv.write_text(
            ",".join(MANIFEST_COLUMNS) + "\n",
            encoding="utf-8",
        )

    def init_batch_calibration(self) -> None:
        self.cfg.batch_calibration_csv.write_text(
            ",".join(CALIBRATION_COLUMNS) + "\n",
            encoding="utf-8",
        )

    def load_calibrated_batches(self) -> dict[str, int]:
        """Load the latest successful calibrated batch for each experiment.

        Returns:
            Mapping of experiment id to the most recently selected stable batch.
            Invalid or missing calibration files are treated as empty.
        """
        if not self.cfg.batch_calibration_csv.exists():
            return {}

        try:
            df = pl.read_csv(self.cfg.batch_calibration_csv)
        except Exception as exc:
            self.log(
                "Ignoring unreadable calibration CSV: "
                f"{self.cfg.batch_calibration_csv} ({exc})"
            )
            return {}

        required = {"experiment_id", "success", "chosen_batch"}
        if not required.issubset(df.columns):
            self.log(
                "Ignoring calibration CSV missing required columns: "
                f"{self.cfg.batch_calibration_csv}"
            )
            return {}

        batches: dict[str, int] = {}
        for row in df.iter_rows(named=True):
            success_raw = row.get("success")
            chosen_raw = row.get("chosen_batch")
            experiment_raw = row.get("experiment_id")
            if experiment_raw in (None, ""):
                continue
            try:
                success = int(success_raw or 0)
                chosen_batch = int(chosen_raw or 0)
            except (TypeError, ValueError):
                continue
            if success and chosen_batch > 0:
                batches[str(experiment_raw)] = chosen_batch
        return batches

    def append_summary(
        self,
        *,
        experiment_id: str,
        stage: str,
        model_path: str,
        knn_csv: Path,
        loss_mode: str,
        multi_primary_loss: str,
        train_files: tuple[Path, ...],
        dms_enabled: bool,
        batch_size: int,
        grad_accum: int,
        mnrl_mini_batch: int,
        max_minutes: int,
    ) -> None:
        sp, acc, f1, auc = self.extract_benchmark_means(knn_csv)
        row: dict[str, Any] = {
            "run_prefix": self.cfg.run_prefix,
            "experiment_id": experiment_id,
            "stage": stage,
            "model_path": model_path,
            "loss_mode": loss_mode,
            "multi_primary_loss": multi_primary_loss,
            "train_files": ";".join(str(path) for path in train_files),
            "dms_enabled": int(dms_enabled),
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "mnrl_mini_batch": mnrl_mini_batch,
            "max_minutes": max_minutes,
            "knn_spearman_mean": sp,
            "knn_accuracy_mean": acc,
            "knn_f1_mean": f1,
            "knn_auc_mean": auc,
            "knn_csv": str(knn_csv),
            "Desc": self._describe_run(
                experiment_id=experiment_id,
                stage=stage,
                loss_mode=loss_mode,
                multi_primary_loss=multi_primary_loss,
            ),
        }
        self._append_csv_row(self.cfg.summary_csv, SUMMARY_COLUMNS, row)

    def append_manifest(self, record: RunRecord) -> None:
        row: dict[str, Any] = {
            "run_prefix": self.cfg.run_prefix,
            "experiment_id": record.experiment_id,
            "stage": record.stage,
            "status": record.status,
            "model_path": str(record.model_path),
            "run_name": record.run_name,
            "loss_mode": record.loss_mode,
            "multi_primary_loss": record.multi_primary_loss,
            "train_files": ";".join(str(path) for path in record.files),
            "batch_size": record.batch_size,
            "grad_accum": record.grad_accum,
            "mnrl_mini_batch": record.mnrl_mini_batch,
            "max_minutes": record.max_minutes,
            "knn_csv": str(record.knn_csv),
            "retries": record.retries,
            "params_json": json.dumps(
                {
                    "loss_mode": record.loss_mode,
                    "multi_primary_loss": record.multi_primary_loss,
                    "batch_size": record.batch_size,
                    "grad_accum": record.grad_accum,
                    "mnrl_mini_batch": record.mnrl_mini_batch,
                    "max_minutes": record.max_minutes,
                },
                sort_keys=True,
            ),
            "Desc": self._describe_run(
                experiment_id=record.experiment_id,
                stage=record.stage,
                loss_mode=record.loss_mode,
                multi_primary_loss=record.multi_primary_loss,
            ),
        }
        self._append_csv_row(self.cfg.manifest_csv, MANIFEST_COLUMNS, row)

    def append_batch_calibration(
        self,
        *,
        experiment_id: str,
        tested_batch: int,
        success: bool,
        retries: int,
        chosen_batch: int,
    ) -> None:
        status = "success" if success else "failed"
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "tested_batch": tested_batch,
            "success": int(success),
            "retries": retries,
            "chosen_batch": chosen_batch,
            "Desc": (
                f"{experiment_id}: calibration {status} at batch={tested_batch}; "
                f"selected={chosen_batch}"
            ),
        }
        self._append_csv_row(
            self.cfg.batch_calibration_csv,
            CALIBRATION_COLUMNS,
            row,
        )

    def run_training(
        self,
        args: list[str],
        log_file: Path,
    ) -> int:
        """Launch training, retrying on OOM by halving batch size.

        Args:
            args: Argument list for ``protein_pipeline.py train``.
            log_file: Path to append combined stdout/stderr output.

        Returns:
            Number of OOM retry attempts consumed (0 = first try succeeded).

        Raises:
            RuntimeError: If training fails for a non-OOM reason or retries
                are exhausted.
        """
        train_env: dict[str, str] | None = (
            {"CUDA_VISIBLE_DEVICES": self.cfg.runtime.train_cuda_device}
            if self.cfg.runtime.train_cuda_device
            else None
        )
        run_args = list(args)
        attempt = 0

        while True:
            attempt_log_start = log_file.stat().st_size if log_file.exists() else 0
            if (
                self.cfg.runtime.train_distributed
                and self.cfg.runtime.train_num_processes > 1
            ):
                cmd = [
                    self.cfg.runtime.python_bin,
                    "-m",
                    "accelerate.commands.launch",
                    "--num_processes",
                    str(self.cfg.runtime.train_num_processes),
                    "--mixed_precision",
                    self.cfg.runtime.train_mixed_precision,
                    "--main_process_port",
                    "0",
                    "protein_pipeline.py",
                    "train",
                    *run_args,
                ]
            else:
                cmd = [
                    self.cfg.runtime.python_bin,
                    "protein_pipeline.py",
                    "train",
                    *run_args,
                ]

            try:
                self.run_cmd(cmd, log_file=log_file, env=train_env)
                return attempt
            except RuntimeError as exc:
                saw_oom = self._log_has_oom(log_file, start_offset=attempt_log_start)

                if not self.cfg.oom_retry.oom_retry_enabled:
                    raise
                if not saw_oom:
                    raise
                if attempt >= self.cfg.oom_retry.oom_retry_max_attempts:
                    raise RuntimeError(
                        f"{exc}\nDetected CUDA OOM and exhausted OOM retries "
                        f"({self.cfg.oom_retry.oom_retry_max_attempts})."
                    ) from exc

                current_batch = self._get_int_flag(run_args, "--batch_size", 1)
                current_grad = self._get_int_flag(
                    run_args, "--gradient_accumulation_steps", 1
                )
                if current_batch <= self.cfg.oom_retry.oom_retry_min_batch:
                    raise RuntimeError(
                        f"{exc}\nDetected CUDA OOM but --batch_size={current_batch} "
                        f"is already at/below OOM_RETRY_MIN_BATCH="
                        f"{self.cfg.oom_retry.oom_retry_min_batch}."
                    ) from exc

                new_batch = max(
                    self.cfg.oom_retry.oom_retry_min_batch, current_batch // 2
                )
                if new_batch == current_batch:
                    raise RuntimeError(
                        f"{exc}\nDetected CUDA OOM but could not reduce --batch_size "
                        f"below {current_batch}."
                    ) from exc

                scale = math.ceil(current_batch / new_batch)
                new_grad = max(1, current_grad * scale)
                attempt += 1

                self.log(
                    "Detected CUDA OOM. Retrying with reduced memory pressure: "
                    f"attempt={attempt}/{self.cfg.oom_retry.oom_retry_max_attempts}, "
                    f"batch_size {current_batch}->{new_batch}, "
                    f"gradient_accumulation_steps {current_grad}->{new_grad}"
                )

                run_args = self._set_int_flag(run_args, "--batch_size", new_batch)
                run_args = self._set_int_flag(
                    run_args,
                    "--gradient_accumulation_steps",
                    new_grad,
                )

    def run_knn_benchmark(
        self, stage: str, model_path: str, force_rerun: bool = False
    ) -> Path:
        output_root = self.cfg.result_root / stage / "knn"
        output_root.mkdir(parents=True, exist_ok=True)
        log_file = self.cfg.log_root / f"{stage}_knn.log"
        task_keys = _full_probe_task_keys()

        if force_rerun:
            self.clear_bench_csvs(output_root)
        elif self.cfg.benchmark.skip_existing_benchmarks:
            existing = self.find_bench_csv(output_root)
            if existing is not None:
                if self._bench_csv_is_valid(existing):
                    self.log(f"Reusing existing KNN benchmark for {stage}: {existing}")
                    return existing
                self.log(
                    f"Existing KNN benchmark is invalid/incomplete for {stage}: {existing}; rerunning"
                )

        cmd = [
            self.cfg.runtime.python_bin,
            "protein_benchmark_suite.py",
            "--model_name",
            model_path,
            "--eval_split",
            "validation",
            "--probe_type",
            "knn",
            "--tasks",
            *task_keys,
            "--device",
            "cuda",
            "--output_dir",
            str(output_root),
        ]
        if self.cfg.benchmark.bench_fast:
            cmd.extend(["--max_samples", "100000"])
        if self.cfg.benchmark.cache_embeddings:
            cmd.append("--cache_embeddings")

        bench_env: dict[str, str] | None = (
            {"CUDA_VISIBLE_DEVICES": self.cfg.runtime.bench_cuda_device}
            if self.cfg.runtime.bench_cuda_device
            else None
        )
        self.log(f"Running KNN benchmark for {stage}")
        self.run_cmd(cmd, log_file=log_file, env=bench_env)

        csv_file = self.find_bench_csv(output_root)
        if csv_file is None or not self._bench_csv_is_valid(csv_file):
            raise RuntimeError(f"KNN benchmark did not produce a CSV for {stage}")
        return csv_file

    def run_stage1(self) -> None:
        if self.cfg.stages.stage1_model_path.exists():
            self.log(
                f"Reusing existing Stage1 model: {self.cfg.stages.stage1_model_path}"
            )
            return

        self.log(f"Training Stage1 Triplet (minutes={self.cfg.train.max_minutes})")
        args = self._common_train_args(
            model=self.cfg.base_model,
            files=(self.cfg.paths.pfam_file, self.cfg.paths.afdb_file),
            loss_mode="triplet",
            max_minutes=self.cfg.train.max_minutes,
            learning_rate=self.cfg.train.learning_rate,
            warmup_steps=self.cfg.train.warmup_steps,
            run_name=self.cfg.stages.stage1_run_name,
            max_map_rows=self.cfg.train.max_map_rows,
            batch_size=self.cfg.train.batch_size,
            gradient_accumulation_steps=self.cfg.train.grad_accum,
        )
        args.extend(
            [
                "--batch_sampler",
                "group_by_label",
                "--min_label_count",
                "2",
                "--triplet_max_samples_per_label",
                "0",
            ]
        )
        if self.cfg.train.compile_training:
            args.append("--compile")
        self.run_training(args, log_file=self.cfg.log_root / "stage1_train.log")
        self.require_path(self.cfg.stages.stage1_model_path)

    def run_stage2(
        self, stage_name: str, run_name: str, model_path: Path, loss_mode: str
    ) -> None:
        if model_path.exists():
            self.log(f"Reusing existing {stage_name} model: {model_path}")
            return

        init_model = self.resolve_stage2_init_model()

        self.log(
            f"Training {stage_name} ({loss_mode}, minutes={self.cfg.train.max_minutes})"
        )
        args = self._common_train_args(
            model=init_model,
            files=(
                self.cfg.paths.hard_neg_parquet,
                self.cfg.paths.afdb_file,
                self.cfg.paths.stringdb_file,
            ),
            loss_mode=loss_mode,
            max_minutes=self.cfg.train.max_minutes,
            learning_rate=self.cfg.train.learning_rate,
            warmup_steps=self.cfg.train.warmup_steps,
            run_name=run_name,
            max_map_rows=self.cfg.train.max_map_rows,
            batch_size=self.cfg.train.batch_size,
            gradient_accumulation_steps=self.cfg.train.grad_accum,
            hard_negatives=True,
        )
        args.extend(
            [
                "--mnrl_mini_batch_size",
                str(self.cfg.train.mnrl_mini_batch),
            ]
        )
        if loss_mode == "multi":
            args.extend(
                [
                    "--multi_mnrl_loss",
                    "cached_mnrl",
                    "--dms_file",
                    str(self.cfg.paths.dms_cosent_file),
                ]
            )
            if self.cfg.train.dms_max_rows > 0:
                args.extend(["--dms_max_rows", str(self.cfg.train.dms_max_rows)])
        if self.cfg.train.compile_training:
            args.append("--compile")
        self.log(f"{stage_name}: init model={init_model}")
        self.run_training(args, log_file=self.cfg.log_root / f"{stage_name}_train.log")
        self.require_path(model_path)

    def evaluate_stage(
        self,
        stage: str,
        model_path: str,
        force_rerun: bool = False,
        *,
        experiment_id: str | None = None,
        loss_mode: str = "benchmark_only",
        multi_primary_loss: str = "",
        train_files: tuple[Path, ...] = (),
        dms_enabled: bool = False,
        batch_size: int | None = None,
        grad_accum: int | None = None,
        mnrl_mini_batch: int | None = None,
        max_minutes: int | None = None,
    ) -> None:
        knn_csv = self.run_knn_benchmark(stage, model_path, force_rerun=force_rerun)
        self.append_summary(
            experiment_id=experiment_id or stage,
            stage=stage,
            model_path=model_path,
            knn_csv=knn_csv,
            loss_mode=loss_mode,
            multi_primary_loss=multi_primary_loss,
            train_files=train_files,
            dms_enabled=dms_enabled,
            batch_size=batch_size or self.cfg.train.batch_size,
            grad_accum=grad_accum or self.cfg.train.grad_accum,
            mnrl_mini_batch=mnrl_mini_batch or self.cfg.train.mnrl_mini_batch,
            max_minutes=max_minutes or self.cfg.train.max_minutes,
        )

    @staticmethod
    def _metric_direction(metric: str) -> str:
        return "lower_is_better" if metric == "MSE" else "higher_is_better"

    @classmethod
    def _metric_status(cls, metric: str, delta: float) -> str:
        if delta == 0:
            return "unchanged"
        if cls._metric_direction(metric) == "lower_is_better":
            return "improved" if delta < 0 else "worse"
        return "improved" if delta > 0 else "worse"

    def write_baseline_vs_stage1_readable_report(self) -> Path | None:
        if not self.cfg.summary_csv.exists():
            return None

        summary_df = pl.read_csv(self.cfg.summary_csv)
        baseline_row = self._latest_row(summary_df, "baseline")
        stage1_row = self._latest_row(summary_df, "stage1_triplet")
        if baseline_row is None or stage1_row is None:
            return None

        baseline_csv = Path(str(baseline_row["knn_csv"]))
        stage1_csv = Path(str(stage1_row["knn_csv"]))
        if not baseline_csv.exists() or not stage1_csv.exists():
            return None

        baseline_df = pl.read_csv(baseline_csv)
        stage1_df = pl.read_csv(stage1_csv)
        if "Task" not in baseline_df.columns or "Task" not in stage1_df.columns:
            return None

        join_keys = ["Task"]
        if "Probe" in baseline_df.columns and "Probe" in stage1_df.columns:
            join_keys.append("Probe")

        metrics = REPORT_METRICS
        baseline_df = self._normalize_metrics(
            baseline_df,
            metrics,
            add_missing=False,
        )
        stage1_df = self._normalize_metrics(
            stage1_df,
            metrics,
            add_missing=False,
        )

        baseline_keep = [*join_keys, *[m for m in metrics if m in baseline_df.columns]]
        stage1_keep = [*join_keys, *[m for m in metrics if m in stage1_df.columns]]
        baseline_df = baseline_df.select(baseline_keep).rename(
            {m: f"baseline_{m}" for m in metrics if m in baseline_keep}
        )
        stage1_df = stage1_df.select(stage1_keep).rename(
            {m: f"stage1_{m}" for m in metrics if m in stage1_keep}
        )

        joined = stage1_df.join(baseline_df, on=join_keys, how="inner")
        if joined.height == 0:
            return None

        records: list[dict[str, str | float]] = []
        for row in joined.iter_rows(named=True):
            task = str(row["Task"])
            probe = str(row["Probe"]) if "Probe" in row else "knn"
            for metric in metrics:
                stage_col = f"stage1_{metric}"
                base_col = f"baseline_{metric}"
                if stage_col not in row or base_col not in row:
                    continue
                stage_val = row.get(stage_col)
                base_val = row.get(base_col)
                if stage_val is None or base_val is None:
                    continue
                stage_float = float(stage_val)
                base_float = float(base_val)
                if math.isnan(stage_float) or math.isnan(base_float):
                    continue
                delta = stage_float - base_float
                status = self._metric_status(metric, delta)
                records.append(
                    {
                        "task": task,
                        "probe": probe,
                        "metric": metric,
                        "better_direction": self._metric_direction(metric),
                        "baseline_value": base_float,
                        "stage1_value": stage_float,
                        "raw_delta_stage1_minus_baseline": delta,
                        "status": status,
                        "Desc": (
                            f"{metric} {status}; stage1={stage_float:.5f}, "
                            f"baseline={base_float:.5f}, delta={delta:+.5f}"
                        ),
                    }
                )

        if not records:
            return None

        out_path = self.cfg.result_root / "baseline_vs_stage1_readable.csv"
        pl.DataFrame(records).sort(["task", "metric"]).write_csv(out_path)
        return out_path

    def write_delta_report(self) -> Path | None:
        summary_df = pl.read_csv(self.cfg.summary_csv)
        baseline_row = self._latest_row(summary_df, "baseline")
        if baseline_row is None:
            return None

        stage_rows = summary_df.filter(pl.col("stage") != "baseline")
        if stage_rows.height == 0:
            return None

        baseline_csv = Path(str(baseline_row["knn_csv"]))
        baseline_df = pl.read_csv(baseline_csv)
        if "Task" not in baseline_df.columns:
            raise RuntimeError("Baseline KNN CSV is missing Task column")

        join_keys = ["Task"]
        if "Probe" in baseline_df.columns:
            join_keys.append("Probe")

        metrics = REPORT_METRICS

        baseline_df = self._normalize_metrics(
            baseline_df,
            metrics,
            add_missing=True,
        )
        baseline_keep = [*join_keys, *metrics]
        baseline_keep = [c for c in baseline_keep if c in baseline_df.columns]
        baseline_df = baseline_df.select(baseline_keep).rename(
            {c: f"baseline_{c}" for c in metrics if c in baseline_keep}
        )

        reports: list[pl.DataFrame] = []
        for row in stage_rows.iter_rows(named=True):
            stage = str(row["stage"])
            stage_csv = Path(str(row["knn_csv"]))
            if not stage_csv.exists():
                continue
            stage_df = pl.read_csv(stage_csv)
            if "Task" not in stage_df.columns:
                continue
            stage_df = self._normalize_metrics(
                stage_df,
                metrics,
                add_missing=True,
            )
            stage_keep = [*join_keys, *metrics]
            stage_keep = [c for c in stage_keep if c in stage_df.columns]
            stage_df = stage_df.select(stage_keep)

            joined = stage_df.join(baseline_df, on=join_keys, how="inner")
            if joined.height == 0:
                continue

            delta_exprs = []
            for metric in metrics:
                base_col = f"baseline_{metric}"
                if metric in joined.columns and base_col in joined.columns:
                    delta_exprs.append(
                        (pl.col(metric) - pl.col(base_col)).alias(f"delta_{metric}")
                    )
            if not delta_exprs:
                continue

            report_df = joined.with_columns(delta_exprs).with_columns(
                pl.lit(stage).alias("stage")
            )
            keep = ["stage", *join_keys]
            for metric in metrics:
                for col in (metric, f"baseline_{metric}", f"delta_{metric}"):
                    if col in report_df.columns:
                        keep.append(col)
            reports.append(report_df.select(keep))

        if not reports:
            raise RuntimeError("No baseline delta rows could be produced")

        delta_df = pl.concat(reports, how="diagonal_relaxed").sort(["stage", "Task"])
        desc_values: list[str] = []
        for row in delta_df.iter_rows(named=True):
            snippets: list[str] = []
            for metric in metrics:
                key = f"delta_{metric}"
                if key not in row:
                    continue
                value = row.get(key)
                if value is None:
                    continue
                delta = float(value)
                if math.isnan(delta):
                    continue
                status = self._metric_status(metric, delta)
                snippets.append(f"{metric} {status} ({delta:+.5f})")
            desc_values.append(
                "; ".join(snippets) if snippets else "no comparable metrics"
            )
        delta_df = delta_df.with_columns(pl.Series("Desc", desc_values))
        delta_df.write_csv(self.cfg.delta_csv)
        return self.cfg.delta_csv

    @staticmethod
    def _normalize_stage_selection(stages: set[str] | None) -> set[str]:
        """Normalize CLI stage selection to concrete pipeline stage names."""
        selected = set(stages or {"all"})
        if "all" in selected:
            return set(PIPELINE_STAGES)
        invalid = selected - set(PIPELINE_STAGES)
        if invalid:
            invalid_text = ", ".join(sorted(invalid))
            raise ValueError(f"Unsupported stage selection: {invalid_text}")
        return selected

    @staticmethod
    def _normalize_experiment_selection(experiments: set[str] | None) -> set[str]:
        """Normalize experiment selection to concrete experiment IDs."""
        selected = set(experiments or {"all"})
        if "all" in selected:
            return set(EXPERIMENT_IDS)
        invalid = selected - set(EXPERIMENT_IDS)
        if invalid:
            invalid_text = ", ".join(sorted(invalid))
            raise ValueError(f"Unsupported experiment selection: {invalid_text}")
        return selected

    def _experiment_presets(self) -> dict[str, ExperimentPreset]:
        """Return single-stage experiment presets for independent runs."""
        hard_neg_files = (
            self.cfg.paths.hard_neg_parquet,
            self.cfg.paths.afdb_file,
            self.cfg.paths.stringdb_file,
        )
        pfam_files = (
            self.cfg.paths.pfam_file,
            self.cfg.paths.afdb_file,
            self.cfg.paths.stringdb_file,
        )
        return {
            "baseline_eval_only": ExperimentPreset(
                experiment_id="baseline_eval_only",
                run_name_suffix="baseline_eval_only",
                stage_name="baseline",
                loss_mode="benchmark_only",
                multi_primary_loss=None,
                files=(),
                hard_negatives=False,
                dms_enabled=False,
            ),
            "triplet_cosent_multi": ExperimentPreset(
                experiment_id="triplet_cosent_multi",
                run_name_suffix="triplet_cosent_multi",
                stage_name="triplet_cosent_multi",
                loss_mode="multi",
                multi_primary_loss="triplet",
                files=(self.cfg.paths.pfam_file, self.cfg.paths.afdb_file),
                hard_negatives=False,
                dms_enabled=True,
            ),
            "mnrl_cosent_multi": ExperimentPreset(
                experiment_id="mnrl_cosent_multi",
                run_name_suffix="mnrl_cosent_multi",
                stage_name="mnrl_cosent_multi",
                loss_mode="multi",
                multi_primary_loss="mnrl",
                files=pfam_files,
                hard_negatives=False,
                dms_enabled=True,
            ),
            "cached_mnrl_cosent_multi": ExperimentPreset(
                experiment_id="cached_mnrl_cosent_multi",
                run_name_suffix="cached_mnrl_cosent_multi",
                stage_name="cached_mnrl_cosent_multi",
                loss_mode="multi",
                multi_primary_loss="cached_mnrl",
                files=hard_neg_files,
                hard_negatives=True,
                dms_enabled=True,
            ),
            "cached_gist_cosent_multi": ExperimentPreset(
                experiment_id="cached_gist_cosent_multi",
                run_name_suffix="cached_gist_cosent_multi",
                stage_name="cached_gist_cosent_multi",
                loss_mode="multi",
                multi_primary_loss="cached_gist",
                files=hard_neg_files,
                hard_negatives=True,
                dms_enabled=True,
            ),
            "gist_cosent_multi": ExperimentPreset(
                experiment_id="gist_cosent_multi",
                run_name_suffix="gist_cosent_multi",
                stage_name="gist_cosent_multi",
                loss_mode="multi",
                multi_primary_loss="gist",
                files=pfam_files,
                hard_negatives=False,
                dms_enabled=True,
            ),
            # DMS ablation: identical to cached_mnrl_cosent_multi without DMS signal.
            "cached_mnrl_no_dms": ExperimentPreset(
                experiment_id="cached_mnrl_no_dms",
                run_name_suffix="cached_mnrl_no_dms",
                stage_name="cached_mnrl_no_dms",
                loss_mode="multi",
                multi_primary_loss="cached_mnrl",
                files=hard_neg_files,
                hard_negatives=True,
                dms_enabled=False,
            ),
            # Two-stage: triplet warm-start -> cached_mnrl fine-tune (15+15 min).
            "staged_cached_mnrl": ExperimentPreset(
                experiment_id="staged_cached_mnrl",
                run_name_suffix="staged_cached_mnrl",
                stage_name="staged_cached_mnrl",
                loss_mode="multi",
                multi_primary_loss="cached_mnrl",
                files=hard_neg_files,
                hard_negatives=True,
                dms_enabled=True,
                chain_from_experiment="triplet_cosent_multi",
            ),
        }

    def _required_paths_for_experiments(self, selected: set[str]) -> list[Path]:
        """Return required dataset paths for selected experiment presets."""
        presets = self._experiment_presets()
        required: list[Path] = []
        for experiment_id in selected:
            preset = presets[experiment_id]
            required.extend(preset.files)
            if preset.dms_enabled:
                required.append(self.cfg.paths.dms_cosent_file)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in required:
            if path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _required_paths_for_selection(self, selected: set[str]) -> list[Path]:
        """Return required dataset paths for the selected pipeline stages."""
        required: list[Path] = []
        if "stage1" in selected:
            required.extend([self.cfg.paths.pfam_file, self.cfg.paths.afdb_file])
        if "stage2a" in selected or "stage2b" in selected:
            required.extend(
                [
                    self.cfg.paths.hard_neg_parquet,
                    self.cfg.paths.afdb_file,
                    self.cfg.paths.stringdb_file,
                ]
            )
        if "stage2b" in selected:
            required.append(self.cfg.paths.dms_cosent_file)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in required:
            if path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _build_experiment_train_args(
        self,
        *,
        preset: ExperimentPreset,
        batch_size: int,
        run_name: str,
        max_steps: int,
        max_minutes: int,
        init_model: str | None = None,
    ) -> list[str]:
        """Build a train command argument list for one experiment preset.

        Args:
            preset: Experiment configuration.
            batch_size: Per-device training batch size.
            run_name: Unique run identifier used for output directory naming.
            max_steps: Hard step cap (0 = unlimited, time-limited by max_minutes).
            max_minutes: Wall-clock training budget in minutes.
            init_model: Override base model checkpoint path.  If ``None``,
                falls back to ``cfg.base_model``.

        Returns:
            Argument list suitable for ``protein_pipeline.py train``.
        """
        primary = preset.multi_primary_loss or "mnrl"
        # Keep legacy behavior for non-cached flows (0 means let trainer derive
        # per-task behavior), but force pipeline default mini-batch for cached
        # losses to keep this campaign stable and reproducible.
        if primary in {"cached_mnrl", "cached_gist"}:
            mnrl_mini_batch_size = PIPELINE_DEFAULT_MNRL_MINI_BATCH
        else:
            mnrl_mini_batch_size = self.cfg.train.mnrl_mini_batch

        args = self._common_train_args(
            model=init_model if init_model is not None else self.cfg.base_model,
            files=preset.files,
            loss_mode=preset.loss_mode,
            max_minutes=max_minutes,
            learning_rate=self.cfg.train.learning_rate,
            warmup_steps=self.cfg.train.warmup_steps,
            run_name=run_name,
            max_map_rows=self.cfg.train.max_map_rows,
            batch_size=batch_size,
            gradient_accumulation_steps=self.cfg.train.grad_accum,
            hard_negatives=preset.hard_negatives,
        )
        args.extend(["--max_steps", str(max_steps)])
        args.extend(["--mnrl_mini_batch_size", str(mnrl_mini_batch_size)])

        if preset.loss_mode == "multi":
            args.extend(["--multi_primary_loss", primary])
            if primary in {"mnrl", "cached_mnrl"}:
                args.extend(["--multi_mnrl_loss", primary])
            if preset.dms_enabled:
                args.extend(["--dms_file", str(self.cfg.paths.dms_cosent_file)])
                if self.cfg.train.dms_max_rows > 0:
                    args.extend(["--dms_max_rows", str(self.cfg.train.dms_max_rows)])
            if primary == "triplet":
                args.extend(
                    [
                        "--batch_sampler",
                        "group_by_label",
                        "--min_label_count",
                        "2",
                        "--triplet_max_samples_per_label",
                        "0",
                    ]
                )

        if self.cfg.train.compile_training:
            args.append("--compile")

        return args

    def _default_batch_for_preset(self, preset: ExperimentPreset) -> int:
        """Return fixed fallback batch size when smoke calibration is disabled."""
        primary = preset.multi_primary_loss or "mnrl"
        if primary in {"cached_mnrl", "cached_gist"}:
            return DEFAULT_CACHED_BATCH
        return DEFAULT_NON_CACHED_BATCH

    def _effective_mnrl_mini_batch_for_preset(self, preset: ExperimentPreset) -> int:
        """Return the mini-batch size effectively passed to protein_pipeline."""
        primary = preset.multi_primary_loss or "mnrl"
        if primary in {"cached_mnrl", "cached_gist"}:
            return PIPELINE_DEFAULT_MNRL_MINI_BATCH
        return self.cfg.train.mnrl_mini_batch

    def _train_single_experiment(
        self,
        *,
        preset: ExperimentPreset,
        batch_size: int,
        max_steps: int,
        max_minutes: int,
    ) -> tuple[Path, str, int, str]:
        """Execute the training phase of one experiment and return metadata.

        Handles ``chain_from_experiment``: when set, ensures the parent model
        exists (training it for ``max_minutes // 2`` if needed) and uses it as
        the init checkpoint, also capping the current stage at half the budget.

        Args:
            preset: Experiment configuration.
            batch_size: Per-device training batch size.
            max_steps: Hard step cap (0 = unlimited).
            max_minutes: Total wall-clock training budget in minutes.

        Returns:
            Tuple of ``(model_path, run_name, retries, status)``.

        Raises:
            FileNotFoundError: If the expected model output is missing after
                training completes.
            RuntimeError: If training fails.
        """
        run_name = f"{self.cfg.run_prefix}_{preset.run_name_suffix}"
        model_path = self.cfg.repo_root / "models" / run_name / "final"
        log_file = self.cfg.log_root / f"{preset.stage_name}_train.log"

        init_model: str | None = None
        effective_minutes = max_minutes

        if preset.chain_from_experiment is not None:
            parent_preset = self._experiment_presets()[preset.chain_from_experiment]
            parent_run_name = f"{self.cfg.run_prefix}_{parent_preset.run_name_suffix}"
            parent_model_path = (
                self.cfg.repo_root / "models" / parent_run_name / "final"
            )
            effective_minutes = max(1, max_minutes // 2)
            parent_batch_size = min(
                batch_size,
                self._default_batch_for_preset(parent_preset),
            )
            if not parent_model_path.exists():
                self.log(
                    f"Training chain parent {parent_preset.experiment_id} "
                    f"({effective_minutes} min, batch={parent_batch_size}) "
                    f"before {preset.experiment_id}"
                )
                parent_args = self._build_experiment_train_args(
                    preset=parent_preset,
                    batch_size=parent_batch_size,
                    run_name=parent_run_name,
                    max_steps=max_steps,
                    max_minutes=effective_minutes,
                )
                parent_log = (
                    self.cfg.log_root / f"{parent_preset.stage_name}_chain_train.log"
                )
                self.run_training(parent_args, log_file=parent_log)
                self.require_path(parent_model_path)
            else:
                self.log(f"Reusing chain parent model: {parent_model_path}")
            init_model = str(parent_model_path)

        retries = 0
        status = "reused"
        if model_path.exists():
            self.log(f"Reusing existing model for {preset.experiment_id}: {model_path}")
        else:
            self.log(
                f"Training {preset.experiment_id}: loss={preset.loss_mode}, "
                f"primary={preset.multi_primary_loss or 'n/a'}, batch={batch_size}"
            )
            train_args = self._build_experiment_train_args(
                preset=preset,
                batch_size=batch_size,
                run_name=run_name,
                max_steps=max_steps,
                max_minutes=effective_minutes,
                init_model=init_model,
            )
            retries = self.run_training(train_args, log_file=log_file)
            self.require_path(model_path)
            status = "trained"

        return model_path, run_name, retries, status

    def _run_single_experiment(
        self,
        *,
        preset: ExperimentPreset,
        batch_size: int,
        max_steps: int,
        max_minutes: int,
    ) -> RunRecord:
        """Train one experiment, benchmark it, and return a complete RunRecord."""
        model_path, run_name, retries, status = self._train_single_experiment(
            preset=preset,
            batch_size=batch_size,
            max_steps=max_steps,
            max_minutes=max_minutes,
        )
        knn_csv = self.run_knn_benchmark(preset.stage_name, str(model_path))
        return RunRecord(
            experiment_id=preset.experiment_id,
            stage=preset.stage_name,
            run_name=run_name,
            model_path=model_path,
            knn_csv=knn_csv,
            loss_mode=preset.loss_mode,
            multi_primary_loss=preset.multi_primary_loss or "",
            files=preset.files,
            batch_size=batch_size,
            grad_accum=self.cfg.train.grad_accum,
            mnrl_mini_batch=self._effective_mnrl_mini_batch_for_preset(preset),
            max_minutes=max_minutes,
            retries=retries,
            status=status,
        )

    def _calibrate_experiment_batch(
        self,
        *,
        preset: ExperimentPreset,
        smoke_max_steps: int,
    ) -> int:
        """Find the highest stable batch size from a preset ladder via short smokes."""
        ladder = CALIBRATION_LADDERS.get(
            preset.experiment_id, (self.cfg.train.batch_size,)
        )
        chosen_batch = 0

        for tested_batch in ladder:
            smoke_run_name = (
                f"{self.cfg.run_prefix}_{preset.run_name_suffix}_smoke_b{tested_batch}"
            )
            smoke_dir = self.cfg.repo_root / "models" / smoke_run_name
            smoke_log = (
                self.cfg.log_root / f"{preset.stage_name}_smoke_b{tested_batch}.log"
            )
            smoke_dir.parent.mkdir(parents=True, exist_ok=True)
            smoke_dir.mkdir(parents=True, exist_ok=True)

            retries = 0
            success = False
            try:
                train_args = self._build_experiment_train_args(
                    preset=preset,
                    batch_size=tested_batch,
                    run_name=smoke_run_name,
                    max_steps=max(1, smoke_max_steps),
                    max_minutes=min(5, max(1, self.cfg.train.max_minutes)),
                )
                retries = self.run_training(train_args, log_file=smoke_log)
                success = True
                chosen_batch = tested_batch
            except RuntimeError as exc:
                if (
                    "oom" not in str(exc).lower()
                    and "out of memory" not in str(exc).lower()
                ):
                    raise
            finally:
                self.append_batch_calibration(
                    experiment_id=preset.experiment_id,
                    tested_batch=tested_batch,
                    success=success,
                    retries=retries,
                    chosen_batch=chosen_batch,
                )
                shutil.rmtree(smoke_dir, ignore_errors=True)

            if not success:
                break

        if chosen_batch <= 0:
            raise RuntimeError(
                f"Calibration failed for {preset.experiment_id}: no stable batch in {ladder}"
            )
        return chosen_batch

    def run_experiment_campaign(
        self,
        *,
        experiments: set[str],
        reset_summary: bool,
        calibrate_smoke: bool,
        calibrate_only: bool,
        smoke_max_steps: int,
    ) -> None:
        """Run experiment presets with CSV reporting and optional calibration.

        Training runs sequentially on the training device while benchmarks for
        already-trained models are dispatched concurrently to a single-worker
        thread pool (typically a dedicated benchmark GPU).  CSV writes are
        performed in the main thread in fixed experiment order to preserve
        reproducible artifact ordering.

        Args:
            experiments: Set of experiment IDs to run (``"all"`` expands to all).
            reset_summary: Re-initialise CSV artefacts before this campaign.
            calibrate_smoke: Run short smoke trials to pick the largest stable
                batch size per experiment before full training.
            calibrate_only: Exit after calibration without running experiments.
            smoke_max_steps: Step budget per smoke calibration trial.
        """
        selected = self._normalize_experiment_selection(experiments)
        for path in self._required_paths_for_experiments(selected):
            self.require_path(path)

        if (
            self.cfg.runtime.train_distributed
            and self.cfg.runtime.train_num_processes > 1
        ):
            self.require_python_module("accelerate")

        if reset_summary or not self.cfg.summary_csv.exists():
            self.init_summary()
        if reset_summary or not self.cfg.manifest_csv.exists():
            self.init_manifest()
        if calibrate_smoke and (
            reset_summary or not self.cfg.batch_calibration_csv.exists()
        ):
            self.init_batch_calibration()

        # Ignore old calibration rows when calibration is disabled; use fixed
        # fallback batches to keep runs predictable and fast to start.
        calibrated_batches = self.load_calibrated_batches() if calibrate_smoke else {}

        self.log(
            "Campaign mode: "
            f"experiments={','.join(sorted(selected))}, "
            f"calibrate_smoke={int(calibrate_smoke)}, calibrate_only={int(calibrate_only)}"
        )

        # Each entry holds everything needed to build a RunRecord once the
        # concurrent benchmark future resolves.
        # (preset, batch_size, model_path, run_name, retries, status, bench_future)
        _PendingType = tuple[ExperimentPreset, int, Path, str, int, str, "Future[Path]"]
        pending: list[_PendingType] = []

        presets = self._experiment_presets()

        with ThreadPoolExecutor(max_workers=1) as bench_exec:
            # Submit baseline benchmark immediately so it overlaps with training.
            baseline_knn_future: Future[Path] | None = None
            if not calibrate_only:
                baseline_knn_future = bench_exec.submit(
                    self.run_knn_benchmark,
                    "baseline",
                    self.cfg.base_model,
                    self.cfg.benchmark.force_baseline_benchmark_rerun,
                )

            for experiment_id in EXPERIMENT_IDS:
                if (
                    experiment_id == "baseline_eval_only"
                    or experiment_id not in selected
                ):
                    continue
                preset = presets[experiment_id]

                if calibrate_smoke:
                    calibrated_batch = calibrated_batches.get(
                        experiment_id, self._default_batch_for_preset(preset)
                    )
                else:
                    calibrated_batch = self._default_batch_for_preset(preset)

                if experiment_id in calibrated_batches and calibrate_smoke:
                    self.log(
                        f"Using calibrated batch for {experiment_id}: {calibrated_batch}"
                    )
                elif not calibrate_smoke:
                    self.log(
                        f"Using fixed fallback batch for {experiment_id}: {calibrated_batch}"
                    )
                if calibrate_smoke:
                    calibrated_batch = self._calibrate_experiment_batch(
                        preset=preset,
                        smoke_max_steps=smoke_max_steps,
                    )
                    calibrated_batches[experiment_id] = calibrated_batch
                    self.log(
                        f"Calibration selected batch for {experiment_id}: {calibrated_batch}"
                    )
                if calibrate_only:
                    continue

                # Train in the main thread (blocks until done).
                model_path, run_name, retries, status = self._train_single_experiment(
                    preset=preset,
                    batch_size=calibrated_batch,
                    max_steps=0,
                    max_minutes=self.cfg.train.max_minutes,
                )
                # Dispatch benchmark to pool thread (runs concurrently with next train).
                bench_future: Future[Path] = bench_exec.submit(
                    self.run_knn_benchmark, preset.stage_name, str(model_path)
                )
                pending.append(
                    (
                        preset,
                        calibrated_batch,
                        model_path,
                        run_name,
                        retries,
                        status,
                        bench_future,
                    )
                )

            # --- Collect results in canonical order and write CSVs ---

            baseline_knn_csv: Path | None = None
            if not calibrate_only and baseline_knn_future is not None:
                baseline_knn_csv = baseline_knn_future.result()
                self.append_summary(
                    experiment_id="baseline_eval_only",
                    stage="baseline",
                    model_path=self.cfg.base_model,
                    knn_csv=baseline_knn_csv,
                    loss_mode="benchmark_only",
                    multi_primary_loss="",
                    train_files=(),
                    dms_enabled=False,
                    batch_size=self.cfg.train.batch_size,
                    grad_accum=self.cfg.train.grad_accum,
                    mnrl_mini_batch=self.cfg.train.mnrl_mini_batch,
                    max_minutes=0,
                )
                self.append_manifest(
                    RunRecord(
                        experiment_id="baseline_eval_only",
                        stage="baseline",
                        run_name=f"{self.cfg.run_prefix}_baseline_eval_only",
                        model_path=Path(self.cfg.base_model),
                        knn_csv=baseline_knn_csv,
                        loss_mode="benchmark_only",
                        multi_primary_loss="",
                        files=(),
                        batch_size=self.cfg.train.batch_size,
                        grad_accum=self.cfg.train.grad_accum,
                        mnrl_mini_batch=self.cfg.train.mnrl_mini_batch,
                        max_minutes=0,
                        retries=0,
                        status="benchmarked",
                    )
                )

            for (
                preset,
                batch_size,
                model_path,
                run_name,
                retries,
                status,
                bench_future,
            ) in pending:
                knn_csv = (
                    bench_future.result()
                )  # propagates exceptions from bench thread
                record = RunRecord(
                    experiment_id=preset.experiment_id,
                    stage=preset.stage_name,
                    run_name=run_name,
                    model_path=model_path,
                    knn_csv=knn_csv,
                    loss_mode=preset.loss_mode,
                    multi_primary_loss=preset.multi_primary_loss or "",
                    files=preset.files,
                    batch_size=batch_size,
                    grad_accum=self.cfg.train.grad_accum,
                    mnrl_mini_batch=self._effective_mnrl_mini_batch_for_preset(preset),
                    max_minutes=self.cfg.train.max_minutes,
                    retries=retries,
                    status=status,
                )
                self.append_summary(
                    experiment_id=record.experiment_id,
                    stage=record.stage,
                    model_path=str(record.model_path),
                    knn_csv=record.knn_csv,
                    loss_mode=record.loss_mode,
                    multi_primary_loss=record.multi_primary_loss,
                    train_files=record.files,
                    dms_enabled=preset.dms_enabled,
                    batch_size=record.batch_size,
                    grad_accum=record.grad_accum,
                    mnrl_mini_batch=record.mnrl_mini_batch,
                    max_minutes=record.max_minutes,
                )
                self.append_manifest(record)

        selected_nonbaseline = any(
            experiment_id != "baseline_eval_only" for experiment_id in selected
        )
        if not calibrate_only and baseline_knn_csv is not None and selected_nonbaseline:
            delta_path = self.write_delta_report()
            if delta_path is not None:
                self.log(f"Baseline deltas written to: {delta_path}")
            else:
                self.log("Skipping delta report: no comparable non-baseline rows found")
        elif not calibrate_only:
            self.log("Skipping delta report: no non-baseline experiments were selected")

        self.log(
            "Campaign finished. Artifacts: "
            f"summary={self.cfg.summary_csv}, manifest={self.cfg.manifest_csv}, "
            f"delta={self.cfg.delta_csv}, calibration={self.cfg.batch_calibration_csv}"
        )

    def run(
        self,
        stages: set[str] | None = None,
        reset_summary: bool = True,
        experiments: set[str] | None = None,
        calibrate_smoke: bool = False,
        calibrate_only: bool = False,
        smoke_max_steps: int = 2,
    ) -> None:
        if experiments is not None:
            self.run_experiment_campaign(
                experiments=experiments,
                reset_summary=reset_summary,
                calibrate_smoke=calibrate_smoke,
                calibrate_only=calibrate_only,
                smoke_max_steps=smoke_max_steps,
            )
            return

        selected = self._normalize_stage_selection(stages)

        for path in self._required_paths_for_selection(selected):
            self.require_path(path)

        if (
            self.cfg.runtime.train_distributed
            and self.cfg.runtime.train_num_processes > 1
        ):
            self.require_python_module("accelerate")

        if selected & {"baseline", "stage1", "stage2a", "stage2b"}:
            if reset_summary or not self.cfg.summary_csv.exists():
                self.init_summary()

        log_lines = [
            f"RUN_PREFIX={self.cfg.run_prefix}",
            f"Selected stages: {', '.join(sorted(selected))}",
            f"TRAIN_DISTRIBUTED={int(self.cfg.runtime.train_distributed)}, "
            f"TRAIN_NUM_PROCESSES={self.cfg.runtime.train_num_processes}, "
            f"TRAIN_MIXED_PRECISION={self.cfg.runtime.train_mixed_precision}",
            f"BENCH_FAST={int(self.cfg.benchmark.bench_fast)}, "
            f"CACHE_EMBEDDINGS={int(self.cfg.benchmark.cache_embeddings)}",
            "Progress bars: "
            f"PROTEIN_PROGRESS_BARS={os.environ.get('PROTEIN_PROGRESS_BARS', 'auto')}, "
            f"PROTEIN_PROGRESS_MIN_INTERVAL={self.cfg.runtime.progress_min_interval:.1f}s",
            "OOM retry policy: "
            f"enabled={int(self.cfg.oom_retry.oom_retry_enabled)}, "
            f"max_attempts={self.cfg.oom_retry.oom_retry_max_attempts}, "
            f"min_batch={self.cfg.oom_retry.oom_retry_min_batch}",
            f"Train knobs: max_minutes={self.cfg.train.max_minutes}, "
            f"max_map_rows={self.cfg.train.max_map_rows}, "
            f"dms_max_rows={self.cfg.train.dms_max_rows}, "
            f"learning_rate={self.cfg.train.learning_rate}, warmup_steps={self.cfg.train.warmup_steps}",
            f"Batch knobs: batch_size={self.cfg.train.batch_size}, grad_accum={self.cfg.train.grad_accum}, "
            f"mnrl_mini_batch={self.cfg.train.mnrl_mini_batch}",
            f"Pooling: {self.cfg.train.pooling_mode}",
            f"Compile training: {int(self.cfg.train.compile_training)}",
            f"Run stage1: {int(self.cfg.stages.run_stage1)}, "
            f"evaluate stage1: {int(self.cfg.stages.evaluate_stage1)}",
            (
                f"Stage2 init model override: {self.cfg.stages.stage2_init_model}"
                if self.cfg.stages.stage2_init_model
                else "Stage2 init source: "
                f"{'stage1' if self.cfg.stages.stage2_use_stage1_init else 'base'}"
            ),
            "Data sampler: round_robin",
            f"Force baseline benchmark rerun: "
            f"{int(self.cfg.benchmark.force_baseline_benchmark_rerun)}",
            "Stage datasets: stage1=[pfam,afdb], "
            "stage2=[pfam,afdb,stringdb], stage2b_plus=[dms_cosent]",
            "Stage order: baseline(knn) -> [optional stage1_triplet] -> "
            "stage2a_cached_mnrl -> stage2b_multi_dms",
        ]
        for line in log_lines:
            self.log(line)

        if "baseline" in selected:
            self.evaluate_stage(
                "baseline",
                self.cfg.base_model,
                force_rerun=self.cfg.benchmark.force_baseline_benchmark_rerun,
                experiment_id="baseline",
            )

        if "stage1" in selected:
            if self.cfg.stages.run_stage1:
                self.run_stage1()
            else:
                self.log("Skipping Stage1 training (RUN_STAGE1=0)")

            if self.cfg.stages.evaluate_stage1:
                if self.cfg.stages.stage1_model_path.exists():
                    self.evaluate_stage(
                        "stage1_triplet",
                        str(self.cfg.stages.stage1_model_path),
                        experiment_id="stage1_triplet",
                        loss_mode="triplet",
                        train_files=(
                            self.cfg.paths.pfam_file,
                            self.cfg.paths.afdb_file,
                        ),
                    )
                    readable_path = self.write_baseline_vs_stage1_readable_report()
                    if readable_path is not None:
                        self.log(f"Stage1 readable report written to: {readable_path}")
                else:
                    self.log(
                        "Skipping Stage1 benchmark: no Stage1 model found at "
                        f"{self.cfg.stages.stage1_model_path}"
                    )
            else:
                self.log("Skipping Stage1 benchmark (EVALUATE_STAGE1=0)")

        stage2_specs = (
            (
                "stage2a",
                "stage2a_cached_mnrl",
                self.cfg.stages.stage2a_run_name,
                self.cfg.stages.stage2a_model_path,
                "cached_mnrl",
            ),
            (
                "stage2b",
                "stage2b_multi_dms",
                self.cfg.stages.stage2b_run_name,
                self.cfg.stages.stage2b_model_path,
                "multi",
            ),
        )
        for stage_key, stage_name, run_name, model_path, loss_mode in stage2_specs:
            if stage_key not in selected:
                continue
            self.run_stage2(stage_name, run_name, model_path, loss_mode)
            self.evaluate_stage(
                stage_name,
                str(model_path),
                experiment_id=stage_name,
                loss_mode=loss_mode,
                multi_primary_loss="cached_mnrl" if loss_mode == "multi" else "",
                train_files=(
                    self.cfg.paths.hard_neg_parquet,
                    self.cfg.paths.afdb_file,
                    self.cfg.paths.stringdb_file,
                ),
                dms_enabled=loss_mode == "multi",
            )

        if "report" in selected:
            delta_path = self.write_delta_report()
            self.log(f"Baseline deltas written to: {delta_path}")
        self.log("Finished staged ablation pipeline")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stage selection and output behavior."""
    parser = argparse.ArgumentParser(description="Run stage-wise ablation pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=CLI_STAGE_CHOICES,
        help=(
            "Stages to execute. "
            "Choices: baseline, stage1, stage2a, stage2b, report, all"
        ),
    )
    parser.add_argument(
        "--reset-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset summary.csv and related CSV artifacts before appending outputs.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=CLI_EXPERIMENT_CHOICES,
        default=None,
        help=(
            "Experiment presets to execute independently. "
            "Choices: "
            + ", ".join(CLI_EXPERIMENT_CHOICES)
            + ". When set, the experiment campaign flow is used instead of --stages."
        ),
    )
    parser.add_argument(
        "--calibrate-batch-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run 1-2 step smoke calibration to pick max stable batch per experiment.",
    )
    parser.add_argument(
        "--calibrate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run calibration only without full training/benchmark experiments.",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=4,
        help="Maximum steps for each calibration smoke trial (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    cfg = RunConfig.from_env(repo_root)
    runner = Runner(cfg)
    runner.run(
        stages=set(args.stages),
        reset_summary=bool(args.reset_summary),
        experiments=set(args.experiments) if args.experiments is not None else None,
        calibrate_smoke=bool(args.calibrate_batch_smoke),
        calibrate_only=bool(args.calibrate_only),
        smoke_max_steps=max(1, int(args.smoke_max_steps)),
    )


if __name__ == "__main__":
    main()
