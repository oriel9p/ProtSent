"""Unit tests for ablation runner safety, reuse, and CSV reporting behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from run_ablation_v2 import (
    BenchmarkConfig,
    DataPaths,
    OomRetryConfig,
    RunRecord,
    RunConfig,
    Runner,
    RuntimeConfig,
    StageFlowConfig,
    TrainConfig,
)


def _make_config(
    tmp_path: Path,
    *,
    skip_existing_benchmarks: bool = True,
) -> RunConfig:
    result_root = tmp_path / "results"
    log_root = tmp_path / "logs"
    models_root = tmp_path / "models"
    stage1_model_path = models_root / "stage1" / "final"
    stage2a_model_path = models_root / "stage2a" / "final"
    stage2b_model_path = models_root / "stage2b" / "final"

    return RunConfig(
        repo_root=tmp_path,
        base_model="facebook/esm2_t12_35M_UR50D",
        run_prefix="test_run",
        result_root=result_root,
        log_root=log_root,
        main_log=log_root / "main.log",
        summary_csv=result_root / "summary.csv",
        delta_csv=result_root / "delta.csv",
        manifest_csv=result_root / "manifest.csv",
        batch_calibration_csv=result_root / "batch_calibration.csv",
        paths=DataPaths(
            pfam_file=tmp_path / "pfam.parquet",
            hard_neg_parquet=tmp_path / "pfam_hardneg.parquet",
            afdb_file=tmp_path / "afdb.parquet",
            stringdb_file=tmp_path / "stringdb.parquet",
            dms_cosent_file=tmp_path / "dms.parquet",
        ),
        runtime=RuntimeConfig(
            python_bin="python3",
            train_distributed=True,
            train_num_processes=2,
            train_mixed_precision="bf16",
            progress_min_interval=1.0,
        ),
        train=TrainConfig(
            pooling_mode="mean",
            optimizer="adamw_torch_fused",
            compile_training=False,
            save_steps=9999,
            max_minutes=1,
            max_map_rows=100,
            dms_max_rows=0,
            learning_rate=8e-5,
            warmup_steps=10,
            batch_size=32,
            grad_accum=1,
            mnrl_mini_batch=32,
        ),
        benchmark=BenchmarkConfig(
            bench_fast=True,
            cache_embeddings=True,
            skip_existing_benchmarks=skip_existing_benchmarks,
            force_baseline_benchmark_rerun=True,
        ),
        oom_retry=OomRetryConfig(
            oom_retry_enabled=True,
            oom_retry_max_attempts=1,
            oom_retry_min_batch=2,
        ),
        stages=StageFlowConfig(
            run_stage1=False,
            evaluate_stage1=False,
            stage2_use_stage1_init=False,
            stage2_init_model="",
            stage1_run_name="stage1",
            stage2a_run_name="stage2a",
            stage2b_run_name="stage2b",
            stage1_model_path=stage1_model_path,
            stage2a_model_path=stage2a_model_path,
            stage2b_model_path=stage2b_model_path,
        ),
    )


def test_log_has_oom_respects_start_offset(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    log_file.write_text("cuda out of memory\n", encoding="utf-8")
    start_offset = log_file.stat().st_size
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("normal training line\n")

    assert Runner._log_has_oom(log_file)
    assert not Runner._log_has_oom(log_file, start_offset=start_offset)


def test_run_knn_benchmark_reuses_only_valid_existing_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    output_root = cfg.result_root / "baseline" / "knn"
    output_root.mkdir(parents=True, exist_ok=True)
    valid_csv = output_root / "bench_existing.csv"
    valid_csv.write_text("Task,Spearman\nfoo,0.1\n", encoding="utf-8")

    def _unexpected_run_cmd(
        cmd: list[str],
        log_file: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        raise AssertionError(f"run_cmd should not execute when CSV is valid: {cmd}")

    monkeypatch.setattr(runner, "run_cmd", _unexpected_run_cmd)

    result = runner.run_knn_benchmark("baseline", cfg.base_model)

    assert result == valid_csv


def test_run_knn_benchmark_reruns_when_existing_csv_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    output_root = cfg.result_root / "baseline" / "knn"
    output_root.mkdir(parents=True, exist_ok=True)
    invalid_csv = output_root / "bench_existing.csv"
    invalid_csv.write_text("Task,Spearman\n", encoding="utf-8")

    run_calls: list[list[str]] = []

    def _fake_run_cmd(
        cmd: list[str],
        log_file: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        run_calls.append(cmd)
        fresh_csv = output_root / "bench_fresh.csv"
        fresh_csv.write_text("Task,Spearman\nfoo,0.2\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_cmd", _fake_run_cmd)

    result = runner.run_knn_benchmark("baseline", cfg.base_model)

    assert len(run_calls) == 1
    cmd = run_calls[0]
    assert cmd[0] == cfg.runtime.python_bin
    assert cmd[1] == "protein_benchmark_suite.py"
    assert "--eval_split" in cmd
    assert cmd[cmd.index("--eval_split") + 1] == "validation"
    assert "--probe_type" in cmd
    assert "knn" in cmd
    assert "--tasks" in cmd
    assert "solubility" in cmd
    assert result.name == "bench_fresh.csv"


def test_run_training_retries_with_reduced_batch_on_oom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    log_file = cfg.log_root / "stage2_train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")

    seen_cmds: list[list[str]] = []

    def _fail_then_succeed(
        cmd: list[str],
        log_file: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        seen_cmds.append(cmd)
        with log_file.open("a", encoding="utf-8") as handle:
            if len(seen_cmds) == 1:
                handle.write("CUDA out of memory\n")
                raise RuntimeError("simulated training failure")
            handle.write("ok\n")

    monkeypatch.setattr(runner, "run_cmd", _fail_then_succeed)

    retry_count = runner.run_training(
        [
            "--batch_size",
            "16",
            "--gradient_accumulation_steps",
            "2",
        ],
        log_file=log_file,
    )

    assert retry_count == 1
    assert len(seen_cmds) == 2
    assert seen_cmds[0][:3] == ["python3", "-m", "accelerate.commands.launch"]
    assert seen_cmds[1][:3] == ["python3", "-m", "accelerate.commands.launch"]
    assert "--batch_size" in seen_cmds[1]
    assert seen_cmds[1][seen_cmds[1].index("--batch_size") + 1] == "8"
    assert "--gradient_accumulation_steps" in seen_cmds[1]
    assert seen_cmds[1][seen_cmds[1].index("--gradient_accumulation_steps") + 1] == "4"


def test_append_summary_writes_csv_with_desc(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    knn_csv = tmp_path / "bench.csv"
    knn_csv.write_text(
        "Task,Spearman,Accuracy,F1,AUC\nfoo,0.2,0.8,0.7,0.75\n", encoding="utf-8"
    )

    runner.init_summary()
    runner.append_summary(
        experiment_id="mnrl_cosent_multi",
        stage="mnrl_cosent_multi",
        model_path="models/test/final",
        knn_csv=knn_csv,
        loss_mode="multi",
        multi_primary_loss="mnrl",
        train_files=(cfg.paths.pfam_file, cfg.paths.afdb_file),
        dms_enabled=True,
        batch_size=64,
        grad_accum=2,
        mnrl_mini_batch=32,
        max_minutes=30,
    )

    csv_text = cfg.summary_csv.read_text(encoding="utf-8")
    assert "Desc" in csv_text.splitlines()[0]
    assert "mnrl_cosent_multi" in csv_text
    assert "primary=mnrl" in csv_text


def test_normalize_experiment_selection_all() -> None:
    selected = Runner._normalize_experiment_selection({"all"})

    assert "baseline_eval_only" in selected
    assert "triplet_cosent_multi" in selected
    assert "gist_cosent_multi" in selected


def test_load_calibrated_batches_uses_highest_successful_choice(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    runner.init_batch_calibration()
    runner.append_batch_calibration(
        experiment_id="mnrl_cosent_multi",
        tested_batch=256,
        success=True,
        retries=0,
        chosen_batch=256,
    )
    runner.append_batch_calibration(
        experiment_id="mnrl_cosent_multi",
        tested_batch=384,
        success=False,
        retries=1,
        chosen_batch=256,
    )
    runner.append_batch_calibration(
        experiment_id="gist_cosent_multi",
        tested_batch=192,
        success=True,
        retries=0,
        chosen_batch=192,
    )

    calibrated = runner.load_calibrated_batches()

    assert calibrated == {
        "mnrl_cosent_multi": 256,
        "gist_cosent_multi": 192,
    }


def test_run_experiment_campaign_uses_saved_calibration_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    for path in (
        cfg.paths.pfam_file,
        cfg.paths.afdb_file,
        cfg.paths.stringdb_file,
        cfg.paths.dms_cosent_file,
    ):
        path.write_text("x", encoding="utf-8")

    runner.init_batch_calibration()
    runner.append_batch_calibration(
        experiment_id="mnrl_cosent_multi",
        tested_batch=384,
        success=True,
        retries=0,
        chosen_batch=384,
    )

    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "Task,Spearman,Accuracy,F1,AUC\nfoo,0.2,0.8,0.7,0.75\n",
        encoding="utf-8",
    )

    captured_batches: list[int] = []

    def _fake_run_knn_benchmark(
        stage: str,
        model_path: str,
        force_rerun: bool = False,
    ) -> Path:
        return bench_csv

    def _fake_run_single_experiment(
        *,
        preset: object,
        batch_size: int,
        max_steps: int,
        max_minutes: int,
    ) -> RunRecord:
        captured_batches.append(batch_size)
        return RunRecord(
            experiment_id="mnrl_cosent_multi",
            stage="mnrl_cosent_multi",
            run_name="test_run_mnrl_cosent_multi",
            model_path=tmp_path / "models" / "mnrl" / "final",
            knn_csv=bench_csv,
            loss_mode="multi",
            multi_primary_loss="mnrl",
            files=(cfg.paths.pfam_file, cfg.paths.afdb_file, cfg.paths.stringdb_file),
            batch_size=batch_size,
            grad_accum=cfg.train.grad_accum,
            mnrl_mini_batch=cfg.train.mnrl_mini_batch,
            max_minutes=max_minutes,
            retries=0,
            status="trained",
        )

    monkeypatch.setattr(runner, "run_knn_benchmark", _fake_run_knn_benchmark)
    monkeypatch.setattr(runner, "_run_single_experiment", _fake_run_single_experiment)
    monkeypatch.setattr(runner, "write_delta_report", lambda: cfg.delta_csv)

    runner.run_experiment_campaign(
        experiments={"mnrl_cosent_multi"},
        reset_summary=True,
        calibrate_smoke=False,
        calibrate_only=False,
        smoke_max_steps=2,
    )

    assert captured_batches == [384]


def test_build_experiment_train_args_uses_base_model_for_campaign_runs(
    tmp_path: Path,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    preset = runner._experiment_presets()["mnrl_cosent_multi"]
    args = runner._build_experiment_train_args(
        preset=preset,
        batch_size=cfg.train.batch_size,
        run_name="campaign_test",
        max_steps=2,
        max_minutes=cfg.train.max_minutes,
    )

    model_index = args.index("--model") + 1

    assert args[model_index] == cfg.base_model


def test_build_experiment_train_args_includes_dms_max_rows_when_configured(
    tmp_path: Path,
) -> None:
    base_cfg = _make_config(tmp_path)
    cfg = RunConfig(
        repo_root=base_cfg.repo_root,
        base_model=base_cfg.base_model,
        run_prefix=base_cfg.run_prefix,
        result_root=base_cfg.result_root,
        log_root=base_cfg.log_root,
        main_log=base_cfg.main_log,
        summary_csv=base_cfg.summary_csv,
        delta_csv=base_cfg.delta_csv,
        manifest_csv=base_cfg.manifest_csv,
        batch_calibration_csv=base_cfg.batch_calibration_csv,
        paths=base_cfg.paths,
        runtime=base_cfg.runtime,
        train=TrainConfig(
            pooling_mode=base_cfg.train.pooling_mode,
            optimizer=base_cfg.train.optimizer,
            compile_training=base_cfg.train.compile_training,
            save_steps=base_cfg.train.save_steps,
            max_minutes=base_cfg.train.max_minutes,
            max_map_rows=base_cfg.train.max_map_rows,
            dms_max_rows=1234,
            learning_rate=base_cfg.train.learning_rate,
            warmup_steps=base_cfg.train.warmup_steps,
            batch_size=base_cfg.train.batch_size,
            grad_accum=base_cfg.train.grad_accum,
            mnrl_mini_batch=base_cfg.train.mnrl_mini_batch,
        ),
        benchmark=base_cfg.benchmark,
        oom_retry=base_cfg.oom_retry,
        stages=base_cfg.stages,
    )
    runner = Runner(cfg)

    preset = runner._experiment_presets()["mnrl_cosent_multi"]
    args = runner._build_experiment_train_args(
        preset=preset,
        batch_size=cfg.train.batch_size,
        run_name="campaign_test",
        max_steps=2,
        max_minutes=cfg.train.max_minutes,
    )

    assert "--dms_max_rows" in args
    assert args[args.index("--dms_max_rows") + 1] == "1234"


def test_write_delta_report_uses_latest_baseline_row(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    baseline_old_csv = tmp_path / "baseline_old.csv"
    baseline_new_csv = tmp_path / "baseline_new.csv"
    experiment_csv = tmp_path / "experiment.csv"

    baseline_old_csv.write_text("Task,Spearman\nfoo,0.10\n", encoding="utf-8")
    baseline_new_csv.write_text("Task,Spearman\nfoo,0.30\n", encoding="utf-8")
    experiment_csv.write_text("Task,Spearman\nfoo,0.40\n", encoding="utf-8")

    runner.init_summary()
    runner.append_summary(
        experiment_id="baseline_eval_only",
        stage="baseline",
        model_path=cfg.base_model,
        knn_csv=baseline_old_csv,
        loss_mode="benchmark_only",
        multi_primary_loss="",
        train_files=(),
        dms_enabled=False,
        batch_size=cfg.train.batch_size,
        grad_accum=cfg.train.grad_accum,
        mnrl_mini_batch=cfg.train.mnrl_mini_batch,
        max_minutes=0,
    )
    runner.append_summary(
        experiment_id="baseline_eval_only",
        stage="baseline",
        model_path=cfg.base_model,
        knn_csv=baseline_new_csv,
        loss_mode="benchmark_only",
        multi_primary_loss="",
        train_files=(),
        dms_enabled=False,
        batch_size=cfg.train.batch_size,
        grad_accum=cfg.train.grad_accum,
        mnrl_mini_batch=cfg.train.mnrl_mini_batch,
        max_minutes=0,
    )
    runner.append_summary(
        experiment_id="mnrl_cosent_multi",
        stage="mnrl_cosent_multi",
        model_path="models/test/final",
        knn_csv=experiment_csv,
        loss_mode="multi",
        multi_primary_loss="mnrl",
        train_files=(cfg.paths.pfam_file, cfg.paths.afdb_file, cfg.paths.stringdb_file),
        dms_enabled=True,
        batch_size=384,
        grad_accum=cfg.train.grad_accum,
        mnrl_mini_batch=cfg.train.mnrl_mini_batch,
        max_minutes=cfg.train.max_minutes,
    )

    delta_path = runner.write_delta_report()

    assert delta_path == cfg.delta_csv
    delta_text = cfg.delta_csv.read_text(encoding="utf-8")
    assert "+0.10000" in delta_text


def test_run_experiment_campaign_baseline_only_skips_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    runner = Runner(cfg)

    bench_csv = tmp_path / "baseline_bench.csv"
    bench_csv.write_text("Task,Spearman\nfoo,0.1\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_knn_benchmark", lambda *args, **kwargs: bench_csv)
    monkeypatch.setattr(
        runner,
        "write_delta_report",
        lambda: (_ for _ in ()).throw(AssertionError("delta report should not run")),
    )

    runner.run_experiment_campaign(
        experiments={"baseline_eval_only"},
        reset_summary=True,
        calibrate_smoke=False,
        calibrate_only=False,
        smoke_max_steps=2,
    )

    assert cfg.summary_csv.exists()
    assert not cfg.delta_csv.exists()
