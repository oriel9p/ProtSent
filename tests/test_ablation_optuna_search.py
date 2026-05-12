"""Unit tests for Optuna ablation search scoring helpers."""

from __future__ import annotations

from pathlib import Path

from ablation_optuna_search import (
    compute_trial_score,
    experiment_uses_dms,
    select_primary_task_delta,
)


def test_select_primary_task_delta_prefers_auc_over_other_metrics() -> None:
    row = {
        "Task": "demo",
        "delta_AUC": "0.02",
        "baseline_AUC": "0.8",
        "delta_Accuracy": "0.10",
        "baseline_Accuracy": "0.5",
    }

    selected = select_primary_task_delta(row, ["AUC", "Accuracy", "Spearman", "F1"])

    assert selected is not None
    assert selected.metric == "AUC"
    assert round(selected.pct_delta, 6) == 2.5


def test_compute_trial_score_aggregates_wins_losses_and_noise(tmp_path: Path) -> None:
    delta_csv = tmp_path / "delta.csv"
    summary_csv = tmp_path / "summary.csv"

    delta_csv.write_text(
        "\n".join(
            [
                "stage,Task,delta_AUC,baseline_AUC,delta_Accuracy,baseline_Accuracy,delta_Spearman,baseline_Spearman,delta_F1,baseline_F1",
                "mnrl_cosent_multi,t1,0.02,0.80,,,,,,",
                "mnrl_cosent_multi,t2,-0.01,0.50,,,,,,",
                "mnrl_cosent_multi,t3,0.0,0.9,,,,,,",
                "other_stage,tx,0.50,1.00,,,,,,",
            ]
        ),
        encoding="utf-8",
    )

    summary_csv.write_text(
        "\n".join(
            [
                "run_prefix,experiment_id,knn_spearman_mean,knn_accuracy_mean,knn_f1_mean,knn_auc_mean",
                "r,baseline_eval_only,0.50,0.70,0.72,0.74",
                "r,mnrl_cosent_multi,0.55,0.71,0.73,0.75",
            ]
        ),
        encoding="utf-8",
    )

    score = compute_trial_score(
        delta_csv=delta_csv,
        summary_csv=summary_csv,
        experiment_id="mnrl_cosent_multi",
        metric_priority=["AUC", "Accuracy", "Spearman", "F1"],
        tie_epsilon=1e-3,
        noise_penalty=0.10,
    )

    assert score.n_tasks == 3
    assert score.wins == 1
    assert score.losses == 1
    assert score.ties == 1
    assert score.std_pct_delta > 0.0
    assert score.objective < score.mean_pct_delta
    assert score.summary_mean_pct_delta > 0.0


def test_experiment_uses_dms_for_cached_mnrl_no_dms() -> None:
    assert not experiment_uses_dms("cached_mnrl_no_dms")
    assert experiment_uses_dms("mnrl_cosent_multi")
