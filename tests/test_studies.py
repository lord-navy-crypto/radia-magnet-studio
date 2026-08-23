from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.cli import build_tasks, summarize
from studies.core import (
    convergence_report,
    convergence_tasks,
    monte_carlo_tasks,
    parameter_scan_tasks,
    parameter_sensitivities,
    results_bundle_bytes,
    run_tasks,
    summarize_monte_carlo,
    sensitivity_svg_bytes,
)
from tests.fake_study_worker import evaluate, fail_for_large_gap


BASE = {"device": "Planar", "gap_mm": 10.0, "error_seed": 0}
SETTINGS = {"axis_samples": 100, "field_margin_periods": 1.0}


def test_task_generation_and_statistics():
    scan = parameter_scan_tasks(BASE, SETTINGS, {"gap_mm": [10, 12], "br_t": [1.1, 1.2]})
    assert len(scan) == 4
    assert len({task["task_id"] for task in scan}) == 4
    mc = monte_carlo_tasks(BASE, SETTINGS, samples=4, seed_start=10)
    assert [task["labels"]["error_seed"] for task in mc] == [10, 11, 12, 13]
    results = [evaluate(task) for task in mc]
    stats = summarize_monte_carlo(results, ["K_peak"])["K_peak"]
    assert stats["n"] == 4 and stats["std"] > 0
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_checkpoint_cancel_resume_and_cache():
    tasks = parameter_scan_tasks(BASE, SETTINGS, {"gap_mm": [10, 11, 12]})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calls = {"n": 0}

        def stop_after_one(done, total, result):
            if result is not None:
                calls["n"] += 1

        first = run_tasks(
            tasks, evaluate, checkpoint_path=root / "checkpoint.json",
            cache_directory=root / "cache", max_workers=1,
            cancel_requested=lambda: calls["n"] >= 1,
            progress=stop_after_one,
        )
        assert first["status"] == "cancelled" and first["completed"] == 1
        resumed = run_tasks(
            tasks, evaluate, checkpoint_path=root / "checkpoint.json",
            cache_directory=root / "cache", max_workers=1,
        )
        assert resumed["status"] == "complete" and resumed["completed"] == 3
        (root / "checkpoint.json").unlink()
        cached = run_tasks(
            tasks, evaluate, checkpoint_path=root / "checkpoint.json",
            cache_directory=root / "cache", max_workers=1,
        )
        assert cached["cache_hits"] == 3


def test_parallel_failures_convergence_and_bundle():
    scan = parameter_scan_tasks(BASE, SETTINGS, {"gap_mm": [10, 12]})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = run_tasks(
            scan, fail_for_large_gap, checkpoint_path=root / "checkpoint.json",
            cache_directory=root / "cache", max_workers=2,
        )
        assert report["status"] == "complete"
        assert report["failures"] == 1

    tasks = convergence_tasks(
        BASE, SETTINGS, segmentations=[1, 2], axis_samples=[100, 200],
        margin_periods=[0.5, 1.0],
    )
    results = [evaluate(task) for task in tasks]
    conv = convergence_report(results, ["K_peak"])
    reference = next(r for r in results if r["task_id"] == conv["reference_task_id"])
    assert reference["labels"] == {
        "segmentation": 2, "axis_samples": 200, "field_margin_periods": 1.0,
    }
    assert all("all_metrics_converged" in row for row in conv["rows"])
    bundle = results_bundle_bytes({"results": results}, summary=conv)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {
            "study_report.json", "study_results.csv", "study_summary.json",
        }
        json.loads(archive.read("study_summary.json"))


def test_cli_config_dispatch_and_summary():
    config = {
        "study_type": "parameter_scan", "base_parameters": BASE,
        "settings": SETTINGS, "grid": {"gap_mm": [10, 12]},
        "objective": "K_peak", "goal": "maximize",
    }
    tasks = build_tasks(config)
    report = {"results": [evaluate(task) for task in tasks]}
    best = summarize(config, report)["best"]
    assert best["parameters"]["gap_mm"] == 12
    sensitivity = parameter_sensitivities(report["results"], "K_peak")
    assert [point["parameter_value"] for point in sensitivity["gap_mm"]] == [10.0, 12.0]
    assert sensitivity_svg_bytes("gap_mm", "K_peak", sensitivity["gap_mm"]).startswith(b"<svg")


if __name__ == "__main__":
    test_task_generation_and_statistics()
    test_checkpoint_cancel_resume_and_cache()
    test_parallel_failures_convergence_and_bundle()
    test_cli_config_dispatch_and_summary()
    print("studies tests passed")
