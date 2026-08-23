def evaluate(task):
    labels = task.get("labels", {})
    seed = int(labels.get("error_seed", 0))
    gap = float(task["parameters"].get("gap_mm", 0.0))
    segmentation = int(labels.get("segmentation", 1))
    samples = int(labels.get("axis_samples", 100))
    margin = float(labels.get("field_margin_periods", 1.0))
    error = 1.0 / (segmentation * samples * max(margin, 0.1))
    return {
        "task_id": task["task_id"],
        "study_type": task["study_type"],
        "labels": labels,
        "parameters": task["parameters"],
        "settings": task["settings"],
        "metrics": {
            "K_peak": 2.0 * gap + seed * 0.01 + error,
            "Bperp_peak_T": gap + seed * 0.001,
        },
        "elapsed_s": 0.001,
        "status": "success",
    }


def fail_for_large_gap(task):
    if float(task["parameters"].get("gap_mm", 0)) > 11:
        raise RuntimeError("synthetic worker failure")
    return evaluate(task)

