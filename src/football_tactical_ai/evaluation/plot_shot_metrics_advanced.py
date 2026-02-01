import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = Path("src/football_tactical_ai/evaluation/plots_advanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)


EVAL_FILES = {
    "WEAK": "src/football_tactical_ai/evaluation/results/logs/shot_weak/shot_weak_evaluation.json",
    "NORMAL": "src/football_tactical_ai/evaluation/results/logs/shot/shot_evaluation.json",
    "STRONG": "src/football_tactical_ai/evaluation/results/logs/shot_strong/shot_strong_evaluation.json",
}


def load_eval(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(eval_data: dict, metric_name: str):
    """Return a flat list across all test cases for a metric in shot_metrics."""
    vals = []
    for case_name, case in eval_data.items():
        shot_metrics = case.get("shot_metrics", [])
        for m in shot_metrics:
            if m is None:
                continue
            v = m.get(metric_name, None)
            if v is None:
                continue
            vals.append(float(v) if not isinstance(v, bool) else (1.0 if v else 0.0))
    return np.asarray(vals, dtype=float)


def extract_metric_by_case(eval_data: dict, metric_name: str):
    """Return dict(case_name -> np.array(values))."""
    out = {}
    for case_name, case in eval_data.items():
        vals = []
        shot_metrics = case.get("shot_metrics", [])
        for m in shot_metrics:
            if m is None:
                continue
            v = m.get(metric_name, None)
            if v is None:
                continue
            vals.append(float(v) if not isinstance(v, bool) else (1.0 if v else 0.0))
        out[case_name] = np.asarray(vals, dtype=float)
    return out


def violin_box_plot(series_dict, title, ylabel, filename):
    """
    series_dict: label -> np.array
    """
    labels = list(series_dict.keys())
    data = [series_dict[k] for k in labels]

    plt.figure(figsize=(8, 4.5))
    parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # overlay boxplot
    plt.boxplot(data, widths=0.15, showfliers=False)

    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    out_path = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def scatter_distance_power(dists, powers, title, filename):
    plt.figure(figsize=(6.5, 5))
    for label in dists.keys():
        x = dists[label]
        y = powers[label]
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        plt.scatter(x, y, alpha=0.35, label=label)

        # trend line (simple linear fit)
        if n >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 100)
            ys = m * xs + b
            plt.plot(xs, ys, linewidth=2)

    plt.xlabel("Shot distance (m)")
    plt.ylabel("Shot power")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    out_path = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def ecdf_plot(series_dict, title, xlabel, filename):
    plt.figure(figsize=(7, 4.5))
    for label, arr in series_dict.items():
        if arr.size == 0:
            continue
        x = np.sort(arr)
        y = np.arange(1, x.size + 1) / x.size
        plt.plot(x, y, label=label, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    out_path = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def per_case_grouped_bar(metric_by_case_by_label, title, ylabel, filename):
    """
    metric_by_case_by_label: label -> dict(case -> np.array(values))
    """
    labels = list(metric_by_case_by_label.keys())
    cases = sorted(next(iter(metric_by_case_by_label.values())).keys())

    means = {lab: [np.mean(metric_by_case_by_label[lab][c]) for c in cases] for lab in labels}
    cis = {}
    for lab in labels:
        cis[lab] = []
        for c in cases:
            arr = metric_by_case_by_label[lab][c]
            if arr.size < 2:
                cis[lab].append(0.0)
            else:
                ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(arr.size)
                cis[lab].append(ci)

    x = np.arange(len(cases))
    width = 0.8 / len(labels)

    plt.figure(figsize=(9, 4.5))
    for i, lab in enumerate(labels):
        plt.bar(x + i * width, means[lab], width=width, yerr=cis[lab], capsize=5, label=lab)

    plt.xticks(x + width * (len(labels) - 1) / 2, cases)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    out_path = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    # carica eval data
    eval_data = {}
    for label, path in EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"[SKIP] missing: {path}")
            continue
        eval_data[label] = load_eval(path)

    if not eval_data:
        raise SystemExit("No evaluation JSON found. Check EVAL_FILES paths.")

   
    power = {lab: extract_metric(d, "shot_power") for lab, d in eval_data.items()}
    dist  = {lab: extract_metric(d, "shot_distance") for lab, d in eval_data.items()}
    step  = {lab: extract_metric(d, "shot_step") for lab, d in eval_data.items()}

    # 1) Distributions (violin + box)
    violin_box_plot(power, "Shot Power distribution (all test cases)", "Shot power", "dist_shot_power_violin.png")
    violin_box_plot(dist,  "Shot Distance distribution (all test cases)", "Shot distance (m)", "dist_shot_distance_violin.png")
    violin_box_plot(step,  "Shot Step distribution (all test cases)", "Step when shot is taken", "dist_shot_step_violin.png")

    # 2) Scatter distance vs power
    scatter_distance_power(dist, power, "Shot distance vs shot power", "scatter_distance_vs_power.png")

    # 3) ECDF of shot step
    ecdf_plot(step, "ECDF: how early players shoot", "Shot step", "ecdf_shot_step.png")

    # 4) Per-test-case comparison (means + CI95)
    power_by_case = {lab: extract_metric_by_case(d, "shot_power") for lab, d in eval_data.items()}
    per_case_grouped_bar(power_by_case, "Mean shot power per test case", "Shot power", "per_case_shot_power.png")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()