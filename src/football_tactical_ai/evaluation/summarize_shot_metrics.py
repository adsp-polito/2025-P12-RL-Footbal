import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_std_ci95(x: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (mean, std(ddof=1), 95% CI half-width) for a 1D array.
    If not enough data, returns Nones appropriately.
    """
    if x.size == 0:
        return None, None, None
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, None, None
    std = float(np.std(x, ddof=1))
    # normal approximation CI (good enough for slide-level analytics)
    ci95 = float(1.96 * std / np.sqrt(x.size))
    return mean, std, ci95


def summarize_shot_eval(evaluation_json_path: str) -> Dict[str, Any]:
    """
    Summarize a {scenario}_evaluation.json produced by eval_single_agent.py.

    Expected structure:
      {
        "<case_name>": {
          "mean": ..., "std": ..., "min": ..., "max": ...,
          "runs": [ ... ],
          "shot_metrics": [
             {"valid_shot": ..., "shot_distance": ..., "shot_step": ..., "shot_angle": ..., "shot_power": ...},
             ...
          ]
        },
        ...
      }
    """
    data = _read_json(evaluation_json_path)

    # Global containers across all test cases
    all_rewards: List[float] = []
    valid_shot: List[int] = []
    shot_power: List[float] = []
    shot_distance: List[float] = []
    shot_step: List[float] = []
    shot_angle: List[float] = []

    per_case: Dict[str, Any] = {}

    for case_name, case in data.items():
        runs = case.get("runs", [])
        metrics = case.get("shot_metrics", [])

        # Rewards
        r = np.asarray(runs, dtype=float)
        all_rewards.extend(r.tolist())

        # Shot metrics (some can be None if no shot happened)
        v = []
        p = []
        d = []
        s = []
        a = []

        for m in metrics:
            if m is None:
                continue
            if m.get("valid_shot") is not None:
                v.append(1 if bool(m["valid_shot"]) else 0)
            if m.get("shot_power") is not None:
                p.append(float(m["shot_power"]))
            if m.get("shot_distance") is not None:
                d.append(float(m["shot_distance"]))
            if m.get("shot_step") is not None:
                s.append(float(m["shot_step"]))
            if m.get("shot_angle") is not None:
                a.append(float(m["shot_angle"]))

        # Aggregate per-case
        v_arr = np.asarray(v, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        d_arr = np.asarray(d, dtype=float)
        s_arr = np.asarray(s, dtype=float)
        a_arr = np.asarray(a, dtype=float)

        per_case[case_name] = {
            "episodes": int(len(r)),
            "reward_mean": float(np.mean(r)) if r.size else None,
            "reward_std": float(np.std(r, ddof=1)) if r.size > 1 else None,
            "valid_shot_rate": float(np.mean(v_arr)) if v_arr.size else None,  # 0..1
            "shot_power_mean": float(np.mean(p_arr)) if p_arr.size else None,
            "shot_distance_mean": float(np.mean(d_arr)) if d_arr.size else None,
            "shot_step_mean": float(np.mean(s_arr)) if s_arr.size else None,
            "shot_angle_mean": float(np.mean(a_arr)) if a_arr.size else None,
        }

        valid_shot.extend(v)
        shot_power.extend(p)
        shot_distance.extend(d)
        shot_step.extend(s)
        shot_angle.extend(a)

    # Global aggregates (across all cases)
    all_rewards_arr = np.asarray(all_rewards, dtype=float)
    valid_arr = np.asarray(valid_shot, dtype=float)
    power_arr = np.asarray(shot_power, dtype=float)
    dist_arr = np.asarray(shot_distance, dtype=float)
    step_arr = np.asarray(shot_step, dtype=float)
    angle_arr = np.asarray(shot_angle, dtype=float)

    reward_mean, reward_std, reward_ci95 = _mean_std_ci95(all_rewards_arr)
    power_mean, power_std, power_ci95 = _mean_std_ci95(power_arr)
    dist_mean, dist_std, dist_ci95 = _mean_std_ci95(dist_arr)
    step_mean, step_std, step_ci95 = _mean_std_ci95(step_arr)

    out = {
        "source_file": evaluation_json_path,
        "episodes_total": int(all_rewards_arr.size),

        # Reward
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_ci95": reward_ci95,

        # Shot success proxy
        "valid_shot_rate": float(np.mean(valid_arr)) if valid_arr.size else None,

        # Key “physics-linked” analytics
        "shot_power_mean": power_mean,
        "shot_power_std": power_std,
        "shot_power_ci95": power_ci95,

        "shot_distance_mean": dist_mean,
        "shot_distance_std": dist_std,
        "shot_distance_ci95": dist_ci95,

        "shot_step_mean": step_mean,
        "shot_step_std": step_std,
        "shot_step_ci95": step_ci95,

        # Optional: angle summary (mean only)
        "shot_angle_mean": float(np.mean(angle_arr)) if angle_arr.size else None,

        "per_case": per_case,
    }
    return out


def print_compact_summary(label: str, summary: Dict[str, Any]) -> None:
    def fmt(m, ci):
        if m is None:
            return "n/a"
        if ci is None:
            return f"{m:.3f}"
        return f"{m:.3f} ± {ci:.3f}"

    print(f"\n===== {label} =====")
    print(f"Episodes: {summary['episodes_total']}")
    print(f"Reward: {fmt(summary['reward_mean'], summary['reward_ci95'])}")
    print(f"Valid shot rate: {summary['valid_shot_rate']:.3f}" if summary["valid_shot_rate"] is not None else "Valid shot rate: n/a")
    print(f"Shot power: {fmt(summary['shot_power_mean'], summary['shot_power_ci95'])}")
    print(f"Shot distance: {fmt(summary['shot_distance_mean'], summary['shot_distance_ci95'])}")
    print(f"Shot step: {fmt(summary['shot_step_mean'], summary['shot_step_ci95'])}")


if __name__ == "__main__":
    weak_json = "src/football_tactical_ai/evaluation/results/logs/shot_weak/shot_weak_evaluation.json"
    normal_json = "src/football_tactical_ai/evaluation/results/logs/shot/shot_evaluation.json"
    strong_json = "src/football_tactical_ai/evaluation/results/logs/shot_strong/shot_strong_evaluation.json"

    items = []
    for label, p in [("WEAK", weak_json), ("NORMAL", normal_json), ("STRONG", strong_json)]:
        if os.path.exists(p):
            items.append((label, p))
        else:
            print(f"[SKIP] File not found: {p}")

    if not items:
        raise SystemExit("No evaluation JSON found. Check the paths at the bottom of this file.")

    summaries = {}
    for label, p in items:
        summaries[label] = summarize_shot_eval(p)
        print_compact_summary(label, summaries[label])

    
    out_path = "src/football_tactical_ai/evaluation/results/logs/shot_summary_metrics.json"
    Path(out_path).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {out_path}")