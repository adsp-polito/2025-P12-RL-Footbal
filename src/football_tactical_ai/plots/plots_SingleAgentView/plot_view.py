import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from football_tactical_ai.env.objects.pitch import Pitch

SAVE_ROOT = "src/football_tactical_ai/plots/plots_SingleAgentView"
os.makedirs(SAVE_ROOT, exist_ok=True)

# GLOBAL STYLE
def set_plot_style():
    """Consistent plotting style."""
    mpl.rcParams['figure.figsize'] = (10, 5)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 13
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['legend.fontsize'] = 11

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = "--"
    mpl.rcParams['grid.alpha'] = 0.25

    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False



# 1) TRAINING REWARD CURVE
def plot_training_rewards(json_path, window=300,
                          save_path=f"{SAVE_ROOT}/SingleAgentShot_RewardCurve.png"):
    """
    Plot a clean, professional smoothed reward curve for the training.
    """

    set_plot_style()

    # Load rewards
    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])
    rewards  = np.array([d["reward"] for d in data])

    # Moving average smoothing
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot main curve
    ax.plot(episodes[window-1:], smooth,
            color="#0072B2", linewidth=2.5)

    # Title and labels
    ax.set_title("Training Reward Curve - OffensiveScenarioView",
                 fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)

    # Grid (clean)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend([f"Moving Average ({window} episodes)"], frameon=True, loc="lower right")

    # Save
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")






# 2) MOVEMENT ACCURACY — Horizontal Bar Chart
def plot_movement_accuracy(eval_json_path, save_path="View_MovementAccuracy.png"):
    """
    Horizontal bar chart for movement accuracy (FOV valid movement ratio)
    """
    set_plot_style()

    with open(eval_json_path) as f:
        data = json.load(f)

    cases = []
    ratios = []

    for case, vals in data.items():
        vm = sum(m["fov_valid_movements"] for m in vals["view_metrics"])
        im = sum(m["fov_invalid_movements"] for m in vals["view_metrics"])
        ratio = vm / max(vm + im, 1)

        cases.append(case.upper())
        ratios.append(ratio)

    # Sort by accuracy
    ordering = np.argsort(ratios)
    cases = np.array(cases)[ordering]
    ratios = np.array(ratios)[ordering]

    fig, ax = plt.subplots(figsize=(9, 5))

    color = "#0072B2"   # BLUE

    ax.barh(cases, ratios, color=color, alpha=0.85)

    # Label next to bars
    for y, r in enumerate(ratios):
        ax.text(r + 0.02, y, f"{r:.2f}", va="center",
                fontsize=12, fontweight="bold", color=color)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Movement Accuracy", fontsize=13)
    ax.set_title("Movement Accuracy Across Test Cases", fontsize=16, fontweight="bold")

    ax.grid(axis="x", linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")




# 3) SHOT ACCURACY
def plot_shot_accuracy(eval_json_path, save_path="View_ShotAccuracy.png"):
    """
    Shot accuracy = (#episodes with a valid shot) / (valid episodes + sum(invalid_shot_fov))
    
    Handles cases where:
        - valid_shot=True but invalid_shot_fov > 0 (episode includes previous invalid attempts)
        - valid_shot=False → only invalid_shot_fov
        - valid_shot=None → episode is ignored (no shot taken)
    """
    set_plot_style()

    # Load JSON
    with open(eval_json_path) as f:
        data = json.load(f)

    cases = []
    shot_acc = []

    for case, vals in data.items():

        valid_count = 0          # number of episodes where a valid shot was eventually taken
        invalid_count = 0        # total number of invalid shots (FOV violations)

        for m in vals["view_metrics"]:

            v = m["valid_shot"]
            invalid_fov = m.get("invalid_shot_fov", 0)

            if v is True:
                # episode ends with a valid shot
                valid_count += 1
                invalid_count += invalid_fov   # include invalid attempts BEFORE the valid shot

            elif v is False:
                # only invalid attempts
                invalid_count += invalid_fov

            # v is None → ignore (no shot made)

        total_shots = valid_count + invalid_count
        ratio = valid_count / total_shots if total_shots > 0 else 0.0

        cases.append(case.upper())
        shot_acc.append(ratio)

    # Sort bars for better visualization
    ordering = np.argsort(shot_acc)
    cases = np.array(cases)[ordering]
    shot_acc = np.array(shot_acc)[ordering]

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    color = "#D55E00"

    ax.barh(cases, shot_acc, color=color, alpha=0.85)

    # Labels
    for y, r in enumerate(shot_acc):
        ax.text(r + 0.02, y, f"{r:.2f}", va="center",
                fontsize=12, fontweight="bold", color=color)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Shot Accuracy", fontsize=13)
    ax.set_title("Shot Accuracy Across Test Cases", fontsize=16, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")




# MAIN EXECUTION

# 1) Reward curve
plot_training_rewards(
    json_path="src/football_tactical_ai/training/rewards/singleAgentView/rewards.json",
    save_path=f"{SAVE_ROOT}/SingleAgentView_RewardCurve.png"
)

# 2) Movement accuracy
#plot_movement_accuracy(
#    eval_json_path="src/football_tactical_ai/evaluation/results/logs/view/view_evaluation.json",
#    save_path=f"{SAVE_ROOT}/SingleAgentView_MovementAccuracy.png"
#)

# 3) Shot accuracy
#plot_shot_accuracy(
#    eval_json_path = f"src/football_tactical_ai/evaluation/results/logs/view/view_evaluation.json",
#    save_path=f"{SAVE_ROOT}/SingleAgentView_ShotAccuracy.png"
#)
