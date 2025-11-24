import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from football_tactical_ai.env.objects.pitch import Pitch


# GLOBAL PLOTTING STYLE
def set_plot_style():
    """Set a consistent plotting style for all figures"""
    mpl.rcParams['figure.figsize'] = (10, 5)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 13
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['legend.fontsize'] = 11

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = "--"
    mpl.rcParams['grid.alpha'] = 0.3

    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False



#  DIRECTORY SETUP
SAVE_ROOT = "src/football_tactical_ai/plots/plots_MultiAgent"
os.makedirs(SAVE_ROOT, exist_ok=True)



def plot_training_rewards(json_path, window=300,
                        save_path=f"{SAVE_ROOT}/MultiAgent_RewardCurve.png"):
    """
    Plot smoothed reward curves for the two attackers in multi-agent training
    """

    set_plot_style()

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])

    # Extract attacker rewards
    att1_rewards = np.array([d["rewards"]["att_1"] for d in data])
    att2_rewards = np.array([d["rewards"]["att_2"] for d in data])

    # Moving averages
    def smooth_curve(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode="valid")

    smooth_att1 = smooth_curve(att1_rewards)
    smooth_att2 = smooth_curve(att2_rewards)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        episodes[window-1:], smooth_att1,
        color="#0072B2", linewidth=2.5,
        label=f"Attacker 1 – MA({window})"
    )
    ax.plot(
        episodes[window-1:], smooth_att2,
        color="#D55E00", linewidth=2.5,
        label=f"Attacker 2 – MA({window})"
    )

    # Titles and labels
    ax.set_title("Training Reward Curve – Multi-Agent (Attackers)",
                 fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, loc="lower right")

    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")



# MAIN EXECUTION
if __name__ == "__main__":
    plot_training_rewards(
        json_path="src/football_tactical_ai/training/rewards/multiAgent/rewards.json",
        window=300,
        save_path=f"{SAVE_ROOT}/MultiAgent_RewardCurve.png"
    )