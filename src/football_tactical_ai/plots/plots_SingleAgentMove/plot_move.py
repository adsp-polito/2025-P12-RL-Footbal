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



# 1) TRAINING REWARD CURVE
def plot_training_rewards(json_path, window=200, save_name="reward_curve.png"):
    """
    Generate and save a smoothed reward curve from the training log
    The JSON file must contain entries of the form:
    {"episode": int, "reward": float}.
    """

    set_plot_style()

    # Load reward values
    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])
    rewards  = np.array([d["reward"] for d in data])

    # Moving average smoothing
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    # Plot figure
    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.20, color="#7f7f7f", linewidth=0.8)
    ax.plot(episodes[window-1:], smooth, color="#0072B2", linewidth=2.2,
            label=f"Moving Average ({window})")

    ax.set_title("Training Reward Curve - OffensiveScenarioMove", fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(frameon=True, loc="lower right")

    # Save figure in target folder
    plt.savefig(save_name, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_name}")




# UTILITY
def compute_average_trajectory(trajs):
    """
    Compute an average attacker trajectory given multiple runs
    """
    min_len = min(len(t) for t in trajs)
    truncated = [t[:min_len] for t in trajs]
    arr = np.stack(truncated, axis=0)
    return np.mean(arr, axis=0)


# Plot single test case
def plot_attacker_trajectories(
    json_path,
    save_dir,
    title="Single-Agent Move - Trajectories"
):
    """
    Load all trajectories from a JSON file and render them on the Pitch.
    Each run is drawn in a light colour, with the average trajectory emphasized.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load trajectories JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    pitch = Pitch()

    # Initialise a new pitch drawing
    fig, ax = pitch.draw_pitch(
        ax=None,
        field_color="green",
        stripes=False,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False
    )


    # Remove axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    all_trajs = []
    cmap = plt.get_cmap("tab20")
    run_idx = 0

    # Iterate over individual runs
    for run_name, run_data in data.items():

        # Validate structure
        if "attacker" not in run_data:
            continue

        traj_norm = np.array(run_data["attacker"])  # shape (T, 2)

        # Convert normalized coordinates → meters
        traj_real = []
        for x_norm, y_norm in traj_norm:
            x_m = x_norm * (pitch.x_max - pitch.x_min) + pitch.x_min
            y_m = y_norm * (pitch.y_max - pitch.y_min) + pitch.y_min
            traj_real.append((x_m, y_m))

        traj_real = np.array(traj_real)
        all_trajs.append(traj_real)

        # Plot each run in light colour
        ax.plot(
            traj_real[:, 0],
            traj_real[:, 1],
            color=cmap(run_idx % 20),
            linewidth=1.0,
            alpha=0.35
        )
        run_idx += 1

    # Compute average trajectory
    if len(all_trajs) > 0:
        avg_traj = compute_average_trajectory(all_trajs)
        ax.plot(
            avg_traj[:, 0],
            avg_traj[:, 1],
            color="black",
            linewidth=2.5,
            alpha=0.9,
            label="Average trajectory"
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
       
    # Save figure
    case_name = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(save_dir, f"{case_name}.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")



# 2) PROCESS ALL TEST CASES
def plot_all_testcases(folder, save_dir):
    """
    Process only trajectory JSON files, ignoring evaluation summary files.
    A file is considered valid if it starts with 'move_trajectories_'.
    """

    for fname in os.listdir(folder):

        if not fname.endswith(".json"):
            continue

        # Only process the correct files
        if not fname.startswith("move_trajectories_"):
            continue

        json_path = os.path.join(folder, fname)

        clean_name = fname.replace(".json", "").split("_")[-1].upper()


        plot_attacker_trajectories(
            json_path=json_path,
            save_dir=save_dir,
            title=f"Move Scenario - {clean_name} Test Case"
        )


# 3) PLOT ALL TEST CASES ON A SINGLE PITCH
def plot_all_testcases_single_pitch(folder, save_path):
    """
    Plot all test case trajectories on a single football pitch.
    Each test case is drawn with a distinct colour, with all its runs in light tone
    and its average trajectory emphasized.
    """

    # Collect files
    traj_files = [
        f for f in os.listdir(folder)
        if f.startswith("move_trajectories_") and f.endswith(".json")
    ]

    if len(traj_files) == 0:
        print("[ERROR] No trajectory files found.")
        return

    traj_files.sort()

    set_plot_style()

    # Create pitch figure
    pitch = Pitch()
    fig, ax = pitch.draw_pitch(
        ax=None,
        field_color="green",
        stripes=False,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False
    )


    # Remove axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Build tab10 without greens
    tab10 = plt.get_cmap("tab10").colors
    EXCLUDE = {"#2ca02c", "#bcbd22"}  # greens
    CUSTOM_TAB10 = [c for c in tab10 if mpl.colors.to_hex(c) not in EXCLUDE]

    # Start colour index
    case_idx = 0

    # Process each test case
    for fname in traj_files:

        json_path = os.path.join(folder, fname)

        # Load JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # Assign a base colour to the entire test case
        base_color = CUSTOM_TAB10[case_idx % len(CUSTOM_TAB10)]

        all_trajs = []

        # Plot individual runs (faint)
        for run_name, run_data in data.items():

            if "attacker" not in run_data:
                continue

            traj_norm = np.array(run_data["attacker"])

            # Denormalize
            traj_real = []
            for x_norm, y_norm in traj_norm:
                x_m = x_norm * (pitch.x_max - pitch.x_min) + pitch.x_min
                y_m = y_norm * (pitch.y_max - pitch.y_min) + pitch.y_min
                traj_real.append((x_m, y_m))

            traj_real = np.array(traj_real)
            all_trajs.append(traj_real)

            # Light line for each run
            ax.plot(
                traj_real[:, 0],
                traj_real[:, 1],
                linewidth=1.5,
                alpha=0.25,
                color=base_color
            )

        # Compute + plot average trajectory
        if len(all_trajs) > 0:
            avg_traj = compute_average_trajectory(all_trajs)
            ax.plot(
                avg_traj[:, 0],
                avg_traj[:, 1],
                linewidth=2.5,
                alpha=0.9,
                color=base_color,
                label=fname.replace("move_trajectories_", "").replace(".json", "")
            )

        case_idx += 1

    # Legend
    ax.legend(frameon=True, loc="upper right")
    ax.set_title("Move Scenario - All Test Cases on Single Pitch", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")



# SCRIPT EXECUTION
SAVE_ROOT = "src/football_tactical_ai/plots/plots_SingleAgentMove"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 1) Reward curve
plot_training_rewards(
    json_path="src/football_tactical_ai/training/rewards/singleAgentMove/rewards.json",
    window=200,
    save_name=f"{SAVE_ROOT}/SingleAgentMove_RewardCurve.png"
)

# 2) All testcases
plot_all_testcases(
    folder="src/football_tactical_ai/evaluation/results/logs/move",
    save_dir=SAVE_ROOT
)

# 3) All testcases in one pitch
plot_all_testcases_single_pitch(
    folder="src/football_tactical_ai/evaluation/results/logs/move",
    save_path=f"{SAVE_ROOT}/SingleAgentMove_AllTestcases.png"
)