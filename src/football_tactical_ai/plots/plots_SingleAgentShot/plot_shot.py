import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

from football_tactical_ai.env.objects.pitch import Pitch


#  DIRECTORY SETUP
SAVE_ROOT = "src/football_tactical_ai/plots/plots_SingleAgentShot"
os.makedirs(SAVE_ROOT, exist_ok=True)

# GLOBAL PLOTTING STYLE
def set_plot_style():
    """Set a consistent plotting style for all shot figures."""
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

# DENORMALIZATION helper
def denorm(arr, pitch=Pitch()):
    return np.column_stack([
        arr[:,0] * (pitch.x_max - pitch.x_min) + pitch.x_min,
        arr[:,1] * (pitch.y_max - pitch.y_min) + pitch.y_min
    ])




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
    ax.set_title("Training Reward Curve - OffensiveScenarioShot",
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



# 2) SINGLE TEST CASE TRAJECTORIES
def plot_shot_trajectories(json_path, save_dir, title):
    """
    Plot only the BEST run using shot_evaluation.json
    """

    os.makedirs(save_dir, exist_ok=True)

    # Extract scenario name (e.g. shot_trajectories_central.json --> "central")
    fname = os.path.basename(json_path)
    scenario_name = fname.replace("shot_trajectories_", "").replace(".json", "")

    # Load trajectory JSON
    with open(json_path, "r") as f:
        traj_data = json.load(f)

    # Load global evaluation JSON
    eval_path = os.path.join(os.path.dirname(json_path), "shot_evaluation.json")
    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    if scenario_name not in eval_data:
        print(f"[ERROR] Scenario '{scenario_name}' not in shot_evaluation.json")
        return

    scenario_eval = eval_data[scenario_name]

    # BEST RUN index
    runs_rewards = scenario_eval["runs"]
    best_run_idx = int(np.argmax(runs_rewards))
    best_run_key = f"run_{best_run_idx+1}"

    if best_run_key not in traj_data:
        print(f"[ERROR] Missing {best_run_key} in {json_path}")
        return

    att_norm = np.array(traj_data[best_run_key]["attacker"])
    ball_norm = np.array(traj_data[best_run_key]["ball"])

    # Shot step
    shot_step = scenario_eval["shot_metrics"][best_run_idx].get("shot_step", None)
    if shot_step is None:
        shot_step = len(att_norm) - 1

    pitch = Pitch()

    # Draw pitch
    fig, ax = pitch.draw_pitch(
        ax=None,
        field_color="green",
        stripes=False,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # COLOR 
    col = "#00BFFF" 


    att_real = denorm(att_norm, pitch=pitch)
    ball_real = denorm(ball_norm, pitch=pitch)

    # Plot attacker path
    ax.plot(
        att_real[:shot_step+2, 0],
        att_real[:shot_step+2, 1],
        color=col,
        linewidth=3.0,
        linestyle="-",
        label="Attacker"
    )

    # Plot ball trajectory (dashed)
    ax.plot(
        ball_real[shot_step-2:, 0],
        ball_real[shot_step-2:, 1],
        color=col,
        linewidth=2.5,
        linestyle="--",
        alpha=1.0,
        label="Ball"
    )

    # MARKERS

    # Starting point (circle)
    start_x, start_y = att_real[0]
    ax.scatter(
        start_x, start_y,
        color=col,
        s=200,
        marker='o',
        edgecolor="black",
        linewidth=0.7,
        zorder=5
    )

    # Shot point (square)
    shot_x, shot_y = att_real[shot_step]
    ax.scatter(
        shot_x, shot_y,
        color=col,
        s=200,
        marker='s',
        edgecolor="black",
        linewidth=0.8,
        zorder=6
    )


    # Title + Save
    ax.set_title(f"{title} — BEST RUN", fontsize=16, fontweight="bold")
    ax.legend(frameon=True, loc="upper left")

    out = os.path.join(save_dir, fname.replace(".json", ".png"))
    plt.tight_layout()
    plt.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out}")




# 3) ALL TEST CASES ON SINGLE PITCH
def plot_all_testcases_single_pitch(folder, save_path):
    """
    For each test-case:
    - Load its trajectories (shot_trajectories_*.json)
    - Use the unique shot_evaluation.json to find the BEST RUN
    - Plot attacker only until the shot step
    - Plot ball trajectory from shot step onward
    - Draw start marker (circle) and shot marker (square)
    """

    # Collect all trajectory files
    traj_files = sorted([
        f for f in os.listdir(folder)
        if f.startswith("shot_trajectories_") and f.endswith(".json")
    ])

    if not traj_files:
        print("[ERROR] No trajectories found.")
        return

    # Load GLOBAL shot_evaluation.json
    eval_path = os.path.join(folder, "shot_evaluation.json")
    if not os.path.exists(eval_path):
        print("[ERROR] shot_evaluation.json NOT FOUND.")
        return

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    # Prepare pitch figure
    set_plot_style()
    pitch = Pitch()

    fig, ax = pitch.draw_pitch(
        ax=None,
        field_color="green",
        stripes=False,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False,
    )
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    palette = plt.get_cmap("tab10").colors
    color_idx = 0

    # Process each trajectory file
    for traj_file in traj_files:

        scenario_name = traj_file.replace("shot_trajectories_", "").replace(".json", "")

        if scenario_name not in eval_data:
            print(f"[WARNING] '{scenario_name}' missing in shot_evaluation.json — skipping.")
            continue

        traj_path = os.path.join(folder, traj_file)

        with open(traj_path, "r") as f:
            traj_data = json.load(f)

        scenario_eval = eval_data[scenario_name]

        rewards = scenario_eval["runs"]
        best_idx = int(np.argmax(rewards))
        best_run_key = f"run_{best_idx+1}"

        if best_run_key not in traj_data:
            print(f"[WARNING] Missing {best_run_key} in {traj_file}")
            continue

        att_norm = np.array(traj_data[best_run_key]["attacker"])
        ball_norm = np.array(traj_data[best_run_key]["ball"])
        shot_step = scenario_eval["shot_metrics"][best_idx].get("shot_step", None)

        if shot_step is None:
            shot_step = len(att_norm) - 1

        # Assign consistent color
        col = palette[color_idx % len(palette)]
        color_idx += 1

        att_real = denorm(att_norm, pitch=pitch)
        ball_real = denorm(ball_norm, pitch=pitch)

        # Plot attacker path until shot
        ax.plot(
            att_real[:shot_step+2,0],
            att_real[:shot_step+2,1],
            color=col,
            linewidth=3.0,
            linestyle="-",
            label=f"{scenario_name} - attacker"
        )

        # Plot ball trajectory after shot
        ax.plot(
            ball_real[shot_step-2:,0],
            ball_real[shot_step-2:,1],
            color=col,
            linewidth=2.5,
            linestyle="--",
            alpha=1.0,
            label=f"{scenario_name} - ball"
        )


        # STARTING POINT MARKER
        start_x, start_y = att_real[0]
        ax.scatter(
            start_x, start_y,
            color=col,
            s=150,
            marker='o',   # circle
            edgecolor="black",
            linewidth=1.0,
            zorder=5
        )

        # SHOT POINT MARKER
        shot_x, shot_y = att_real[shot_step]
        ax.scatter(
            shot_x, shot_y,
            color=col,
            s=150,
            marker='s',   # square
            edgecolor="black",
            linewidth=1.0,
            zorder=6
        )


    # LEGEND — trajectories + start and shot markers

    # Dummy legend entries
    start_handle = mlines.Line2D(
        [], [], 
        color="white",
        marker='o',
        markersize=10,
        linestyle='None',
        markeredgecolor="black",
        label="Start point"
    )

    shot_handle = mlines.Line2D(
        [], [], 
        color="white",
        marker='s',
        markersize=12,
        linestyle='None',
        markeredgecolor="black",
        label="Shot point"
    )

    # Merge with existing trajectory legend entries
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([start_handle, shot_handle])
    labels.extend(["Start point", "Shot point"])

    ax.legend(handles, labels, frameon=True, loc="upper left")

    ax.set_title("Shot Scenario — BEST RUN per Test Case", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")







# EXECUTION
SAVE_ROOT = "src/football_tactical_ai/plots/plots_SingleAgentShot"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 1) Reward curve
plot_training_rewards(
    json_path="src/football_tactical_ai/training/rewards/singleAgentShot/rewards.json",
    window=300,
    save_path=f"{SAVE_ROOT}/SingleAgentShot_RewardCurve.png"
)

# 2) Single test case plots
plot_folder = "src/football_tactical_ai/evaluation/results/logs/shot"

for fname in os.listdir(plot_folder):

    # Process only JSON files
    if not fname.endswith(".json"):
        continue

    # Process only shot trajectory files
    if not fname.startswith("shot_trajectories_"):
        continue

    json_path = os.path.join(plot_folder, fname)

    # Extract clean test-case name (e.g., "left", "right", "edge_box")
    clean_name = fname.replace(".json", "").split("shot_trajectories_")[-1]

    plot_shot_trajectories(
        json_path=json_path,
        save_dir=SAVE_ROOT,
        title=f"Shot Scenario - {clean_name.upper()} Test Case"
    )

# 3) All testcases combined
plot_all_testcases_single_pitch(
    folder=plot_folder,
    save_path=f"{SAVE_ROOT}/SingleAgentShot_AllTestcases.png"
)
