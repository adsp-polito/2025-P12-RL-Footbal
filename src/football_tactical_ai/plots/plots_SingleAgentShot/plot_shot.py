import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from football_tactical_ai.env.objects.pitch import Pitch


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


# 1) TRAINING REWARD CURVE
def plot_training_rewards(json_path, window=200, save_name="Shot_RewardCurve.png"):
    set_plot_style()

    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])
    rewards  = np.array([d["reward"] for d in data])

    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.25, color="#888888", linewidth=0.8)
    ax.plot(episodes[window-1:], smooth, color="#009E73", linewidth=2.2,
            label=f"Moving Average ({window})")

    ax.set_title("Training Reward Curve - OffensiveScenarioShot", fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()

    plt.savefig(save_name, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_name}")




# UTILITY
def compute_average_trajectory(trajs):
    min_len = min(len(t) for t in trajs)
    truncated = [t[:min_len] for t in trajs]
    return np.mean(np.stack(truncated, axis=0), axis=0)



# 2) SINGLE TEST CASE TRAJECTORIES
def plot_shot_trajectories(json_path, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    pitch = Pitch()

    fig, ax = pitch.draw_pitch(
        ax=None,
        field_color="green",
        stripes=False,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False,
    )

    # Remove axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    cmap = plt.get_cmap("tab20")
    run_idx = 0

    all_att = []
    all_ball = []

    for run_data in data.values():

        if "attacker" not in run_data or "ball" not in run_data:
            continue

        att_norm = np.array(run_data["attacker"])
        ball_norm = np.array(run_data["ball"])

        # DENORMALIZE
        att_real = []
        ball_real = []

        for x, y in att_norm:
            att_real.append((
                x*(pitch.x_max-pitch.x_min)+pitch.x_min,
                y*(pitch.y_max-pitch.y_min)+pitch.y_min
            ))

        for x, y in ball_norm:
            ball_real.append((
                x*(pitch.x_max-pitch.x_min)+pitch.x_min,
                y*(pitch.y_max-pitch.y_min)+pitch.y_min
            ))

        att_real = np.array(att_real)
        ball_real = np.array(ball_real)

        all_att.append(att_real)
        all_ball.append(ball_real)

        col = cmap(run_idx % 20)

        # attacker
        ax.plot(att_real[:,0], att_real[:,1], color=col, alpha=0.55, linewidth=1.4)

        # ball
        ax.plot(ball_real[:,0], ball_real[:,1], color=col, alpha=0.45,
                linewidth=1.2, linestyle="--")

        run_idx += 1

    # AVERAGE TRAJECTORIES
    if all_att:
        avg_att = compute_average_trajectory(all_att)
        ax.plot(avg_att[:,0], avg_att[:,1], color="black", linewidth=2.4, label="Average attacker")

    if all_ball:
        avg_ball = compute_average_trajectory(all_ball)
        ax.plot(avg_ball[:,0], avg_ball[:,1], color="red", linewidth=2.0,
                linestyle="--", label="Average ball")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(frameon=True)

    out = os.path.join(save_dir, os.path.basename(json_path).replace(".json",".png"))
    plt.tight_layout()
    plt.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out}")



# 3) ALL TEST CASES ON SINGLE PITCH
def plot_all_testcases_single_pitch(folder, save_path):
    traj_files = sorted([
        f for f in os.listdir(folder)
        if f.startswith("shot_trajectories_") and f.endswith(".json")
    ])

    if not traj_files:
        print("[ERROR] No files found.")
        return

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

    for fname in traj_files:
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)

        col = palette[color_idx % len(palette)]
        color_idx += 1

        all_att = []
        all_ball = []

        for run_data in data.values():

            if "attacker" not in run_data or "ball" not in run_data:
                continue

            att = np.array(run_data["attacker"])
            ball = np.array(run_data["ball"])

            # DENORMALIZE
            att_real = []
            ball_real = []

            for x, y in att:
                att_real.append((
                    x*(pitch.x_max-pitch.x_min)+pitch.x_min,
                    y*(pitch.y_max-pitch.y_min)+pitch.y_min
                ))

            for x, y in ball:
                ball_real.append((
                    x*(pitch.x_max-pitch.x_min)+pitch.x_min,
                    y*(pitch.y_max-pitch.y_min)+pitch.y_min
                ))

            att_real = np.array(att_real)
            ball_real = np.array(ball_real)

            all_att.append(att_real)
            all_ball.append(ball_real)

            # attacker faint
            ax.plot(att_real[:,0], att_real[:,1], color=col, alpha=0.55, linewidth=1.3)

            # ball faint
            ax.plot(ball_real[:,0], ball_real[:,1], color=col, alpha=0.45,
                    linewidth=1.1, linestyle="--")

        # AVERAGE TRAJECTORIES
        if all_att:
            avg_att = compute_average_trajectory(all_att)
            ax.plot(avg_att[:,0], avg_att[:,1], color=col, linewidth=2.3,
                    label=f"{fname.replace('shot_trajectories_', '').replace('.json', '')} - attacker")

        if all_ball:
            avg_ball = compute_average_trajectory(all_ball)
            ax.plot(avg_ball[:,0], avg_ball[:,1], color=col, linewidth=2.0,
                    linestyle="--", alpha=0.9,
                    label=f"{fname.replace('shot_trajectories_', '').replace('.json', '')} - ball")

    ax.legend(frameon=True)
    ax.set_title("Shot Scenario - All Test Cases", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")






# 4) SHOT DISTANCE vs TIME TO SHOT
def plot_shot_scatter(eval_json_path, save_path):
    set_plot_style()

    with open(eval_json_path) as f:
        data = json.load(f)

    time_s, dist_s = [], []

    for v in data.values():
        if "shot_metrics" not in v:
            continue
        for m in v["shot_metrics"]:
            if m["valid_shot"]:
                time_s.append(m["time_to_shot"])
                dist_s.append(m["shot_distance"])

    fig, ax = plt.subplots()
    ax.scatter(time_s, dist_s, alpha=0.7, color="#009E73")

    ax.set_title("Shot Distance vs Time to Shot", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time to Shot (seconds)")
    ax.set_ylabel("Shot Distance (m)")

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
    save_name=f"{SAVE_ROOT}/SingleAgentShot_RewardCurve.png"
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

# 4) Distance vs time
plot_shot_scatter(
    eval_json_path="src/football_tactical_ai/evaluation/results/logs/shot/shot_evaluation.json",
    save_path=f"{SAVE_ROOT}/Shot_Distance_vs_Time.png"
)