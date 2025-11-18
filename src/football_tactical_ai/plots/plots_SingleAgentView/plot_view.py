import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from football_tactical_ai.env.objects.pitch import Pitch


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
def plot_training_rewards(json_path, window=200, save_name="View_RewardCurve.png"):
    set_plot_style()

    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])
    rewards  = np.array([d["reward"] for d in data])

    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.25, color="#888888", linewidth=0.8)
    ax.plot(episodes[window-1:], smooth, color="#0072B2", linewidth=2.2,
            label=f"Moving Average ({window})")

    ax.set_title("Training Reward Curve – OffensiveScenarioView", fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()

    plt.savefig(save_name, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_name}")



# UTILITY – average trajectory
def compute_average_trajectory(trajs):
    min_len = min(len(t) for t in trajs)
    truncated = [t[:min_len] for t in trajs]
    return np.mean(np.stack(truncated, axis=0), axis=0)



# 2) TRAJECTORIES FOR VIEW SCENARIO
def plot_view_trajectories(json_path, save_dir, title):
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

        # attacker movement
        ax.plot(att_real[:,0], att_real[:,1], color=col, alpha=0.55, linewidth=1.4)

        # ball trajectory
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



# 3) MOVEMENT ACCURACY — Horizontal Bar Chart
def plot_movement_accuracy(eval_json_path, save_path="View_MovementAccuracy.png"):
    set_plot_style()

    with open(eval_json_path) as f:
        data = json.load(f)

    cases = []
    ratios = []

    for case, vals in data.items():

        vm = sum([m["fov_valid_movements"] for m in vals["view_metrics"]])
        im = sum([m["fov_invalid_movements"] for m in vals["view_metrics"]])

        ratio = vm / max(vm + im, 1)

        cases.append(case.upper())
        ratios.append(ratio)

    sorting = np.argsort(ratios)
    cases = np.array(cases)[sorting]
    ratios = np.array(ratios)[sorting]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(cases))

    # valid part
    ax.barh(y, ratios, color="#009E73", alpha=0.8)
    # invalid part
    #ax.barh(y, 1 - ratios, left=ratios, color="#D55E00", alpha=0.7)

    # labels
    for i, r in enumerate(ratios):
        ax.text(r + 0.02, i, f"{r:.2f}", va="center",
                fontsize=12, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(cases, fontsize=13)
    ax.set_xlim(0, 1.05)

    ax.set_xlabel("Movement Accuracy (Valid Movement Ratio)")
    ax.set_title("FOV Movement Accuracy Across Test Cases", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")



# 4) SHOT ACCURACY — Lollipop Chart
def plot_shot_accuracy(eval_json_path, save_path="View_ShotAccuracy.png"):
    set_plot_style()

    with open(eval_json_path) as f:
        data = json.load(f)

    cases = []
    shot_acc = []

    for case, vals in data.items():

        valid_list = []
        shot_list = []

        for m in vals["view_metrics"]:
            if m["valid_shot"] is True:
                valid_list.append(1)
                shot_list.append(1)
            elif m["valid_shot"] is False:
                valid_list.append(0)
                shot_list.append(1)
            # if None → no shot → do not count the episode

        # compute ratio only on episodes where a shot was taken
        if len(shot_list) == 0:
            ratio = 0.0
        else:
            ratio = sum(valid_list) / len(shot_list)



        cases.append(case.upper())
        shot_acc.append(ratio)

    x = np.arange(len(cases))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hlines(y=x, xmin=0, xmax=shot_acc, color="#0072B2", linewidth=2)
    ax.scatter(shot_acc, x, color="#0072B2", s=140)

    for xi, r in zip(x, shot_acc):
        ax.text(r + 0.03, xi, f"{r:.2f}", va="center",
                fontsize=12, fontweight="bold")

    ax.set_yticks(x)
    ax.set_yticklabels(cases)
    ax.set_xlim(0, 1.10)

    ax.set_xlabel("Shot Accuracy (Valid Shots Ratio)")
    ax.set_title("Shot Decision Accuracy Across Test Cases", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {save_path}")



# MAIN EXECUTION

SAVE_ROOT = "src/football_tactical_ai/plots/plots_SingleAgentView"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 1) Reward curve
plot_training_rewards(
    json_path="src/football_tactical_ai/training/rewards/singleAgentView/rewards.json",
    save_name=f"{SAVE_ROOT}/SingleAgentView_RewardCurve.png"
)

# 2) Trajectories (per test-case)
plot_folder = "src/football_tactical_ai/evaluation/results/logs/view"

for fname in os.listdir(plot_folder):
    if fname.endswith(".json") and fname.startswith("view_trajectories_"):
        json_path = os.path.join(plot_folder, fname)
        clean = fname.replace(".json", "").split("view_trajectories_")[-1]
        plot_view_trajectories(
            json_path=json_path,
            save_dir=SAVE_ROOT,
            title=f"View Scenario – {clean.upper()} Test Case"
        )

# 3) Movement accuracy
plot_movement_accuracy(
    eval_json_path="src/football_tactical_ai/evaluation/results/logs/view/view_evaluation.json",
    save_path=f"{SAVE_ROOT}/SingleAgentView_MovementAccuracy.png"
)

# 4) Shot accuracy
plot_shot_accuracy(
    eval_json_path
=f"src/football_tactical_ai/evaluation/results/logs/view/view_evaluation.json",
    save_path=f"{SAVE_ROOT}/SingleAgentView_ShotAccuracy.png"
)
