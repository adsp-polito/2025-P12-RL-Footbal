import os
import json
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO   
from football_tactical_ai.helpers.helperTraining import ensure_dirs
from football_tactical_ai.helpers.helperFunctions import normalize
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi
from football_tactical_ai.env.objects.pitch import Pitch

from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.configs.configEvaluationMultiAgent import SCENARIOS_MULTI, COMMON


# FIXED RESET WRAPPER FOR MULTI-AGENT 2v1
def fixed_env_multi(pitch, attackers_m, defender_m, fps, max_steps):

    # Crea env multi-agent
    env = FootballMultiEnv({
        "fps": fps,
        "max_steps": max_steps,
        "time_step": 1.0 / fps,
        "n_attackers": 2,
        "n_defenders": 1,
        "include_goalkeeper": False,
    })

    # Buffer per traiettorie
    env.traj_A1 = []
    env.traj_A2 = []
    env.traj_ball = []

    # RESET WRAPPER
    original_reset = env.reset

    def fixed_reset(seed=None, options=None):

        obs, info = original_reset(seed=seed, options=options)

        env.traj_A1 = []
        env.traj_A2 = []
        env.traj_ball = []

        # Normalize initial positions
        A1_norm = normalize(*attackers_m[0])
        A2_norm = normalize(*attackers_m[1])
        D_norm  = normalize(*defender_m)

        # Manually set positions
        env.players["att_1"].reset_position(A1_norm)
        env.players["att_2"].reset_position(A2_norm)
        env.players["def_1"].reset_position(D_norm)

        # Ball to attacker 1
        env.ball.set_position(A1_norm)
        env.ball.set_owner("att_1")

        return env._get_obs(), {}

    env.reset = fixed_reset

    # STEP WRAPPER
    original_step = env.step

    def tracked_step(actions):

        obs, rewards, terminated, truncated, info = original_step(actions)

        # Track A1
        ax1, ay1 = env.players["att_1"].get_position()
        env.traj_A1.append((float(ax1), float(ay1)))

        # Track A2
        ax2, ay2 = env.players["att_2"].get_position()
        env.traj_A2.append((float(ax2), float(ay2)))

        # Track ball
        bx, by = env.ball.get_position()
        env.traj_ball.append((float(bx), float(by)))

        return obs, rewards, terminated, truncated, info

    env.step = tracked_step

    return env


# MAIN EVALUATION FUNCTION
def evaluate_multi(scenario="2v1"):
    """
    Evaluate a trained multi-agent 2v1 model across predefined test cases.
    This version includes:
      - reward statistics
      - shooting statistics
      - passing statistics
      - possession statistics
      - episode outcomes
      - trajectories
    """

    cfg = SCENARIOS_MULTI[scenario]

    print(f"\n=== MULTI-AGENT EVALUATION: {scenario.upper()} ===")

    # Load global configuration
    fps       = COMMON["fps"]
    max_steps = COMMON["max_steps"]

    model_path     = cfg["model_path"]
    save_video_dir = cfg["save_video_dir"]
    save_logs_dir  = cfg["save_logs_dir"]

    ensure_dirs(save_video_dir)
    ensure_dirs(save_logs_dir)

    pitch = Pitch()

    # Load trained model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Final outputs
    summary_eval = {}
    summary_traj = {}

    test_cases = cfg["test_cases"]
    print(f"Total test cases: {len(test_cases)}")

    # Loop over test cases
    for idx, case in enumerate(test_cases):

        name = case["name"]
        A_m  = case["attackers_start"]
        D_m  = case["defender_start"]

        print(f"\n→ Test case {idx+1}/{len(test_cases)}: {name}")
        print(f"   A1, A2 start (m): {A_m}")
        print(f"   Defender start : {D_m}")

        # Prepare storage for this test case
        summary_eval[name] = {}
        summary_traj[name] = {}

        rewards_A1 = []
        rewards_A2 = []

        # METRIC ACCUMULATORS

        # Shooting
        shots_A1 = shots_A2 = 0
        goals_A1 = goals_A2 = 0

        # Passing A1 <-> A2
        attempt_A1_A2 = attempt_A2_A1 = 0
        complete_A1_A2 = complete_A2_A1 = 0

        # Possession
        poss_A1 = poss_A2 = 0
        lost_A1 = lost_A2 = 0

        # Episode-level outcomes
        episodes_with_goal = 0
        episodes_out = 0

        N = 20

        # Run N episodes
        for run in tqdm(range(N), desc="runs"):

            # Create environment
            env = fixed_env_multi(
                pitch       = pitch,
                attackers_m = A_m,
                defender_m  = D_m,
                fps         = fps,
                max_steps   = max_steps,
            )

            # Run VIDEO render only for run == 0
            if run == 0:
                evaluate_and_render_multi(
                    model      = model,
                    env        = env,
                    pitch      = pitch,
                    save_path  = os.path.join(save_video_dir, f"multi_eval_{name}.mp4"),
                    episode    = 1,
                    fps        = fps,
                    show_grid    = cfg["render"]["show_grid"],
                    show_heatmap = cfg["render"]["show_heatmap"],
                    show_rewards = cfg["render"]["show_rewards"],
                    full_pitch   = cfg["render"]["full_pitch"],
                    show_names   = cfg["render"]["show_names"],
                )

            # MANUAL EPISODE ROLLOUT FOR METRICS
            obs, _ = env.reset()
            done = False

            while not done:

                # MODEL ACTIONS
                actions = {
                    agent_id: model.predict(obs[agent_id], deterministic=True)[0]
                    for agent_id in env.agents
                }

                obs, rewards, terminated, truncated, infos = env.step(actions)

                # REWARD
                rewards_A1.append(float(rewards["att_1"]))
                rewards_A2.append(float(rewards["att_2"]))

                # SHOOTING
                if infos["att_1"].get("shot_attempted"):
                    shots_A1 += 1
                if infos["att_2"].get("shot_attempted"):
                    shots_A2 += 1

                if infos["att_1"].get("goal_scored"):
                    goals_A1 += 1
                if infos["att_2"].get("goal_scored"):
                    goals_A2 += 1

                # PASSING
                # A1 → A2
                if infos["att_1"].get("pass_attempted") and infos["att_1"].get("pass_to") == "att_2":
                    attempt_A1_A2 += 1
                if infos["att_1"].get("pass_completed") and infos["att_1"].get("pass_to") == "att_2":
                    complete_A1_A2 += 1

                # A2 → A1
                if infos["att_2"].get("pass_attempted") and infos["att_2"].get("pass_to") == "att_1":
                    attempt_A2_A1 += 1
                if infos["att_2"].get("pass_completed") and infos["att_2"].get("pass_to") == "att_1":
                    complete_A2_A1 += 1

                # POSSESSION
                if infos["att_1"].get("has_ball"):
                    poss_A1 += 1
                if infos["att_2"].get("has_ball"):
                    poss_A2 += 1

                if infos["att_1"].get("possession_lost"):
                    lost_A1 += 1
                if infos["att_2"].get("possession_lost"):
                    lost_A2 += 1

                # OUTCOME
                if infos["att_1"].get("goal_team") is not None:
                    episodes_with_goal += 1
                if infos["att_1"].get("ball_out_by") is not None:
                    episodes_out += 1

                done = terminated["__all__"] or truncated["__all__"]

            # Store trajectories
            summary_traj[name][f"run_{run+1}"] = {
                "A1":   env.traj_A1,
                "A2":   env.traj_A2,
                "ball": env.traj_ball
            }

        # COMPUTE AND SAVE STATISTICS FOR THIS TEST CASE
        summary_eval[name] = {
            "A1": {
                "mean": float(np.mean(rewards_A1)),
                "std":  float(np.std(rewards_A1)),
                "min":  float(np.min(rewards_A1)),
                "max":  float(np.max(rewards_A1)),
                "runs": rewards_A1,
            },
            "A2": {
                "mean": float(np.mean(rewards_A2)),
                "std":  float(np.std(rewards_A2)),
                "min":  float(np.min(rewards_A2)),
                "max":  float(np.max(rewards_A2)),
                "runs": rewards_A2,
            },

            # SHOOTING
            "shooting_stats": {
                "shots_A1": shots_A1,
                "shots_A2": shots_A2,
                "goals_A1": goals_A1,
                "goals_A2": goals_A2,
            },

            # PASSING
            "passing_stats": {
                "attempted_A1_to_A2": attempt_A1_A2,
                "attempted_A2_to_A1": attempt_A2_A1,
                "completed_A1_to_A2": complete_A1_A2,
                "completed_A2_to_A1": complete_A2_A1,
            },

            # POSSESSION
            "possession_stats": {
                "poss_time_A1": poss_A1,
                "poss_time_A2": poss_A2,
                "lost_possession_A1": lost_A1,
                "lost_possession_A2": lost_A2,
            },

            # EPISODE OUTCOMES
            "episode_outcomes": {
                "episodes_with_goal": episodes_with_goal,
                "episodes_out_of_play": episodes_out,
            }
        }

        # Save trajectories JSON
        traj_path = os.path.join(save_logs_dir, f"multi_traj_{name}.json")
        with open(traj_path, "w") as f:
            json.dump(summary_traj[name], f, indent=2)

        print(f"   → Trajectories saved to {traj_path}")

    # Save global evaluation summary
    eval_path = os.path.join(save_logs_dir, "multi_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(summary_eval, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"All results saved to: {eval_path}")
    print("==============================================\n")

    return summary_eval



if __name__ == "__main__":
    evaluate_multi("2v1")
