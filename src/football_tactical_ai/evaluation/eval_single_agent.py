import os
import json
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

from football_tactical_ai.helpers.helperTraining import ensure_dirs
from football_tactical_ai.helpers.helperFunctions import normalize
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render
from football_tactical_ai.env.objects.pitch import Pitch

# Scenario classes
from football_tactical_ai.env.scenarios.singleAgent.move import OffensiveScenarioMoveSingleAgent
from football_tactical_ai.env.scenarios.singleAgent.shot import OffensiveScenarioShotSingleAgent
from football_tactical_ai.env.scenarios.singleAgent.view import OffensiveScenarioViewSingleAgent

# Evaluation configs
from football_tactical_ai.configs.configEvaluationSingleAgent import SCENARIOS, COMMON


# FIXED RESET WRAPPER
# Creates an environment with fixed starting positions
def fixed_env(env_class, pitch, attacker_m, defender_m, fps, max_steps):

    env = env_class(pitch=pitch, max_steps=max_steps, fps=fps)

    # TRAJECTORY STORAGE
    env.attacker_traj = []
    env.ball_traj = []

    # RESET WRAPPER
    original_reset = env.reset

    def fixed_reset(seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)

        # Reset trajectory buffer
        env.attacker_traj = []
        env.ball_traj = []

        att_norm = normalize(attacker_m[0], attacker_m[1])
        def_norm = normalize(defender_m[0], defender_m[1])

        # attacker + ball
        env.attacker.reset_position(att_norm)
        env.ball.set_position(att_norm)
        env.ball.set_owner(env.attacker)

        # defender
        env.defender.reset_position(def_norm)

        return env._get_obs(), {}

    env.reset = fixed_reset


    # STEP WRAPPER
    original_step = env.step

    def tracked_step(action):
        obs, reward, done, truncated, info = original_step(action)

        # Store attacker (x, y) every step
        ax, ay = env.attacker.get_position()
        env.attacker_traj.append((float(ax), float(ay)))

        # Store ball (x, y) every step
        bx, by = env.ball.get_position()
        env.ball_traj.append((float(bx), float(by)))

        return obs, reward, done, truncated, info

    env.step = tracked_step

    return env




# MAIN EVALUATION FUNCTION
def evaluate_single(scenario="move"):

    print(f"\nRunning evaluation: {scenario.upper()}")
    cfg = SCENARIOS[scenario]

    # Paths
    model_path      = cfg["model_path"]
    save_video_dir  = cfg["save_video_dir"]
    save_logs_dir   = cfg["save_logs_dir"]

    ensure_dirs(save_video_dir)
    ensure_dirs(save_logs_dir)

    fps = COMMON["fps"]
    max_steps = COMMON["max_steps"]

    pitch = Pitch()

    # Environment class selection
    env_class = {
        "move": OffensiveScenarioMoveSingleAgent,
        "shot": OffensiveScenarioShotSingleAgent,
        "view": OffensiveScenarioViewSingleAgent,
    }[scenario]

    # Load model
    print(f"→ Loading model: {model_path}")
    model = PPO.load(model_path)

    print("\n----------------------------------")
    print(f"   Total test cases: {len(cfg['test_cases'])}")
    print("----------------------------------")

    # Final containers
    summary_eval = {}     # ONLY evaluation statistics
    summary_traj = {}     # ONLY trajectories


    # Test each case in config
    for idx, case in enumerate(cfg["test_cases"]):

        print(f"\nTest case {idx+1}/{len(cfg['test_cases'])}: {case['name']}")

        # Init containers
        summary_traj[case["name"]] = {}
        summary_eval[case["name"]] = {}

        attacker_m = case["attacker_start"]
        defender_m = case["defender_start"]

        print(f" Attacker start (m): {attacker_m}")
        print(f" Defender start (m): {defender_m}")

        # Number of evaluation runs
        N = 20

        rewards = []
        extra_metrics = [] if scenario == "shot" or scenario == "view" else None


        # RUN THROUGH EPISODES
        for run in tqdm(range(N), desc="  Evaluation runs", unit="run"):

            env = fixed_env(env_class, pitch, attacker_m, defender_m, fps, max_steps)

            # Only FIRST run saves a video
            save_path = os.path.join(save_video_dir, f"{scenario}_eval_{idx+1}.mp4") if run == 0 else None

            # Evaluate scenario
            r = evaluate_and_render(
                model=model,
                env=env,
                pitch=pitch,
                save_path=save_path,
                episode=run+1,
                fps=fps,
                show_grid    = cfg["render"]["show_grid"],
                show_heatmap = cfg["render"]["show_heatmap"],
                show_rewards = cfg["render"]["show_rewards"],
                full_pitch   = cfg["render"]["full_pitch"],
                show_info    = cfg["render"]["show_info"],
                show_fov     = cfg["render"]["show_fov"],
            )


            # SAVE TRAJECTORIES
            run_key = f"run_{run+1}"
            summary_traj[case["name"]][run_key] = {
                "attacker": env.attacker_traj.copy(),
                "ball":     env.ball_traj.copy(),
            }



            # SAVE REWARD VALUE
            if isinstance(r, (float, int)):
                rewards.append(r)
            else:
                rewards.append(r["reward"])


            # SHOT-SPECIFIC METRICS
            if scenario == "shot":
                extra_metrics.append({
                    "valid_shot": env.valid_shot,
                    "shot_distance": env.shot_distance,
                    "shot_time": env.shot_time,
                    "shot_angle": env.shot_angle,
                    "shot_power": env.shot_power,
                    "reward_components": env.reward_components,
                })

            # VIEW-SPECIFIC METRICS
            if scenario == "view":
                extra_metrics.append({
                    "valid_shot": env.valid_shot,
                    "shot_distance": env.shot_distance,
                    "shot_time": env.shot_time,
                    "shot_angle": env.shot_angle,
                    "shot_power": env.shot_power,
                    "reward_components": env.reward_components,
                    "fov_valid_movements": env.fov_valid_movements,
                    "fov_invalid_movements": env.fov_invalid_movements,
                    "fov_valid_ratio": env.fov_valid_movements / max(env.fov_invalid_movements + env.fov_valid_movements, 1),
                    "invalid_shot_fov": env.invalid_shot_fov,
                    "valid_shot_ratio": env.valid_shot_ratio,
                })




        # STORE EVALUATION SUMMARY FOR THIS CASE
        summary_eval[case["name"]] = {
            "mean": float(np.mean(rewards)),
            "std":  float(np.std(rewards)),
            "min":  float(np.min(rewards)),
            "max":  float(np.max(rewards)),
            "runs": rewards,
        }

        if scenario == "shot":
            summary_eval[case["name"]]["shot_metrics"] = extra_metrics
        elif scenario == "view":
            summary_eval[case["name"]]["view_metrics"] = extra_metrics


        # PRINT SUMMARY
        print(f"\n → Mean reward: {np.mean(rewards):.3f}")
        print(f" → Std reward:  {np.std(rewards):.3f}")
        print(f" → Min reward:  {np.min(rewards):.3f}")
        print(f" → Max reward:  {np.max(rewards):.3f}")
        print(f" → N={N} runs")

        # SAVE TRAJECTORIES SEPARATELY
        traj_json_path = os.path.join(save_logs_dir, f"{scenario}_trajectories_{case['name']}.json")
        with open(traj_json_path, "w") as f:
            json.dump(summary_traj[case["name"]], f, indent=2)

        print(f" → Trajectories saved to {traj_json_path}")


    # SAVE GLOBAL AGGREGATED EVALUATION
    log_path = os.path.join(save_logs_dir, f"{scenario}_evaluation.json")
    with open(log_path, "w") as f:
        json.dump(summary_eval, f, indent=2)

    print("\n================ FINAL SUMMARY ================")
    print(f" Scenario: {scenario.upper()}")
    print(f" Results saved to: {log_path}")
    print("==============================================\n")

    return summary_eval

# RUN DIRECTLY
if __name__ == "__main__":
    #evaluate_single("move")
    evaluate_single("shot")
    #evaluate_single("view")