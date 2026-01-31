import os
import json
import numpy as np
import torch
from tqdm import tqdm

from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
from football_tactical_ai.helpers.helperTraining import ensure_dirs
from football_tactical_ai.helpers.helperFunctions import normalize
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi
from football_tactical_ai.env.objects.pitch import Pitch

from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.configs.configEvaluationMultiAgent import SCENARIOS_MULTI, COMMON

def multi_agent_action_fn(obs_dict, algo):
    """
    Compute actions for all agents in a multi-agent RLlib environment

    This function is called at every environment step (or during rendering)
    to generate an action for each active agent
    It supports the typical RLlib multi-policy setup where each agent 
    has a dedicated policy with the same name (e.g. 'att_1', 'att_2', 'def_1')
    """

    actions = {}

    for agent_id, obs in obs_dict.items():

        # SAFETY CHECK 1: observation must not be None
        if obs is None:
            # If an agent has no valid observation
            # we produce a zero action or fallback action.
            actions[agent_id] = np.zeros( algo.get_policy(agent_id).action_space.shape )
            continue

        # SAFETY CHECK 2: verify policy exists
        if agent_id not in algo.workers.local_worker().policy_map:
            raise KeyError(
                f"[multi_agent_action_fn] No policy found for agent_id '{agent_id}'. "
                f"Available policies: {list(algo.workers.local_worker().policy_map.keys())}"
            )

        # ACTION COMPUTATION
        policy = algo.get_policy(agent_id)

        action, _, _ = policy.compute_single_action(
            obs,
            explore=False   # no stochasticity during evaluation
        )

        actions[agent_id] = action

    return actions



def fixed_env_multi(pitch, attackers_m, fps, max_steps):
    """
    Create a multi-agent FootballMultiEnv with deterministic initial positions

    This wrapper is used for evaluation test cases, where attackers and defender
    must start from predefined real-world coordinates. The positions are 
    normalized to the environment coordinate system and applied AFTER the 
    environment's internal reset(), ensuring a clean and reproducible episode

    Additionally, trajectory buffers are attached to the environment so that
    agent and ball movement can be logged step-by-step for post-analysis
    """

    # 1. Base environment creation
    env = FootballMultiEnv({
        "fps": fps,
        "max_steps": max_steps,
        "time_step": 1.0 / fps,
        "n_attackers": 2,
        "n_defenders": 0,
        "include_goalkeeper": True,
        "randomize_positions": False,   # critical for deterministic evaluation
    })

    # 2. Initialize trajectory buffers (for logging)
    env.traj_A1   = []
    env.traj_A2   = []
    env.traj_ball = []

    # Keep reference to the original reset() method
    original_reset = env.reset

    # 3. Override RESET to enforce deterministic starting positions
    def fixed_reset(seed=None, options=None):
        """
        Reset environment, apply deterministic starting positions, 
        and reinitialize trajectory buffers.
        """

        # Allow environment to reset normally first
        obs, info = original_reset(seed=seed, options=options)

        # Clear trajectory logs for the new episode
        env.traj_A1.clear()
        env.traj_A2.clear()
        env.traj_ball.clear()

        # Normalize input positions (meters → [0,1] normalized space)
        A1_norm = normalize(*attackers_m[0])
        A2_norm = normalize(*attackers_m[1])

        # Apply manual player placement
        env.players["att_1"].reset_position(A1_norm)
        env.players["att_2"].reset_position(A2_norm)

        # Give ball possession to attacker 1
        env.ball.set_position(A1_norm)
        env.ball.set_owner("att_1")

        # Return updated observations
        return obs, {}

    # Inject the wrapper reset into the environment
    env.reset = fixed_reset

    # 4. Override STEP to record trajectories at each simulation step
    original_step = env.step

    def tracked_step(actions):
        """
        Call the original step(), but additionally record each agent's and the 
        ball's positions into trajectory buffers.
        """

        obs, rewards, terminated, truncated, info = original_step(actions)

        # Log attacker 1 coordinates
        x1, y1 = env.players["att_1"].get_position()
        env.traj_A1.append((float(x1), float(y1)))

        # Log attacker 2 coordinates
        x2, y2 = env.players["att_2"].get_position()
        env.traj_A2.append((float(x2), float(y2)))

        # Log ball coordinates
        bx, by = env.ball.get_position()
        env.traj_ball.append((float(bx), float(by)))

        return obs, rewards, terminated, truncated, info

    # Inject the wrapper step into the environment
    env.step = tracked_step

    return env



# MAIN EVALUATION FUNCTION
def evaluate_multi(pos1, pos2, scenario="what_if"):
    """
    Evaluate a trained multi-agent PPO model (RLlib) over predefined what-if test case.

    The evaluation pipeline performs:
        - deterministic resets using fixed_env_multi()
        - rollout of N episodes per test-case
        - multi-policy RLlib action selection (one policy per agent_id)
        - reward statistics (aggregated over all steps and runs)
        - per-run shooting, passing, and possession statistics
        - episode-level outcomes
        - trajectory storage (attacker 1, attacker 2, ball)
        - optional video rendering for the first episode of each test case
    """

    cfg = SCENARIOS_MULTI[scenario]

    print(f"\n=== MULTI-AGENT EVALUATION: {scenario.upper()} ===")

    # Global configuration
    fps       = COMMON["fps"]
    max_steps = COMMON["max_steps"]

    model_path     = cfg["model_path"]      # IMPORTANT: must be a RLlib checkpoint directory (absolute path)
    save_video_dir = cfg["save_video_dir"]
    save_logs_dir  = cfg["save_logs_dir"]

    ensure_dirs(save_video_dir)
    ensure_dirs(save_logs_dir)

    pitch = Pitch()

    # Load trained RLlib multi-agent model
    tune.register_env("football_multi_env", lambda cfg: FootballMultiEnv(cfg))
    print(f"Loading RLlib checkpoint: {model_path}")
    algo = Algorithm.from_checkpoint(model_path)

    summary_eval = {}
    summary_traj = {}

    test_cases = [
        {
            "name": "what_if",
            "attackers_start": [
                (pos1[0], pos1[1]),  
                (pos2[0], pos2[1]),  
            ],
        },
    ]
    print(f"Total test cases: {len(test_cases)}")

    # Evaluation loop over all predefined test cases
    for idx, case in enumerate(test_cases):

        name = case["name"]
        A_m  = case["attackers_start"]

        print(f"\n→ Test case {idx+1}/{len(test_cases)}: {name}")
        print(f"   Attacker starts (m): {A_m}")

        # Storage for results of this test case
        summary_eval[name] = {}
        summary_traj[name] = {}

        # Reward storage (aggregated over ALL runs and ALL steps)
        rewards_per_agent = {
            "att_1": [],
            "att_2": [],
        }

        # Per-run metric containers
        per_run_shooting   = []
        per_run_passing    = []
        per_run_possession = []

        # Episode outcomes aggregated over runs
        episodes_with_goal = 0
        episodes_out       = 0

        N = 20  # episodes per test-case

        # Run N evaluation episodes
        for run in tqdm(range(N), desc="runs"):

            # Create environment with deterministic reset wrapper
            env = fixed_env_multi(
                pitch       = pitch,
                attackers_m = A_m,
                fps         = fps,
                max_steps   = max_steps,
            )

            # VIDEO RENDERING for the first episode ONLY
            if run == 0:
                evaluate_and_render_multi(
                    model = algo,
                    env       = env,
                    pitch     = pitch,
                    save_path = os.path.join(save_video_dir, f"multi_eval_{name}.mp4"),
                    episode   = 1,
                    fps       = fps,
                    show_grid    = cfg["render"]["show_grid"],
                    show_heatmap = cfg["render"]["show_heatmap"],
                    show_rewards = cfg["render"]["show_rewards"],
                    full_pitch   = cfg["render"]["full_pitch"],
                    show_names   = cfg["render"]["show_names"],
                    policy_mapping_fn = lambda agent_id: agent_id
                )

            # ROLLOUT LOOP (MANUAL STEP-BY-STEP EVALUATION)
            obs, _ = env.reset()
            done = False

            # Per-run counters (reset at each episode)
            shots_A1 = shots_A2 = 0
            goals_A1 = goals_A2 = 0

            attempt_A1_A2 = attempt_A2_A1 = 0
            complete_A1_A2 = complete_A2_A1 = 0

            poss_A1 = poss_A2 = 0
            lost_A1 = lost_A2 = 0

            while not done:

                actions = {}

                for agent_id in env.agents:

                    # 1. Get the policy module for this agent
                    module = algo.get_module(agent_id)

                    # 2. Prepare the observation as a batch tensor (1, obs_dim)
                    obs_arr = np.array(obs[agent_id], dtype=np.float32).reshape(1, -1)
                    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32)

                    # 3. Forward inference (new RLlib API)
                    with torch.no_grad():
                        out = module.forward_inference({"obs": obs_tensor})

                    # 4. Get the logits
                    dist_inputs = out.get("action_dist_inputs", out.get("logits"))

                    # 5. Build the policy action distribution
                    dist_class = module.get_train_action_dist_cls()
                    dist = dist_class.from_logits(dist_inputs)

                    # 6. Deterministic action (greedy)
                    if hasattr(dist, "loc"):
                        action = torch.tanh(dist.loc)        # continuous
                    else:
                        action = torch.argmax(dist_inputs, dim=-1)  # discrete

                    # Convert to numpy
                    action = action.cpu().numpy().flatten()

                    # CLIP for safety
                    low, high = env.action_space(agent_id).low, env.action_space(agent_id).high
                    action = np.clip(action, low, high)

                    # 7. Save the action
                    actions[agent_id] = action

                # Take step
                obs, rewards, terminated, truncated, infos = env.step(actions)

                #  Collect reward statistics (aggregated)
                rewards_per_agent["att_1"].append(float(rewards["att_1"]))
                rewards_per_agent["att_2"].append(float(rewards["att_2"]))

                # Shooting statistics (per-run)
                if infos["att_1"].get("shot_attempted"):
                    shots_A1 += 1
                if infos["att_2"].get("shot_attempted"):
                    shots_A2 += 1

                if infos["att_1"].get("goal_scored"):
                    goals_A1 += 1
                if infos["att_2"].get("goal_scored"):
                    goals_A2 += 1

                # Passing statistics (per-run)
                if infos["att_1"].get("pass_attempted") and infos["att_1"].get("pass_to") == "att_2":
                    attempt_A1_A2 += 1
                if infos["att_1"].get("pass_completed") and infos["att_1"].get("pass_to") == "att_2":
                    complete_A1_A2 += 1

                if infos["att_2"].get("pass_attempted") and infos["att_2"].get("pass_to") == "att_1":
                    attempt_A2_A1 += 1
                if infos["att_2"].get("pass_completed") and infos["att_2"].get("pass_to") == "att_1":
                    complete_A2_A1 += 1

                # Possession statistics (per-run)
                if infos["att_1"].get("has_ball"):
                    poss_A1 += 1
                if infos["att_2"].get("has_ball"):
                    poss_A2 += 1

                if infos["att_1"].get("possession_lost"):
                    lost_A1 += 1
                if infos["att_2"].get("possession_lost"):
                    lost_A2 += 1

                # Episode outcomes (aggregated)
                if infos["att_1"].get("goal_team") is not None:
                    episodes_with_goal += 1
                if infos["att_1"].get("ball_out_by") is not None:
                    episodes_out += 1

                done = terminated["__all__"] or truncated["__all__"]

            # Store per-run statistics for this episode
            per_run_shooting.append({
                "shots_A1": shots_A1,
                "shots_A2": shots_A2,
                "goals_A1": goals_A1,
                "goals_A2": goals_A2,
            })

            per_run_passing.append({
                "attempted_A1_to_A2": attempt_A1_A2,
                "attempted_A2_to_A1": attempt_A2_A1,
                "completed_A1_to_A2": complete_A1_A2,
                "completed_A2_to_A1": complete_A2_A1,
            })

            per_run_possession.append({
                "poss_time_A1": poss_A1,
                "poss_time_A2": poss_A2,
                "lost_possession_A1": lost_A1,
                "lost_possession_A2": lost_A2,
            })

            # Store trajectories for this run
            summary_traj[name][f"run_{run+1}"] = {
                "A1":   env.traj_A1,
                "A2":   env.traj_A2,
                "ball": env.traj_ball
            }

        # Build final summary structure for this test case
        summary_eval[name] = {
            "totals": {
                "agents": {
                    "att_1": {
                        "mean": float(np.mean(rewards_per_agent["att_1"])),
                        "std":  float(np.std(rewards_per_agent["att_1"])),
                        "min":  float(np.min(rewards_per_agent["att_1"])),
                        "max":  float(np.max(rewards_per_agent["att_1"])),
                    },
                    "att_2": {
                        "mean": float(np.mean(rewards_per_agent["att_2"])),
                        "std":  float(np.std(rewards_per_agent["att_2"])),
                        "min":  float(np.min(rewards_per_agent["att_2"])),
                        "max":  float(np.max(rewards_per_agent["att_2"])),
                    },
                },
                "episode_outcomes": {
                    "episodes_with_goal": episodes_with_goal,
                    "episodes_out_of_play": episodes_out,
                }
            },
            "runs": {}
        }

        # Fill per-run metrics
        for run_idx in range(N):
            summary_eval[name]["runs"][f"run_{run_idx+1}"] = {
                "shooting_stats":   per_run_shooting[run_idx],
                "passing_stats":    per_run_passing[run_idx],
                "possession_stats": per_run_possession[run_idx],
            }

        # Save trajectories to file
        traj_path = os.path.join(save_logs_dir, f"multi_traj_{name}.json")
        with open(traj_path, "w") as f:
            json.dump(summary_traj[name], f, indent=2)

        print(f"   → Trajectories saved to {traj_path}")

        # Print summary statistics for all attackers (generic)
        print("\n   === STATISTICS (AGGREGATED REWARDS) ===")
        for agent_id, stats in summary_eval[name]["totals"]["agents"].items():
            print(
                f"   {agent_id.upper()} → "
                f"mean: {stats['mean']:.3f}, "
                f"std: {stats['std']:.3f}, "
                f"min: {stats['min']:.3f}, "
                f"max: {stats['max']:.3f}"
            )
        print()

    # Save global summary
    eval_path = os.path.join(save_logs_dir, "multi_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(summary_eval, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"Results saved to: {eval_path}")
    print("==============================================\n")

    return summary_eval




if __name__ == "__main__":
    evaluate_multi((50, 30), (60, 40),"what_if")
