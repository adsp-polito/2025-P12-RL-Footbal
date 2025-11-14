import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import torch
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.configs import configTrainSingleAgent as CFG_SA
from football_tactical_ai.helpers.helperFunctions import ensure_dirs
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.configs import configTrainMultiAgent as CFG_MA
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi

import warnings
import ray
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

# Reduce RLlib/Ray log verbosity
logger = logging.getLogger("ray")
logger.setLevel(logging.ERROR)

ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False, num_gpus=1)







def train_SingleAgent(scenario="move"):
    """
    Train a PPO agent for a single-agent scenario (move, shot, view),
    using the configuration in configTrainSingleAgent.
    """

    # Load scenario configuration
    cfg = CFG_SA.SCENARIOS[scenario]

    # Load environment class dynamically
    env_path, class_name = cfg["env_class"].split(":")
    module = importlib.import_module(env_path)
    env_class = getattr(module, class_name)

    # Prepare directories
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))
    ensure_dirs(cfg["paths"]["rewards_dir"])

    # Core parameters
    fps = cfg["fps"]
    episodes = cfg["episodes"]
    eval_every = cfg["eval_every"]
    max_steps = cfg["max_steps"]
    total_timesteps = episodes * max_steps

    pitch = Pitch()

    # Create environments
    env = Monitor(env_class(pitch=pitch, max_steps=max_steps, fps=fps))
    eval_env = env_class(pitch=pitch, max_steps=max_steps, fps=fps)

    # SB3 PPO
    model = PPO("MlpPolicy", env, **cfg["ppo"])

    # Storage for rewards
    episode_rewards_log = []

    # HEADER
    print("\n" + "=" * 125)
    print("Starting PPO Single-Agent Training".center(125))
    print("=" * 125)

    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {episodes}")
    print(f"{'Evaluation every:':25} {eval_every}")
    print(f"{'Seconds per episode:':25} {cfg['seconds_per_episode']}")
    print(f"{'FPS:':25} {fps}")
    print(f"{'Steps per episode:':25} {max_steps}")
    print(f"{'Total timesteps:':25} {total_timesteps}")

    print("\nPPO Training Parameters:")
    for key, val in cfg["ppo"].items():
        print(f"  {key:25} {val}")
    print("=" * 125 + "\n")

    print("Starting training...\n")

    # TRAINING LOOP
    for ep in trange(1, episodes + 1, desc="Episodes Progress"):

        # One episode worth of timesteps
        model.learn(total_timesteps=max_steps, reset_num_timesteps=False)

        # FAST EVALUATION (no render)
        cumulative_reward_quick = evaluate_and_render(
            model=model,
            env=eval_env,
            pitch=pitch,
            save_path=None,
            episode=ep,
            fps=fps,
            eval_mode="fast",
            **cfg["render"],
        )

        episode_rewards_log.append({
            "episode": ep,
            "reward": cumulative_reward_quick
        })

        # FULL EVALUATION (with render)
        if ep % eval_every == 0 or ep in (1, episodes):

            save_render = os.path.join(cfg["paths"]["save_render_dir"],
                                       f"episode_{ep}.mp4")

            cumulative_reward_full = evaluate_and_render(
                model=model,
                env=eval_env,
                pitch=pitch,
                save_path=save_render,
                episode=ep,
                fps=fps,
                eval_mode="full",
                **cfg["render"],
            )

            print(f"[Episode {ep}] Eval reward = {cumulative_reward_full:.4f}")

    # SAVE REWARD LOG
    rewards_path = os.path.join(cfg["paths"]["rewards_dir"], "rewards.json")
    with open(rewards_path, "w") as f:
        json.dump(episode_rewards_log, f, indent=2)

    print(f"\nSaved reward log → {rewards_path}")

    # SAVE MODEL
    model.save(cfg["paths"]["save_model_path"])
    print(f"Model saved → {cfg['paths']['save_model_path']}")











def env_creator(config):
    """
    Create a FootballMultiEnv instance for RLlib.
    """
    env_cfg = config.get("env_config", config)
    return FootballMultiEnv(env_cfg)



def make_schedule_fn(schedule: list):
    """
    Convert a schedule [[t0, v0], [t1, v1], ...] into a function f(timestep) → value.

    Args:
        schedule (list): List of [timestep, value] pairs.
    
    Returns:
        function: A function that interpolates the value according to the timestep.
    """
    def fn(timestep: int):
        # Linear interpolation between schedule points
        for i in range(len(schedule) - 1):
            t0, v0 = schedule[i]
            t1, v1 = schedule[i + 1]
            if t0 <= timestep < t1:
                alpha = (timestep - t0) / float(t1 - t0)
                return v0 + alpha * (v1 - v0)
        return schedule[-1][1]  # after the last point → last value
    return fn


def train_MultiAgent(scenario: str = "multiagent", role_based: bool = False):
    """
    Train a PPO multi-agent setup with RLlib (new API stack).

    Args:
        scenario (str): Name of the scenario defined in CFG_MA.SCENARIOS.
        role_based (bool): 
            - If True → one shared policy per role (attacker, defender, goalkeeper).
            - If False → one independent policy per agent (e.g. att_1, att_2, def_1, gk_1).
    """

    # Register env with RLlib
    register_env("football_multi_env", env_creator)

    # Load scenario configuration
    cfg = CFG_MA.SCENARIOS[scenario]
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))

    # Build a base environment to extract obs/action spaces
    base_env = FootballMultiEnv(cfg["env_config"])

    # Define policies and mapping
    if role_based:
        # Shared policy per role type (attacker, defender, goalkeeper)
        policies = {}

        if base_env.attacker_ids:
            policies["attacker_policy"] = (
                None,
                base_env.observation_space(base_env.attacker_ids[0]),
                base_env.action_space(base_env.attacker_ids[0]),
                {},
            )
        if base_env.defender_ids:
            policies["defender_policy"] = (
                None,
                base_env.observation_space(base_env.defender_ids[0]),
                base_env.action_space(base_env.defender_ids[0]),
                {},
            )
        if base_env.gk_ids:
            policies["goalkeeper_policy"] = (
                None,
                base_env.observation_space(base_env.gk_ids[0]),
                base_env.action_space(base_env.gk_ids[0]),
                {},
            )

        def policy_mapping_fn(agent_id, *args, **kwargs):
            if agent_id.startswith("att"):
                return "attacker_policy"
            if agent_id.startswith("def"):
                return "defender_policy"
            if agent_id.startswith("gk"):
                return "goalkeeper_policy"
            raise ValueError(f"Unknown agent_id: {agent_id}")

    else:
        # Independent policy for each agent
        policies = {
            agent_id: (
                None,
                base_env.observation_space(agent_id),
                base_env.action_space(agent_id),
                {},
            )
            for agent_id in base_env.agents
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id


    # RLlib PPO configuration
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=cfg["rllib"]["model"]["uses_new_env_api"],
            enable_env_runner_and_connector_v2=cfg["rllib"]["model"]["uses_new_env_api"])
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework(cfg["rllib"]["framework"])
        .training(
            lr=cfg["rllib"]["lr"],
            #lr_schedule=cfg["rllib"].get("lr_schedule", None),
            gamma=cfg["rllib"]["gamma"],
            lambda_=cfg["rllib"]["lambda"],
            entropy_coeff=cfg["rllib"]["entropy_coeff"],
            #entropy_coeff_schedule=cfg["rllib"].get("entropy_coeff_schedule", None),
            clip_param=cfg["rllib"]["clip_param"],
            vf_clip_param=cfg["rllib"].get("vf_clip_param", None),
            vf_loss_coeff=cfg["rllib"].get("vf_loss_coeff", None),
            grad_clip=cfg["rllib"].get("grad_clip", None),
            train_batch_size=cfg["rllib"]["train_batch_size"],
            minibatch_size=cfg["rllib"]["minibatch_size"],
            num_epochs=cfg["rllib"]["num_epochs"],
        )
        .resources(num_gpus=cfg["rllib"].get("num_gpus", 0))
        .rl_module(model_config=cfg["rllib"]["model"])  
        .env_runners(
            num_env_runners=cfg["rllib"]["num_workers"],
            rollout_fragment_length=cfg["rllib"]["rollout_fragment_length"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
    )

    # Build PPO algorithm
    algo = config.build_algo()

    # Force policies on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nForcing all RLlib modules to device: {device} ...")

    try:
        multi_module = algo.env_runner.module
        for pid, submodule in multi_module._rl_modules.items():
            try:
                submodule.model.to(device)
                print(f"  → moved '{pid}' to {device}")
            except Exception as e:
                print(f"  → failed to move '{pid}': {e}")
    except Exception as e:
        print(f"Could not move RLlib modules: {e}")

    # LOGGING HEADER
    print("\n" + "=" * 125)
    print("Starting PPO Multi-Agent Training".center(125))
    print("=" * 125)

    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {cfg['episodes']}")
    print(f"{'Time per episode:':25} {cfg['seconds_per_episode']} seconds")
    print(f"{'Evaluation every:':25} {cfg['eval_every']} episodes")
    print(f"{'FPS:':25} {cfg['fps']}")
    print(f"{'Steps per episode:':25} {cfg['env_config']['max_steps']}")
    print(f"{'Total timesteps:':25} {cfg['episodes'] * cfg['env_config']['max_steps']}")

    print("\nRLlib Training Parameters:")
    for key, val in cfg["rllib"].items():
        print(f"  {key:25} {val}")
    print("=" * 125 + "\n")


    # List to store (all) episode rewards
    episode_rewards_log = []

    # TRAINING LOOP
    for ep in trange(1, cfg["episodes"] + 1, desc="Episodes Progress"):
        result = algo.train()

        # Always evaluate INSTANTLY without render to get per-agent rewards
        eval_env_quick = FootballMultiEnv(cfg["env_config"])
        cumulative_reward_quick = evaluate_and_render_multi(
            model=algo,
            env=eval_env_quick,
            pitch=Pitch(),
            episode=ep,
            fps=cfg["fps"],
            policy_mapping_fn=policy_mapping_fn,
            eval_mode="fast",      # <-- FAST MODE
        )

        # save episode rewards
        episode_rewards_log.append({
            "episode": ep,
            "rewards": cumulative_reward_quick
        })

        # Evaluate at first, every eval_every, and last episode
        if ep % cfg["eval_every"] == 0 or ep in (1, cfg["episodes"]):
            eval_env = FootballMultiEnv(cfg["env_config"])
            save_render = os.path.join(cfg["paths"]["save_render_dir"], f"episode_{ep}.mp4")

            cumulative_reward = evaluate_and_render_multi(
                model=algo,
                env=eval_env,
                pitch=Pitch(),
                save_path=save_render,
                episode=ep,
                fps=cfg["fps"],
                policy_mapping_fn=policy_mapping_fn,
                eval_mode="full",     # <-- FULL MODE
                **cfg["render"],
            )

            # Log results
            print(f"\n[Episode {ep}] Evaluation results:")
            for aid, rew in cumulative_reward.items():
                role = eval_env.players[aid].get_role()
                print(f"  {aid:10s} ({role}) -> {rew:.2f}")


    # SAVE REWARD LOG
    save_plot_dir = os.path.abspath(cfg["paths"]["rewards_dir"])

    # Ensure directory exists
    ensure_dirs(save_plot_dir)

    rewards_path = os.path.join(save_plot_dir, "rewards.json")
    with open(rewards_path, "w") as f:
        json.dump(episode_rewards_log, f, indent=2)

    print(f"Saved reward log → {rewards_path}")

    # SAVE MODEL
    checkpoint_dir = algo.save(os.path.abspath(cfg["paths"]["save_model_path"]))
    print(f"Model saved at {checkpoint_dir}")
