import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
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
ray.init(ignore_reinit_error=True, log_to_driver=False)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

# Reduce RLlib/Ray log verbosity
logger = logging.getLogger("ray")
logger.setLevel(logging.ERROR)


def train_SingleAgent(scenario="move"):
    """
    Train a PPO agent on the specified single-agent scenario defined in the config.
    
    This function performs training, evaluation and rendering at defined intervals, 
    and saves both the model and plots of cumulative reward.

    Args:
        scenario (str): The key of the scenario to train (e.g., "move", "shot", "view").
    """
    # Load scenario configuration
    cfg = CFG_SA.SCENARIOS[scenario]

    # Load environment class dynamically
    env_path, class_name = cfg["env_class"].split(":")
    module = importlib.import_module(env_path)
    env_class = getattr(module, class_name)

    # Ensure output directories exist
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))

    # Unpack core parameters
    fps = cfg["fps"]
    episodes = cfg["episodes"]
    max_steps_per_episode = cfg["max_steps"]
    total_timesteps = episodes * max_steps_per_episode
    eval_every = cfg["eval_every"]

    # Create pitch
    pitch = Pitch()

    # Create training and evaluation environments
    env = Monitor(env_class(pitch=pitch, max_steps=max_steps_per_episode, fps=fps))
    eval_env = env_class(pitch=pitch, max_steps=max_steps_per_episode, fps=fps)

    # Create PPO model using scenario config
    model = PPO("MlpPolicy", env, **cfg["ppo"])

    # Initialize tracking for evaluation
    eval_rewards = []
    eval_episodes = []

    # Logging training configuration
    print("\n" + "=" * 100)
    print("Starting PPO Training".center(100))
    print("=" * 100)
    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {episodes}")
    print(f"{'Evaluation every:':25} {eval_every} episodes")
    print(f"{'FPS:':25} {fps}")
    print(f"{'Seconds per episode:':25} {cfg['seconds_per_episode']}")
    print(f"{'Steps per episode:':25} {max_steps_per_episode}")
    print(f"{'Total timesteps:':25} {total_timesteps}")
    print("=" * 100 + "\n")

    print("\nStarting training...")

    # Training loop with progress bar
    for ep in trange(episodes, desc="Episodes Progress"):
        episode = ep + 1  # true episode number starting from 1
        model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

        # Periodic evaluation and rendering
        if episode % eval_every == 0 or episode == 1:
            save_render = os.path.join(cfg["paths"]["save_render_dir"], f"episode_{episode}.mp4")
            cumulative_reward = evaluate_and_render(
                model,
                eval_env,
                pitch,
                save_path=save_render,
                episode=episode,
                fps=fps,
                **cfg["render"]
            )
            print(f"[Episode {episode}] Evaluation cumulative reward: {cumulative_reward:.4f}")
            eval_rewards.append(cumulative_reward)
            eval_episodes.append(episode)

    # Save trained model
    model.save(cfg["paths"]["save_model_path"])

    # Plot cumulative rewards
    plt.close('all')
    plt.figure(figsize=(10, 4))
    plt.plot(eval_episodes, eval_rewards, marker='o', linestyle='-')
    plt.title(f"{scenario.capitalize()} - Cumulative Reward", fontsize=16, fontweight='bold')
    plt.xlabel("Episodes", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Rewards", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plot_path"])
    plt.show()









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

    # Handle LR and entropy schedules (constant or schedule list)
    lr = cfg["rllib"]["lr"]  
    entropy_coeff = cfg["rllib"]["entropy_coeff"] 

    # RLlib PPO configuration
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=cfg["rllib"]["model"]["uses_new_env_api"],
            enable_env_runner_and_connector_v2=cfg["rllib"]["model"]["uses_new_env_api"])
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework(cfg["rllib"]["framework"])
        .training(
            lr=lr,
            gamma=cfg["rllib"]["gamma"],
            lambda_=cfg["rllib"]["lambda"],
            entropy_coeff=entropy_coeff,
            train_batch_size=cfg["rllib"]["train_batch_size"],
            minibatch_size=cfg["rllib"]["minibatch_size"],
            num_epochs=cfg["rllib"]["num_epochs"],
        )
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

    # LOGGING HEADER
    print("\n" + "=" * 100)
    print("Starting PPO Multi-Agent Training".center(100))
    print("=" * 100)

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
    print("=" * 100 + "\n")

    # TRAINING LOOP
    for ep in trange(1, cfg["episodes"] + 1, desc="Episodes Progress"):
        result = algo.train()

        # Evaluate at first, every eval_every, and last episode
        if ep % cfg["eval_every"] == 0 or ep in (1, cfg["episodes"]):
            eval_env = FootballMultiEnv(cfg["env_config"])
            save_render = os.path.join(cfg["paths"]["save_render_dir"], f"episode_{ep}.mp4")

            cumulative_reward = evaluate_and_render_multi(
                algo, eval_env, Pitch(),
                save_path=save_render,
                episode=ep,
                fps=cfg["fps"],
                policy_mapping_fn=policy_mapping_fn,
                **cfg["render"],
            )

            # Log results
            print(f"\n[Episode {ep}] Evaluation results:")
            for aid, rew in cumulative_reward.items():
                role = eval_env.players[aid].get_role()
                print(f"  {aid:10s} ({role}) -> {rew:.2f}")

    # SAVE MODEL
    checkpoint_dir = algo.save(os.path.abspath(cfg["paths"]["save_model_path"]))
    print(f"Model saved at {checkpoint_dir}")
