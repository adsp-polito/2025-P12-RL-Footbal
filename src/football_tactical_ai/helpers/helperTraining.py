import os
import matplotlib.pyplot as plt
import importlib
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.configs import configTrainSingleAgent as CFG_SA
from football_tactical_ai.helpers.helperFunctions import ensure_dirs
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render

import supersuit as ss
from football_tactical_ai.configs import configTrainMultiAgent as CFG_MA
from pettingzoo.utils.wrappers import BaseParallelWrapper
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi


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


def train_MultiAgent(scenario: str = "multiagent"):
    """
    Train a PPO agent on the specified multi-agent scenario.

    - Custom loop with tqdm to monitor episodes
    - Calls evaluate_and_render_multi() every eval_every episodes
    """

    # Load config and environment
    cfg = CFG_MA.SCENARIOS[scenario]
    env_path, class_name = cfg["env_class"].split(":")
    module = importlib.import_module(env_path)
    env_class = getattr(module, class_name)

    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))

    fps = cfg["fps"]
    episodes = cfg["episodes"]
    max_steps_per_episode = cfg["max_steps"]
    total_timesteps = episodes * max_steps_per_episode
    eval_every = cfg["eval_every"]

    pitch = Pitch()

    # Training env wrapped for SB3
    train_env = env_class(config=cfg["env_config"])
    vec_env = pettingzoo_env_to_vec_env_v1(train_env)
    vec_env = concat_vec_envs_v1(
        vec_env,
        num_vec_envs=cfg.get("num_envs", 1),
        base_class="stable_baselines3"
    )

    # PPO model
    ppo_kwargs = {**cfg["ppo"]}
    ppo_kwargs.pop("seed", None)
    model = PPO("MlpPolicy", vec_env, **ppo_kwargs)

    # Logging
    print("\n" + "=" * 100)
    print("Starting PPO Multi-Agent Training".center(100))
    print("=" * 100)
    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {episodes}")
    print(f"{'Evaluation every:':25} {eval_every} episodes")
    print(f"{'FPS:':25} {fps}")
    print(f"{'Steps per episode:':25} {max_steps_per_episode}")
    print(f"{'Total timesteps:':25} {total_timesteps}")
    print("=" * 100 + "\n")

    eval_rewards, eval_episodes = [], []

    print("\nStarting training...")

    # Training loop (episode by episode)
    for ep in trange(episodes, desc="Episodes Progress"):
        episode = ep + 1

        # Learn only for one episode worth of timesteps
        model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

        # Periodic evaluation
        if episode % eval_every == 0 or episode == 1:
            save_render = os.path.join(cfg["paths"]["save_render_dir"], f"episode_{episode}.mp4")
            eval_env = env_class(config=cfg["env_config"])   # fresh raw env for evaluation

            cumulative_reward = evaluate_and_render_multi(
                model, eval_env, pitch,
                save_path=save_render,
                episode=episode,
                fps=fps,
                **cfg["render"]
            )

            print(f"[Episode {episode}] Eval cumulative reward: {cumulative_reward:.2f}")
            eval_rewards.append(cumulative_reward)
            eval_episodes.append(episode)

    # Save model
    model.save(cfg["paths"]["save_model_path"])

    # Plot eval rewards
    plt.close('all')
    plt.figure(figsize=(10, 4))
    plt.plot(eval_episodes, eval_rewards, marker='o')
    plt.title(f"{scenario} - Cumulative Reward (Multi-Agent)", fontsize=16)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plot_path"])
    plt.show()
