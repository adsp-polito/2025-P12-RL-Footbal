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

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.configs import configTrainMultiAgent as CFG_MA
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi

import warnings, logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)


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
    return FootballMultiEnv(config)


def train_MultiAgent(scenario: str = "multiagent"):
    """
    Train a PPO agent with RLlib on the specified multi-agent scenario.
    Scenario: attackers vs defenders (+ goalkeeper).
    
    This version uses a **single shared policy** across all agents (attackers, defenders, GK).
    Distinctions between roles are handled by the observation space,
    which includes a one-hot encoding of the agent's role (e.g., CF, LCB, GK).
    
    Advantages of this setup:
    - Efficient: one shared neural network, larger and more diverse training data.
    - Flexible: role one-hot embedding allows the policy to specialize behavior per role.
    - Coordinated: attackers/defenders/GK still "move as a team" since they optimize the same policy.
    
    Compatible with:
    - Gymnasium >= 0.29
    - Ray / RLlib >= 2.30
    """

    # Register the environment with RLlib
    register_env("football_multi_env", env_creator)

    # Load scenario configuration (paths, env settings, RLlib parameters, etc.)
    cfg = CFG_MA.SCENARIOS[scenario]
    env_path, class_name = cfg["env_class"].split(":")
    module = importlib.import_module(env_path)
    env_class = getattr(module, class_name)

    # Ensure output directories exist
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))

    # Extract key parameters
    fps = cfg["fps"]
    episodes = cfg["episodes"]
    max_steps_per_episode = cfg["env_config"]["max_steps"]
    eval_every = cfg["eval_every"]

    pitch = Pitch()

    # Initialize Ray (skip if already running)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Build a temporary environment to extract observation/action spaces
    base_env = FootballMultiEnv(cfg["env_config"])

    # POLICY DEFINITION (single shared policy with role embedding)
    # One shared policy across all roles.
    # The role-specific behavior is learned because the observation
    # includes a one-hot role embedding (self.roles_list).
    #
    # obs_space: identical across all agents
    # action_space: identical across all agents (7D vector)
    #
    policies = {
        "shared_policy": (
            None,
            base_env.observation_space("att_1"),
            base_env.action_space("att_1"),
            {}
        )
    }

    # RLlib PPO configuration
    config = (
        PPOConfig()
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework(cfg["rllib"]["framework"])
        .training(
            lr=cfg["rllib"]["lr"],
            gamma=cfg["rllib"]["gamma"],
            lambda_=cfg["rllib"]["lambda"],
            entropy_coeff=cfg["rllib"]["entropy_coeff"],
            train_batch_size=cfg["rllib"]["train_batch_size"],
            minibatch_size=cfg["rllib"]["minibatch_size"],
            num_epochs=cfg["rllib"]["num_epochs"],
            model=cfg["rllib"]["model"],  # neural network structure
        )
        .env_runners(
            num_env_runners=cfg["rllib"]["num_workers"],
            rollout_fragment_length=cfg["rllib"]["rollout_fragment_length"],
        )
        .multi_agent(
            policies=policies,
            # Mapping function: all agents → same shared policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],  # Only train the shared policy
        )
    )

    # Build the RLlib PPO algorithm
    algo = config.build_algo()

    # LOGGING (training configuration overview)
    print("\n" + "=" * 100)
    print("Starting PPO Multi-Agent Training (Shared Policy)".center(100))
    print("=" * 100)
    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {episodes}")
    print(f"{'Evaluation every:':25} {eval_every} episodes")
    print(f"{'FPS:':25} {fps}")
    print(f"{'Steps per episode:':25} {max_steps_per_episode}")
    print(f"{'Total timesteps:':25} {episodes * max_steps_per_episode}")
    print(f"{'Time per episode:':25} {cfg['seconds_per_episode']} seconds")
    print("\nRLlib Training Parameters:")
    for key, val in cfg["rllib"].items():
        print(f"  {key:25} {val}")
    print("=" * 100 + "\n")

    # TRAINING LOOP
    eval_rewards, eval_episodes = [], []

    for ep in trange(1, episodes + 1, desc="Episodes Progress", initial=1):
        result = algo.train()

        # Periodic evaluation + rendering
        if ep % eval_every == 0 or ep == 1:
            save_render = os.path.join(
                cfg["paths"]["save_render_dir"], f"episode_{ep}.mp4"
            )
            eval_env = FootballMultiEnv(cfg["env_config"])

            cumulative_reward = evaluate_and_render_multi(
                algo, eval_env, pitch,
                save_path=save_render,
                episode=ep,
                fps=fps,
                **cfg["render"],
            )

            # Print per-agent rewards
            print(f"\n[Episode {ep}] Evaluation results")
            for agent_id, rew in cumulative_reward.items():
                role = base_env.players[agent_id].get_role()
                print(f"  {agent_id:10s} -> {rew: .2f}")

            # Save raw rewards for later plotting
            eval_rewards.append(cumulative_reward)
            eval_episodes.append(ep)

    # SAVE MODEL CHECKPOINT
    save_model_path = os.path.abspath(cfg["paths"]["save_model_path"])
    checkpoint_dir = algo.save(save_model_path)
    print(f"Model saved at {checkpoint_dir}")

    # PLOT EVALUATION REWARDS
    if eval_rewards:
        plt.close('all')
        agent_ids = list(eval_rewards[0].keys())
        n_agents = len(agent_ids)

        fig, axes = plt.subplots(n_agents, 1, figsize=(10, 4 * n_agents), sharex=True)
        if n_agents == 1:
            axes = [axes]  # Ensure iterable

        for idx, agent_id in enumerate(agent_ids):
            rewards_agent = [d[agent_id] for d in eval_rewards]
            axes[idx].plot(eval_episodes, rewards_agent, marker='o')
            role = base_env.players[agent_id].get_role()
            axes[idx].set_title(f"{scenario} - Cumulative Reward for {agent_id} ({role})", fontsize=14)
            axes[idx].set_ylabel("Reward")
            axes[idx].grid(True)

        axes[-1].set_xlabel("Episodes")
        plt.tight_layout()
        plt.savefig(cfg["paths"]["plot_path"])
        plt.show()
