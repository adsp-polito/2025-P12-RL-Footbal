import sys
import os
import subprocess

# Ensure required packages are installed
required_packages = ["gymnasium", "stable-baselines3", "tqdm", "matplotlib", "numpy"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.scenarios.singleAgent.offensiveScenarioMoveSingleAgent import OffensiveScenarioMoveSingleAgent
from helpers.visuals import render_episode
from env.objects.pitch import Pitch


def ensure_dirs():
    """
    Ensure necessary directories for saving outputs exist.
    """
    os.makedirs("training/renders/singleAgentMove", exist_ok=True)
    os.makedirs("training/models", exist_ok=True)


def evaluate_and_render(model, env, pitch, save_path=None, episode=0,
                        show_grid=False, show_heatmap=False,
                        show_rewards=False, full_pitch=True):
    """
    Evaluate a trained model on a single episode and optionally render it as a video.

    Rewards per frame are collected only if save_path is provided to save memory.

    Args:
        model: Trained PPO agent.
        env: Evaluation environment instance.
        pitch: Pitch instance for rendering.
        save_path (str): Optional path to save the video.
        episode (int): Current episode number for logging purposes.
        show_grid (bool): Whether to draw grid lines on the pitch.
        show_heatmap (bool): Whether to color cells based on reward values.
        show_rewards (bool): Whether to display numeric reward values inside cells.
        full_pitch (bool): Whether to render the full pitch.

    Returns:
        float: The cumulative reward accumulated during this evaluation episode.
    """
    obs, _ = env.reset()
    done = False
    states = []
    rewards_per_frame = [] if save_path else None  # Collect rewards only if saving
    cumulative_reward = 0.0

    # Add initial state before any action is taken (frame 0)
    attacker_copy = env.attacker.copy()
    defender_copy = env.defender.copy()
    ball_copy = env.ball.copy()

    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

    if save_path:
        rewards_per_frame.append(0.0)  # No reward yet for frame 0


    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward

        # Copy current environment state for rendering
        attacker_copy = env.attacker.copy()
        defender_copy = env.defender.copy()
        ball_copy = env.ball.copy()

        states.append({
            "player": attacker_copy,
            "ball": ball_copy,
            "opponents": [defender_copy]
        })

        if save_path:
            rewards_per_frame.append(reward)

    if save_path:
        render_episode(
            states,
            pitch=pitch,
            save_path=save_path,
            fps=24,
            full_pitch=full_pitch,
            show_grid=show_grid,
            show_heatmap=show_heatmap,
            show_rewards=show_rewards,
            reward_grid=env.reward_grid,
            rewards_per_frame=rewards_per_frame,
            show_info=True
        )
        print(f"[Episode {episode}] Evaluation cumulative reward: {cumulative_reward:.4f}")

    return cumulative_reward


def train_and_monitor(episodes=1000, seconds_per_episode=10, fps=24,
                      eval_every_episodes=100, show_grid=False,
                      show_heatmap=False, show_rewards=False):
    """
    Train a PPO agent on the OffensiveScenarioMove environment.
    Save videos and collect evaluation rewards every 'eval_every_episodes' episodes for plotting.

    Args:
        episodes (int): Total number of training episodes.
        seconds_per_episode (int): Duration of each episode in seconds.
        fps (int): Simulation frequency (frames per second).
        eval_every_episodes (int): Frequency (in episodes) to save video and evaluation reward.
        show_grid (bool): Whether to draw grid lines on evaluation renders.
        show_heatmap (bool): Whether to show the reward heatmap in evaluation renders.
        show_rewards (bool): Whether to show reward values inside evaluation renders.
    """
    ensure_dirs()

    max_steps_per_episode = seconds_per_episode * fps
    total_timesteps = episodes * max_steps_per_episode

    print(f"\n{'=' * 50}")
    print(f"Total timesteps (approx.): {total_timesteps}")
    print(f"Steps per episode: {max_steps_per_episode}")

    # Create pitch instance once and assert consistency
    pitch = Pitch()
    pitch.assert_coordinates_match_helpers()

    # Create environments for training and evaluation, passing pitch explicitly to env
    env = Monitor(OffensiveScenarioMoveSingleAgent(pitch=pitch, max_steps=max_steps_per_episode, fps=fps))
    eval_env = OffensiveScenarioMoveSingleAgent(pitch=pitch, max_steps=max_steps_per_episode, fps=fps)

    # PPO agent configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        seed=42,
        n_steps=max_steps_per_episode,
        batch_size=48,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        ent_coef=0.01,
    )

    eval_rewards = []    # Store rewards for plotting
    eval_episodes = []   # Store episodes evaluated

    print("Starting training...")
    for episode in trange(1, episodes + 1, desc="Episodes Progress"):
        # Train for one full episode worth of timesteps
        model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

        # Evaluate and save video periodically, including the first episode
        if (episode) % eval_every_episodes == 0 or episode == 1:
            save_render = f"training/renders/singleAgentMove/episode_{episode}.mp4"
            cumulative_reward = evaluate_and_render(
                model,
                eval_env,
                pitch,
                save_path=save_render,
                episode=episode,
                show_grid=show_grid,
                show_heatmap=show_heatmap,
                show_rewards=show_rewards,
                full_pitch=True
            )
            eval_rewards.append(cumulative_reward)
            eval_episodes.append(episode)

    # Save the final trained model
    model.save("training/models/singleAgentMoveModel")

    # Close all matplotlib plots
    plt.close('all')

    # Plot evaluation rewards over time
    plt.figure(figsize=(10, 4))
    plt.plot(eval_episodes, eval_rewards, marker='o', linestyle='-')
    plt.title(f"Cumulative Reward every {eval_every_episodes} Episodes", fontsize=16, fontweight='bold')
    plt.xlabel("Episodes", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Rewards", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training/renders/singleAgentMove/SingleAgentMoveRewards.png")
    plt.show()


if __name__ == "__main__":
    train_and_monitor(
        episodes=2000,
        seconds_per_episode=10,
        fps=24,
        eval_every_episodes=200,
        show_grid=False,
        show_heatmap=True,
        show_rewards=False
    )