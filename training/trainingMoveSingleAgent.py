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

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.offensiveScenarioMoveSingleAgent import OffensiveScenarioMoveSingleAgent
from helpers.visuals import render_episode


# Ensure output directories exist for saving renders and models
def ensure_dirs():
    os.makedirs("training/renders/singleAgentMove", exist_ok=True)
    os.makedirs("training/models", exist_ok=True)


def evaluate_and_render(model, env, save_path=None, episode=0,
                        show_grid=False, show_heatmap=False, show_rewards=False, full_pitch=True):
    """
    Runs a single evaluation episode and optionally saves it as a video.

    Args:
        model: Trained PPO agent.
        env: Evaluation environment instance.
        save_path (str): File path for saving the animation.
        episode (int): Current episode number, used for logging.
        show_grid (bool): Whether to draw grid lines on the pitch.
        show_heatmap (bool): Whether to color cells based on reward.
        show_rewards (bool): Whether to display reward values inside cells.
        full_pitch (bool): Whether to render the full pitch.

    Returns:
        float: Final reward from the episode.
    """
    obs, _ = env.reset()
    done = False
    states = []

    # Collect environment states until episode termination
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Deep copy environment state for rendering (players and ball)
        attacker_copy = env.attacker.copy()
        defender_copy = env.defender.copy()
        ball_copy = env.ball.copy()

        states.append({
            "player": attacker_copy,
            "ball": ball_copy,
            "opponents": [defender_copy]
        })

    # Render the episode and optionally save as video
    if save_path:
        render_episode(
            states,
            save_path=save_path,
            fps=24,
            full_pitch=full_pitch,
            show_grid=show_grid,
            show_heatmap=show_heatmap,
            show_rewards=show_rewards,
            env=env
        )
        print(f"[Episode {episode + 1}] Evaluation reward: {reward:.4f}")

    return reward


def train_and_monitor(episodes=1000, seconds_per_episode=10, fps=24,
                      eval_every_episodes=100, show_grid=False, show_heatmap=False, show_rewards=False):
    """
    Train a PPO agent on the OffensiveScenarioMove environment.
    Periodically evaluates the policy and saves animations.

    Args:
        episodes (int): Total number of training episodes.
        seconds_per_episode (int): Duration of each episode in seconds.
        fps (int): Frames per second for each episode.
        eval_every_episodes (int): Frequency (in episodes) to render evaluation.
        show_grid (bool): Whether to draw grid lines on evaluation renders.
        show_heatmap (bool): Whether to show reward heatmap in evaluation renders.
        show_rewards (bool): Whether to show reward values in evaluation renders.
    """
    ensure_dirs()

    max_steps_per_episode = seconds_per_episode * fps
    total_timesteps = episodes * max_steps_per_episode

    print(f"\n{'=' * 50}")
    print(f"Total timesteps (approx.): {total_timesteps}")
    print(f"Steps per episode: {max_steps_per_episode}")

    # Monitored environments for training and evaluation
    env = Monitor(OffensiveScenarioMoveSingleAgent(max_steps=max_steps_per_episode))
    eval_env = OffensiveScenarioMoveSingleAgent(max_steps=max_steps_per_episode)

    # PPO model configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        seed=42,
        n_steps=fps * seconds_per_episode,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        ent_coef=0.01,  # Entropy coefficient to encourage exploration
    )

    all_rewards = []
    episodes_list = []

    print("Starting training...")
    for episode in trange(episodes, desc="Episodes Progress"):
        model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

        # Periodically evaluate and render
        save_render = None
        if (episode + 1) % eval_every_episodes == 0:
            save_render = f"training/renders/singleAgentMove/episode_{episode + 1}.mp4"

        reward = evaluate_and_render(
            model,
            eval_env,
            save_path=save_render,
            episode=episode,
            show_grid=show_grid,
            show_heatmap=show_heatmap,
            show_rewards=show_rewards,
            full_pitch=True
        )
        all_rewards.append(reward)
        episodes_list.append(episode + 1)

    # Save final trained model
    model.save("training/models/single_agent_move_model")

    # Plot evaluation rewards over episodes
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_list, all_rewards, marker='.')
    plt.title("Evaluation Reward over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (Evaluation)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training/renders/singleAgentMove/single_agent_training_progress.png")
    plt.show()


# Execute training when run directly
if __name__ == "__main__":
    train_and_monitor(
        episodes=20000,
        seconds_per_episode=10,
        fps=24,
        eval_every_episodes=2000,
        show_grid=False,
        show_heatmap=True,
        show_rewards=False
    )
