import sys
import os
import subprocess

# Ensure necessary packages are installed
required_packages = ["gymnasium", "stable-baselines3", "tqdm", "matplotlib", "numpy"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Add root project directory to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange  
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.offensiveScenarioMoveSingleAgent import OffensiveScenarioMoveSingleAgent
from helpers.visuals import render_episode

# Ensure the necessary folders exist for saving outputs (renders and models)
def ensure_dirs():
    os.makedirs("training/renders/singleAgentMove", exist_ok=True)
    os.makedirs("training/models", exist_ok=True)

# Evaluate the agent on a fresh environment and optionally render the episode as video
def evaluate_and_render(model, env, save_path=None):
    obs, _ = env.reset()
    done = False
    states = []

    # Run until episode termination
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Store a copy of the current state for rendering later
        attacker_copy = env.attacker.copy()
        defender_copy = env.defender.copy()
        ball_copy = env.ball.copy()

        states.append({
            "player": attacker_copy,
            "ball": ball_copy,
            "opponents": [defender_copy]
        })

    # If a path is specified, save the animation
    if save_path:
        render_episode(states, save_path=save_path, fps=24, full_pitch=True)

    # Return the final reward for monitoring progress
    return reward

def train_and_monitor(episodes=1000, seconds_per_episode=10, fps=24, eval_every_episodes=100):

    """
    Train a PPO agent on the OffensiveScenarioMove environment.

    Args:
        episodes (int): Total number of episodes for training.
        seconds_per_episode (int): Duration of each episode in seconds.
        fps (int): Frames per second (determines steps per episode).
        eval_every_episodes (int): Evaluate and render the policy every N episodes.
    """
     
    ensure_dirs()

    # Calculate steps based on fps and duration
    max_steps_per_episode = seconds_per_episode * fps
    total_timesteps = episodes * max_steps_per_episode

    # Print training configuration
    print(f"\n\n\n{'='*50}")
    print(f"Total timesteps (approx): {total_timesteps}, Steps per episode: {max_steps_per_episode}")

    # Create environment and model
    env = Monitor(OffensiveScenarioMoveSingleAgent())
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        seed=42,
        n_steps=240,  # Match max_steps
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        ent_coef=0.5,  # Encourage exploration
    )


    rewards = []
    episodes_list = []

    # Episodic loop with tqdm
    print("Starting training...")
    for episode in trange(episodes, desc="Episodes Progress"):
        model.learn(total_timesteps=max_steps_per_episode, reset_num_timesteps=False)

        if (episode + 1) % eval_every_episodes == 0:
            reward = evaluate_and_render(
                model,
                OffensiveScenarioMoveSingleAgent(),
                save_path=f"training/renders/singleAgentMove/episode_{episode + 1}.mp4"
            )
            rewards.append(reward)
            episodes_list.append(episode + 1)
            print(f"[Episode {episode + 1}] Eval reward: {reward:.2f}")

    # Save final model
    model.save("training/models/single_agent_move_model")

    # Plot reward progression
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_list, rewards, marker='o')
    plt.title("Evaluation Reward over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (Final of Evaluation Episode)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training/renders/training_progress.png")
    plt.show()



# Execute training if run directly
if __name__ == "__main__":
    train_and_monitor(
        episodes=10000,
        seconds_per_episode=10,
        fps=24,
        eval_every_episodes=2000
    )