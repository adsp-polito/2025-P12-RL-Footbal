import os
import matplotlib.pyplot as plt
import importlib
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.configs import train_single_agent as CFG
from football_tactical_ai.helpers.helperFunctions import ensure_dirs
from football_tactical_ai.helpers.helper_evaluation import evaluate_and_render

def train_SingleAgent(scenario="move"):
    """
    Train a PPO agent on the specified single-agent scenario defined in the config.
    
    This function performs training, evaluation and rendering at defined intervals, 
    and saves both the model and plots of cumulative reward.

    Args:
        scenario (str): The key of the scenario to train (e.g., "move", "shot", "view").
    """
    # Load scenario configuration
    cfg = CFG.SCENARIOS[scenario]

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
