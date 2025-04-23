import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import threading
from tqdm import trange

# Project import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RLEnvironment.offensiveScenarioEnv import OffensiveScenarioEnv, X_MIN, X_MAX, Y_MIN, Y_MAX
from RLEnvironment.player import Player
from RLEnvironment.ball import Ball
from helpers.visuals import render_episode
from RLEnvironment.pitch import CELL_SIZE

# === Global Normalization ===
PITCH_WIDTH = X_MAX - X_MIN
PITCH_HEIGHT = Y_MAX - Y_MIN

# === Q-learning Hyperparameters ===
alpha = 0.1                   # Learning rate
gamma = 0.99                  # Discount factor for future rewards
epsilon = 1.0                 # Initial exploration rate
epsilon_decay = 0.995         # Decay rate per episode
epsilon_min = 0.05            # Minimum epsilon threshold

episodes = 5000               # Total training episodes
max_fps = 480                 # Maximum timesteps per episode
render_every = 100            # Render interval (episodes)

# === Grid dimensions based on cell size ===
GRID_WIDTH = (X_MAX - X_MIN) // CELL_SIZE
GRID_HEIGHT = (Y_MAX - Y_MIN) // CELL_SIZE

def to_grid(pos):
    """
    Converts a normalized (x, y) position into grid cell indices.
    This is used to discretize the environment for Q-table lookup.

    Args:
        pos (np.ndarray): Normalized coordinates [0, 1] x [0, 1]

    Returns:
        (int, int): Row and column indices for the grid
    """
    x, y = pos
    abs_x = x * PITCH_WIDTH + X_MIN
    abs_y = y * PITCH_HEIGHT + Y_MIN
    col = int(abs_x // CELL_SIZE)
    row = int(abs_y // CELL_SIZE)
    col = np.clip(col, 0, GRID_WIDTH - 1)
    row = np.clip(row, 0, GRID_HEIGHT - 1)
    return row, col

# Initialize Q-table
Q = np.zeros((GRID_HEIGHT, GRID_WIDTH, 9))

# Initialize environment
env = OffensiveScenarioEnv(render_mode="human")

# === Q-learning Main Loop ===
for ep in trange(episodes, desc="Training"):
    obs, _ = env.reset()
    total_reward = 0
    states = []

    for step in range(max_fps):
        row, col = to_grid(obs[0:2])

        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[row, col])

        # Step environment
        obs, reward, done, _, info = env.step(action)
        next_row, next_col = to_grid(obs[0:2])

        # Q-learning update
        td_target = reward + gamma * np.max(Q[next_row, next_col])
        td_error = td_target - Q[row, col, action]
        Q[row, col, action] += alpha * td_error
        total_reward += reward

        # Store snapshot for animation
        player_copy = Player(player_id=env.player.player_id)
        player_copy.set_position(env.player.get_position())
        player_copy.has_ball = env.player.has_ball
        player_copy.last_direction = env.player.last_direction.copy()

        ball_copy = Ball()
        ball_copy.set_position(env.ball.get_position())
        ball_copy.owner_id = env.ball.owner_id

        states.append({
            "player": player_copy,
            "ball": ball_copy,
            "opponents": []
        })

        if done:
            break

    # Epsilon decay
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # === Episode Visualization ===
    # === Episode Visualization ===
    # === Episode Visualization ===
    if ep % render_every == 0:
        print(f"\nEpisode {ep}: Total reward = {total_reward:.3f}")

        plt.figure()
        anim = render_episode(states, show_grid=True, show_cell_ids=False)
        plt.show(block=False)
        plt.pause(len(states) / 24)  # Show animation for its real-time duration
        plt.close('all')

env.close()