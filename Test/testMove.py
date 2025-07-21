import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from env.offensiveScenarioMoveSingleAgent import OffensiveScenarioMoveSingleAgent
from helpers.visuals import render_episode
from time import time


# Create environment instance
env = OffensiveScenarioMoveSingleAgent()
obs, info = env.reset()

# Collect states for rendering later
states = []

done = False
while not done:
    # Sample random action from action space (continuous in [-1, 1])
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    # Copy current environment state (players and ball)
    attacker_copy = env.attacker.copy()
    defender_copy = env.defender.copy()
    ball_copy = env.ball.copy()

    # Store state for later visualization
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

# Ensure output directory exists for saving animation
os.makedirs('video', exist_ok=True)

# Measure rendering time
time_start = time()
print("Rendering episode...")

# Render the episode and save as .mp4
anim = render_episode(
    states,
    fps=24,
    full_pitch=True,
    show_grid=False,
    show_heatmap=True,
    show_rewards=False,
    env=env,
    save_path="video/testMove.mp4"
)

time_end = time()
print("Rendering complete. Animation saved in the 'video' directory.")
print(f"Rendering took {time_end - time_start:.2f} seconds.")