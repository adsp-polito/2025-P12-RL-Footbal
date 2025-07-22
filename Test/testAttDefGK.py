import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from players.playerGoalkeeper import PlayerGoalkeeper
from env.objects.ball import Ball
from helpers.visuals import render_episode
from env.objects.pitch import X_MIN, Y_MIN, X_MAX, Y_MAX
import numpy as np
from helpers.helperFunctions import normalize
from time import time

# Create players and ball
attacker = PlayerAttacker()
defender = PlayerDefender()
goalkeeper = PlayerGoalkeeper()
ball = Ball()

# Set initial positions (normalized coordinates)
attacker.reset_position(*normalize(60, 40))    # Center field
defender.reset_position(*normalize(100, 40))   # Further behind
goalkeeper.reset_position(*normalize(120, 40)) # In goal center
ball.position = normalize(60, 40)

# Simulation settings
time_per_step = 1 / 24  # 24 FPS
x_range = X_MAX - X_MIN
y_range = Y_MAX - Y_MIN

# Store states frame-by-frame
states = []

for frame in range(240):

    # Attacker moves randomly towards the goal
    action_att = np.array([
        np.random.uniform(1, 1),   # Always towards the opponent's goal (right)
        np.random.uniform(-1, 1)   # Random lateral movement
    ])
    attacker.move_with_action(action_att, time_per_step, x_range, y_range)

    # Defender moves towards the attacker (basic pursuit behavior)
    att_pos = np.array(attacker.get_position())
    def_pos = np.array(defender.get_position())
    direction = att_pos - def_pos
    distance_to_att = np.linalg.norm(direction)

    if distance_to_att > 0.01:
        direction = direction / distance_to_att
    else:
        direction = np.array([0.0, 0.0])

    defender.move_with_action(direction, time_per_step, x_range, y_range)

    # Goalkeeper pseudo-random small movement (noise simulation)
    action_gk = np.array([
        np.random.uniform(-1, 1),
        np.random.uniform(-1, 1)
    ])
    goalkeeper.move_with_action(action_gk, time_per_step, x_range, y_range)

    # Ball follows attacker (basic dribbling offset)
    ball.position = (attacker.get_position()[0] + 0.01, attacker.get_position()[1])

    # Deep copy of current state for rendering (players and ball)
    attacker_copy = attacker.copy()
    defender_copy = defender.copy()
    goalkeeper_copy = goalkeeper.copy()

    ball_copy = Ball()
    ball_copy.position = ball.position

    # Save current state
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy, goalkeeper_copy]
    })

# Ensure output directory exists for saving animation
os.makedirs('videoTest', exist_ok=True)

time_start = time()
print("Rendering episode...")

# Render the full episode
anim = render_episode(states, 
                      fps=24, 
                      full_pitch=True,
                      show_grid=False,
                      show_heatmap=False,
                      show_rewards=False,
                      save_path="videoTest/testAttDefGK.mp4")

time_end = time()

print("Rendering complete. Animations saved in the 'videoTest' directory.")
print(f"Rendering took {time_end - time_start:.2f} seconds.")
