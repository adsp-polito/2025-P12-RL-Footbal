import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from players.playerGoalkeeper import PlayerGoalkeeper
from env.ball import Ball
from helpers.visuals import render_episode
from env.pitch import X_MIN, Y_MIN, X_MAX, Y_MAX
import numpy as np


def normalize_to_field(x, y):
    x_norm = (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = (y - Y_MIN) / (Y_MAX - Y_MIN)
    return x_norm, y_norm


# Create players and ball
attacker = PlayerAttacker()
defender = PlayerDefender()
goalkeeper = PlayerGoalkeeper()
ball = Ball()

# Set initial positions (normalized)
attacker.reset_position(*normalize_to_field(60, 40))    # Center field
defender.reset_position(*normalize_to_field(100, 40))    # Further behind
goalkeeper.reset_position(*normalize_to_field(120, 40))   # In goal center
ball.position = normalize_to_field(60, 40)


# Store states frame-by-frame
states = []

for frame in range(48):
    # Attacker moves towards right (the goal)
    attacker.move([0.05, 0], 0.05)

    # Defender accelerates faster to reach the attacker
    defender.move([-0.08, 0], 0.05)

    # Goalkeeper random small movement (pseudo-random, realistic)
    direction_x = np.random.uniform(-0.1, 0.1)
    direction_y = np.random.uniform(-0.1, 0.1)
    goalkeeper.move([direction_x, direction_y], 0.02)


    # Ball follows the attacker
    ball.position = (attacker.get_position()[0] + 0.01, attacker.get_position()[1])

    # Copy attacker
    attacker_copy = PlayerAttacker(
        shooting=attacker.shooting,
        passing=attacker.passing,
        dribbling=attacker.dribbling,
        speed=attacker.speed
    )
    attacker_copy.reset_position(*attacker.get_position())

    # Copy defender
    defender_copy = PlayerDefender(
        tackling=defender.tackling,
        marking=defender.marking,
        anticipation=defender.anticipation,
        speed=defender.speed
    )
    defender_copy.reset_position(*defender.get_position())

    # Copy goalkeeper
    goalkeeper_copy = PlayerGoalkeeper(
        reflexes=goalkeeper.reflexes,
        diving=goalkeeper.diving,
        positioning=goalkeeper.positioning,
        speed=goalkeeper.speed
    )
    goalkeeper_copy.reset_position(*goalkeeper.get_position())

    # Copy ball
    ball_copy = Ball()
    ball_copy.position = ball.position

    # Save state
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy, goalkeeper_copy]
    })


# Render
anim = render_episode(states, fps=24, show_grid=False, show_cell_ids=False, full_pitch=True)
plt.show()
