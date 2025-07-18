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
from helpers.helperFunctions import normalize

# Create players and ball
attacker = PlayerAttacker()
defender = PlayerDefender()
goalkeeper = PlayerGoalkeeper()
ball = Ball()

# Set initial positions (normalized)
attacker.reset_position(*normalize(60, 40))    # Center field
defender.reset_position(*normalize(100, 40))   # Further behind
goalkeeper.reset_position(*normalize(120, 40)) # In goal center
ball.position = normalize(60, 40)

time_per_step = 1 / 24

attacker_max_speed = 10.0
defender_max_speed = 10.0
goalkeeper_max_speed = 10.0

# Store states frame-by-frame
states = []

for frame in range(120):

    # Attacker moves randomly towards the goal
    direction_x = np.random.uniform(1, 1)
    direction_y = np.random.uniform(-1, 1)

    speedATT = attacker.speed * attacker_max_speed * time_per_step

    dx = direction_x * speedATT / (X_MAX - X_MIN)
    dy = direction_y * speedATT / (Y_MAX - Y_MIN)

    attacker.move([dx, dy])

    # Defender moves towards the attacker
    att_pos = np.array(attacker.get_position())
    def_pos = np.array(defender.get_position())
    direction = att_pos - def_pos
    distance_to_att = np.linalg.norm(direction)

    if distance_to_att > 0.01:
        direction = direction / distance_to_att

        speedDEF = defender.speed * defender_max_speed * time_per_step
        dx = direction[0] * speedDEF / (X_MAX - X_MIN)
        dy = direction[1] * speedDEF / (Y_MAX - Y_MIN)
        defender.move([dx, dy])
    else:
        defender.move([0, 0])

    # Goalkeeper small pseudo-random movement
    direction_x = np.random.uniform(-1, 1)
    direction_y = np.random.uniform(-1, 1)

    speedGK = goalkeeper.speed * goalkeeper_max_speed * time_per_step

    dx = direction_x * speedGK / (X_MAX - X_MIN)
    dy = direction_y * speedGK / (Y_MAX - Y_MIN)
    goalkeeper.move([dx, dy], 0.02)

    # Ball follows attacker
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
anim = render_episode(states, fps=24, show_grid=True, show_cell_ids=True, full_pitch=True)
plt.show()
