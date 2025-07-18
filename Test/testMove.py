import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from env.offensiveScenarioMoveSingleAgent import OffensiveScenarioMoveSingleAgent
from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from env.ball import Ball
from helpers.visuals import render_episode


# Create environment
env = OffensiveScenarioMoveSingleAgent()
obs, info = env.reset()

states = []

done = False
while not done:
    # Move attacker with a random action
    action = env.action_space.sample()  # Random action in the range [-1, 1]
    obs, reward, done, truncated, info = env.step(action)

    # Copy attacker
    attacker_copy = PlayerAttacker(
        shooting=env.attacker.shooting,
        passing=env.attacker.passing,
        dribbling=env.attacker.dribbling,
        speed=env.attacker.speed
    )
    attacker_copy.position = env.attacker.get_position()

    # Copy defender
    defender_copy = PlayerDefender(
        tackling=env.defender.tackling,
        marking=env.defender.marking,
        anticipation=env.defender.anticipation,
        speed=env.defender.speed
    )
    defender_copy.position = env.defender.get_position()

    # Copy ball
    ball_copy = Ball()
    ball_copy.position = env.ball.position

    # Save current state
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

# Render the full episode
anim = render_episode(states, fps=24, full_pitch=True)
plt.show()
