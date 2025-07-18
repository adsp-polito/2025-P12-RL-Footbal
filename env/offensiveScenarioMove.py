import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from gym import spaces
import numpy as np

from helpers.visuals import render_episode
from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from env.ball import Ball
from env.pitch import X_MIN, X_MAX, Y_MIN, Y_MAX, FIELD_HEIGHT, CENTER_Y, GOAL_HEIGHT, FIELD_WIDTH
from helpers.helperFunctions import normalize, distance


# Coordinate System Design:
#
# This environment works in normalized coordinates within [0, 1] for X and Y.
# Player positions (attacker, defender) and the ball are stored and updated normalized.
# Movements are computed in this normalized space.
#
# Normalization:
# - The physical pitch spans X_MIN to X_MAX and Y_MIN to Y_MAX (in meters).
# - Positions are normalized between 0 and 1 with these limits.
#
# Rewards and rendering:
# - Rewards are computed in meters, denormalizing positions as needed.
# - Rendering denormalizes to draw correctly on the pitch in meters.
#
# Summary:
# 1. Environment stores positions in [0, 1].
# 2. Actions move positions in [0, 1].
# 3. Observations are returned normalized.
# 4. Rendering converts to meters.
# 5. Rewards are computed in meters after denormalizing.

class OffensiveScenarioMove(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Action space: attacker moves in (x, y) direction
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: attacker x,y - defender x,y - ball x,y - possession flag
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        # Game entities
        self.attacker = PlayerAttacker()
        self.defender = PlayerDefender()
        self.ball = Ball()

        # Possession and step management
        self.has_possession = True
        self.done = False
        self.steps = 0
        self.max_steps = 240  # 24 FPS, ~10 seconds

        # Simulation parameters
        self.time_per_step = 1 / 24
        self.attacker_max_speed = 10.0
        self.defender_max_speed = 10.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.attacker.reset_position(*normalize(60, 40))
        self.defender.reset_position(*normalize(110, 40))
        self.ball.position = normalize(60, 40)

        self.has_possession = True
        self.done = False
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one simulation step in the environment.

        The attacker moves based on the input action. The ball is positioned slightly 
        ahead of the attacker, simulating ball control while dribbling. The defender 
        automatically moves towards the attacker. Rewards are assigned based on 
        attacker's progress, loss of possession, out-of-bounds, or scoring.

        Args:
            action (np.array): Action vector for attacker movement (x, y), values in [-1, 1].

        Returns:
            obs (np.array): Current observation of the environment.
            reward (float): Immediate reward from the environment.
            done (bool): Whether the episode has ended.
            truncated (bool): Always False in this environment.
            info (dict): Empty dictionary.
        """

        # Attacker movement (normalized coordinates)
        attacker_speed = self.attacker.speed * self.attacker_max_speed
        dx = action[0] * attacker_speed * self.time_per_step / (X_MAX - X_MIN)
        dy = action[1] * attacker_speed * self.time_per_step / (Y_MAX - Y_MIN)
        self.attacker.move([dx, dy])

        # Ball offset in front of attacker to simulate control/dribbling
        direction = np.array([dx, dy])
        if np.linalg.norm(direction) > 0:
            direction /= np.linalg.norm(direction)
            ball_offset = 0.01  # normalized distance ahead of attacker
            ball_pos_x = self.attacker.position[0] + direction[0] * ball_offset
            ball_pos_y = self.attacker.position[1] + direction[1] * ball_offset
            self.ball.position = (ball_pos_x, ball_pos_y)
        else:
            # If attacker is standing still, keep ball just slightly ahead
            self.ball.position = (self.attacker.position[0] + 0.01, self.attacker.position[1])

        # Defender movement towards attacker (normalized coordinates)
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        direction = np.array([att_x - def_x, att_y - def_y])
        distance_to_attacker = np.linalg.norm(direction)

        defender_speed = self.defender.speed * self.defender_max_speed
        max_step_x = defender_speed * self.time_per_step / (X_MAX - X_MIN)
        max_step_y = defender_speed * self.time_per_step / (Y_MAX - Y_MIN)

        if distance_to_attacker != 0:
            direction /= distance_to_attacker
            def_dx = direction[0] * max_step_x
            def_dy = direction[1] * max_step_y
        else:
            def_dx, def_dy = 0.0, 0.0

        self.defender.move([def_dx, def_dy])

        # Possession check
        norm_threshold = 0.5 / np.sqrt((X_MAX - X_MIN) * (Y_MAX - Y_MIN))
        if distance(self.defender.get_position(), self.ball.position) < norm_threshold:
            if np.random.rand() < self.defender.tackling:
                self.has_possession = False
                self.done = True
                reward = -1.0
                self._print_positions(att_x, att_y, def_x, def_y, "Ball stolen", reward)
                print(f"STEP {self.steps + 1:3} | FINAL REWARD: {reward:.4f}")
                return self._get_obs(), reward, self.done, False, {}

        # Denormalize positions to meters for reward calculation
        att_x_m = att_x * (X_MAX - X_MIN) + X_MIN
        att_y_m = att_y * (Y_MAX - Y_MIN) + Y_MIN

        reward = -0.01  # Small negative reward for inactivity

        # Out of bounds penalty (unless scoring goal)
        GOAL_MIN_Y = CENTER_Y - GOAL_HEIGHT / 2
        GOAL_MAX_Y = CENTER_Y + GOAL_HEIGHT / 2
        if (att_x_m < 0 or att_x_m > FIELD_WIDTH or att_y_m < 0 or att_y_m > FIELD_HEIGHT):
            if not (att_x_m > FIELD_WIDTH and GOAL_MIN_Y <= att_y_m <= GOAL_MAX_Y):
                reward = -2.0
                self.done = True
                self._print_positions(att_x, att_y, def_x, def_y, "Out of bounds", reward)
                print(f"STEP {self.steps + 1:3} | FINAL REWARD: {reward:.4f}")
                return self._get_obs(), reward, self.done, False, {}

        # Progressive reward for moving towards the goal line
        distance_x = max(FIELD_WIDTH - att_x_m, 0.1)
        reward += (1.5 - np.log(distance_x)) * 0.01

        # Goal scored
        if att_x_m > FIELD_WIDTH and GOAL_MIN_Y <= att_y_m <= GOAL_MAX_Y:
            reward = 5.0
            self.done = True
            self._print_positions(att_x, att_y, def_x, def_y, "Goal scored", reward)
            print(f"STEP {self.steps + 1:3} | FINAL REWARD: {reward:.4f}")
            return self._get_obs(), reward, self.done, False, {}

        # Step print
        self._print_positions(att_x, att_y, def_x, def_y, f"Dist to Goal: {distance_x:.2f}m", reward)

        # Check timeout condition
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            print(f"STEP {self.steps:3} | Timeout reached | FINAL REWARD: {reward:.4f}")
            return self._get_obs(), reward, self.done, False, {}

        # Return Gym step output
        return self._get_obs(), reward, self.done, False, {}




    def _print_positions(self, att_x, att_y, def_x, def_y, status, reward):
        """
        Print the positions of attacker, defender, and ball in meters.
        """

        # Denormalize positions to meters
        att_x_m = att_x * (X_MAX - X_MIN) + X_MIN
        att_y_m = att_y * (Y_MAX - Y_MIN) + Y_MIN
        def_x_m = def_x * (X_MAX - X_MIN) + X_MIN
        def_y_m = def_y * (Y_MAX - Y_MIN) + Y_MIN
        ball_x_m = self.ball.position[0] * (X_MAX - X_MIN) + X_MIN
        ball_y_m = self.ball.position[1] * (Y_MAX - Y_MIN) + Y_MIN

        print(f"STEP {self.steps:3} | ATTACKER: ({att_x_m:.2f}, {att_y_m:.2f}) | "
              f"DEFENDER: ({def_x_m:.2f}, {def_y_m:.2f}) | BALL: ({ball_x_m:.2f}, {ball_y_m:.2f}) | "
              f"{status} | REWARD: {reward:.4f}")

    def _get_obs(self):
        """
        Get the current observation in normalized coordinates.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = normalize(*self.attacker.get_position())
        def_x, def_y = normalize(*self.defender.get_position())
        ball_x, ball_y = normalize(*self.ball.position)

        # Possession flag
        possession = 1.0 if self.has_possession else 0.0

        return np.array([att_x, att_y, def_x, def_y, ball_x, ball_y, possession], dtype=np.float32)

    def close(self):
        pass
