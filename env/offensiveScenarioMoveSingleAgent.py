import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium
from gymnasium import spaces
import numpy as np
from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from env.ball import Ball
from env.pitch import X_MIN, X_MAX, Y_MIN, Y_MAX, FIELD_HEIGHT, CENTER_Y, GOAL_HEIGHT, FIELD_WIDTH, CELL_SIZE
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


class OffensiveScenarioMoveSingleAgent(gymnasium.Env):
    """
    Gym environment simulating a football 1v1 offensive scenario.

    The attacker attempts to advance towards the goal while maintaining ball possession.
    The defender tries to steal the ball. Rewards encourage progression towards the goal, penalize
    losing possession or going out of bounds, and reward scoring.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=240):
        super().__init__()

        # Action space: attacker moves in x and y, continuous control
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: positions (normalized) and possession
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        # Entities
        self.attacker = PlayerAttacker()
        self.defender = PlayerDefender()
        self.ball = Ball()

        # Simulation parameters
        self.has_possession = True
        self.done = False
        self.steps = 0
        self.max_steps = max_steps  # standard 24 FPS * 10 seconds

        # Grid parameters for reward shaping (CELL_SIZE is in meters from pitch.py)
        self.num_cells_x = int((X_MAX - X_MIN) / CELL_SIZE)
        self.num_cells_y = int((Y_MAX - Y_MIN) / CELL_SIZE)

        # Simulation Parameters
        self.time_per_step = 1 / 24
        self.attacker_max_speed = 10.0  # m/s ~ 36 km/h
        self.defender_max_speed = 10.0  # m/s ~ 36 km/h

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        The attacker and defender are placed at their starting positions, the ball is centered,
        and the game state is initialized.
        """

        # Reset the random seed if provided
        super().reset(seed=seed)
        self.attacker.reset_position(*normalize(60, 40))
        self.defender.reset_position(*normalize(110, 40))
        self.ball.position = normalize(60, 40)

        # Reset game state
        self.has_possession = True
        self.done = False
        self.steps = 0
        
        # Initialize current column (using real-world meters)
        att_x, _ = self.attacker.get_position()
        x_m = att_x * (X_MAX - X_MIN) + X_MIN
        self.last_column = int(x_m // 5)

        return self._get_obs(), {}


    def step(self, action):
        """Execute one environment step: attacker moves, ball follows, defender pursues"""

        # Attacker movement using OOP encapsulation
        self.attacker.move_with_action(
            action=action,
            time_per_step=self.time_per_step,
            x_range=X_MAX - X_MIN,
            y_range=Y_MAX - Y_MIN
        )

        # Ball offset in attacker's direction (dribbling)
        self._update_ball_position(action)

        # Defender moves automatically towards attacker using OOP encapsulation
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        direction = np.array([att_x - def_x, att_y - def_y])
        distance_to_attacker = np.linalg.norm(direction)
        if distance_to_attacker != 0:
            direction /= distance_to_attacker
        else:
            direction = np.array([0.0, 0.0])

        self.defender.move_with_action(
            direction,
            time_per_step=self.time_per_step,
            x_range=X_MAX - X_MIN,
            y_range=Y_MAX - Y_MIN
        )

        # Possession check
        if self._check_possession_loss():
            reward = -1.0
            self._log_step("Ball stolen", reward)
            return self._get_obs(), reward, True, False, {}

        # Reward logic
        reward = self._compute_reward()
        self._log_step(f"Dist to Goal", reward)

        # Timeout check
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            # print(f"STEP {self.steps:3} | Timeout reached | FINAL REWARD: {reward:.4f}")
            return self._get_obs(), reward, self.done, False, {}

        return self._get_obs(), reward, self.done, False, {}

    def _update_ball_position(self, action):
        """
        Update the ball position based on the attacker's movement with a slight offset.
        If the attacker is stationary, the ball moves slightly forward
        to simulate dribbling
        """

        direction = np.array([action[0], action[1]])
        if np.linalg.norm(direction) > 0:
            direction /= np.linalg.norm(direction)
            offset = 0.01
            ball_x = self.attacker.position[0] + direction[0] * offset
            ball_y = self.attacker.position[1] + direction[1] * offset
            self.ball.position = (ball_x, ball_y)
        else:
            self.ball.position = (self.attacker.position[0] + 0.01, self.attacker.position[1])
            
    def _check_possession_loss(self):
        """
        Check if the defender has stolen the ball from the attacker.
        The defender steals the ball if they are close enough and have a chance to tackle.
        """
        # Define a threshold for tackling based on the distance to the ball
        threshold = 0.5 / np.sqrt((X_MAX - X_MIN) * (Y_MAX - Y_MIN)) # Normalized distance threshold (1 meter)
        if distance(self.defender.get_position(), self.ball.position) < threshold:
            if np.random.rand() < self.defender.tackling:
                self.has_possession = False
                self.done = True
                return True
        return False
    
    def _get_position_reward(self, x, y):
        """
        Compute a reward based on the attacker's position on the pitch in meters.
        Reward is higher closer to the opponent's goal (x -> 125) and slightly favors central y positions.
        """

        # Normalize x and y to [0, 1] according to real field dimensions
        x_norm = (x - X_MIN) / (X_MAX - X_MIN)
        y_norm = (y - Y_MIN) / (Y_MAX - Y_MIN)

        # Reward increases linearly with x (progression towards opponent's goal)
        x_reward = -0.5 + 1.0 * x_norm  # From -0.5 (X_MIN) to +0.5 (X_MAX)

        # Slight penalty for lateral positions: y=0.5 is central, sides get a small penalty
        y_penalty = -0.15 * abs(y_norm - 0.5) * 2  # Max -0.15 at edges, 0 at center

        return x_reward + y_penalty


    def _compute_reward(self):
        """
        Reward shaping based solely on smooth positional reward (x, y) over the pitch.
        Encourages progression towards opponent's goal and allows realistic lateral movement.
        """
        reward = 0.0

        # Get attacker's position in meters (denormalized)
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (X_MAX - X_MIN) + X_MIN
        y_m = att_y * (Y_MAX - Y_MIN) + Y_MIN

        # Base time penalty to encourage movement
        reward -= 0.1

        # Position-based reward shaping (x progression and y positioning)
        reward += self._get_position_reward(x_m, y_m)

        # Terminal conditions: out of bounds or goal
        if self._is_out_of_bounds(x_m, y_m):
            self.done = True
            reward -= 3.5
            self._log_step("Out of bounds", reward)
            return reward

        if self._is_goal(x_m, y_m):
            self.done = True
            reward += 5.0
            self._log_step("Goal scored", reward)
            return reward

        return reward


    def _is_out_of_bounds(self, x, y):
        """
        Check if the ball is out of bounds
        """
        GOAL_MIN_Y = CENTER_Y - GOAL_HEIGHT / 2
        GOAL_MAX_Y = CENTER_Y + GOAL_HEIGHT / 2
        return (x < 0 or x > FIELD_WIDTH or y < 0 or y > FIELD_HEIGHT) and not (x > FIELD_WIDTH and GOAL_MIN_Y <= y <= GOAL_MAX_Y)

    def _is_goal(self, x, y):
        """
        Check if the ball is in the net
        """
        GOAL_MIN_Y = CENTER_Y - GOAL_HEIGHT / 2
        GOAL_MAX_Y = CENTER_Y + GOAL_HEIGHT / 2
        return x > FIELD_WIDTH and GOAL_MIN_Y <= y <= GOAL_MAX_Y

    def _log_step(self, status, reward):
        """
        Log the current step with positions and status.
        """

        # Denormalize positions for logging
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()

        # Note: X_MIN, X_MAX, Y_MIN, Y_MAX are defined in pitch
        att_x_m = att_x * (X_MAX - X_MIN) + X_MIN
        att_y_m = att_y * (Y_MAX - Y_MIN) + Y_MIN
        def_x_m = def_x * (X_MAX - X_MIN) + X_MIN
        def_y_m = def_y * (Y_MAX - Y_MIN) + Y_MIN
        ball_x_m = self.ball.position[0] * (X_MAX - X_MIN) + X_MIN
        ball_y_m = self.ball.position[1] * (Y_MAX - Y_MIN) + Y_MIN

        # print(f"STEP {self.steps:3} | ATTACKER: ({att_x_m:.2f}, {att_y_m:.2f}) | DEFENDER: ({def_x_m:.2f}, {def_y_m:.2f}) | BALL: ({ball_x_m:.2f}, {ball_y_m:.2f}) | {status} | REWARD: {reward:.4f}")

    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball, and possession status.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = normalize(*self.attacker.get_position())
        def_x, def_y = normalize(*self.defender.get_position())
        ball_x, ball_y = normalize(*self.ball.position)

        # Possession status: 1 if attacker has possession, 0 otherwise
        possession = 1.0 if self.has_possession else 0.0
        return np.array([att_x, att_y, def_x, def_y, ball_x, ball_y, possession], dtype=np.float32)

    def close(self):
        """
        Close the environment and release resources.
        """
        # No specific resources to release in this environment
        # but this method is here for compatibility with gym's API
        pass
