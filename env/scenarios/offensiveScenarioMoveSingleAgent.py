import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium
from gymnasium import spaces
import numpy as np
from players.playerAttacker import PlayerAttacker
from players.playerDefender import PlayerDefender
from env.objects.ball import Ball
from helpers.helperFunctions import normalize

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

    def __init__(self, pitch, max_steps=240, fps=24):

        # Initialize the parent class
        # This sets up the environment with the necessary metadata and configurations
        super().__init__()

        # Pitch object for rendering and coordinate normalization
        self.pitch = pitch

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
        self.num_cells_x = self.pitch.num_cells_x
        self.num_cells_y = self.pitch.num_cells_y
        self.cell_size = self.pitch.CELL_SIZE

        # Simulation Parameters
        self.fps = fps                          # Frames per second for rendering (standard 24 FPS)
        self.time_per_step = 1.0 / self.fps     # Time per step in seconds 

        # Reward grid for position-based rewards
        self.reward_grid = None

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

        return self._get_obs(), {}


    def step(self, action):
        """Execute one environment step: attacker moves, ball follows, defender pursues."""

        # Attacker movement
        self.attacker.move_with_action(
            action=action,
            time_per_step=self.time_per_step,
            x_range=self.pitch.X_MAX - self.pitch.X_MIN,
            y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
        )

        # Ball follows attacker (dribbling)
        self._update_ball_position(action)

        # Defender movement towards attacker
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        direction = np.array([att_x - def_x, att_y - def_y])
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)
        else:
            direction = np.array([0.0, 0.0])

        self.defender.move_with_action(
            direction,
            time_per_step=self.time_per_step,
            x_range=self.pitch.X_MAX - self.pitch.X_MIN,
            y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
        )

        # Compute cumulative reward (including possession loss, bounds, etc.)
        reward = self._compute_reward()

        # Timeout check
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
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
        The defender steals the ball if they are within a certain distance threshold
        and their tackling ability is sufficient.
        """
        # Get real positions in meters
        def_x, def_y = self.defender.get_position()
        def_x = def_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        def_y = def_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN
        ball_x = self.ball.position[0] * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        ball_y = self.ball.position[1] * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

        threshold = 0.5  # meters
        if np.sqrt((def_x - ball_x) ** 2 + (def_y - ball_y) ** 2) < threshold:
            if np.random.rand() < self.defender.tackling:
                self.has_possession = False
                self.done = True
                return True
        return False
    
    def _build_reward_grid(self):
        """
        Builds the reward grid based on pitch dimensions and stores it.
        Out of bounds cells are assigned strong negative rewards (-5.0).
        Cells inside the goal area (behind goal line and within goal height) get high positive reward (+5.0).
        """
        grid = np.zeros((self.num_cells_x, self.num_cells_y))

        goal_min_y = self.pitch.CENTER_Y - self.pitch.GOAL_HEIGHT / 2
        goal_max_y = self.pitch.CENTER_Y + self.pitch.GOAL_HEIGHT / 2
        goal_x_min = self.pitch.FIELD_WIDTH  # goal line
        goal_x_max = self.pitch.FIELD_WIDTH + self.pitch.GOAL_DEPTH  # depth of the goal area

        for i in range(self.num_cells_x):
            for j in range(self.num_cells_y):

                # Calculate center of the cell in meters
                cell_x = self.pitch.X_MIN + (i + 0.5) * self.cell_size
                cell_y = self.pitch.Y_MIN + (j + 0.5) * self.cell_size

                # Check if out of bounds (excluding goal area)
                is_out = (cell_x < 0 or cell_x > self.pitch.FIELD_WIDTH or cell_y < 0 or cell_y > self.pitch.FIELD_HEIGHT)

                # Check if inside the goal area (behind goal line, within goal height)
                is_goal_area = (goal_x_min <= cell_x <= goal_x_max) and (goal_min_y <= cell_y <= goal_max_y)

                # Assign rewards
                if is_out and not is_goal_area:
                    grid[i, j] = -5.0  # penalty out of bounds
                elif is_goal_area:
                    grid[i, j] = 5.0   # high reward for goal area
                else:
                    x_norm = (cell_x - self.pitch.X_MIN) / (self.pitch.X_MAX - self.pitch.X_MIN)
                    y_norm = (cell_y - self.pitch.Y_MIN) / (self.pitch.Y_MAX - self.pitch.Y_MIN)
                    x_reward = -0.5 + 1.0 * x_norm
                    y_penalty = -0.15 * abs(y_norm - 0.5) * 2
                    grid[i, j] = x_reward + y_penalty

        self.reward_grid = grid

    def _get_position_reward(self, x, y):
        """
        Returns the reward from the reward grid for the attacker's position.
        Builds the grid on first access.
        """
        if self.reward_grid is None:
            self._build_reward_grid()

        cell_x = int((x - self.pitch.X_MIN) / self.cell_size)
        cell_y = int((y - self.pitch.Y_MIN) / self.cell_size)

        cell_x = np.clip(cell_x, 0, self.num_cells_x - 1)
        cell_y = np.clip(cell_y, 0, self.num_cells_y - 1)

        return self.reward_grid[cell_x, cell_y]



    def _compute_reward(self):
        """
        Computes the reward for the current step based on the attacker's position.
        Looks up the reward grid, applies a time penalty, and checks for goals or possession loss.
        """
        reward = 0.0  # Initialize reward for this step

        # Convert attacker's position from normalized [0, 1] to meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        y_m = att_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

        # Apply small time penalty to encourage active movement
        reward -= 0.1

        # Get reward based on position from reward grid
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += pos_reward

        # If in out-of-bounds cell, terminate the episode
        if pos_reward <= -4.0:
            self.done = True
            return reward

        # Check if a goal has been scored
        if self._is_goal(x_m, y_m):
            self.done = True
            reward += 5.0
            return reward  

        # Check if possession was lost
        if self._check_possession_loss():
            reward -= 1.0
            self.done = True
            return reward

        return reward

    def _is_goal(self, x, y):
        """
        Check if the ball is in the net
        """
        GOAL_MIN_Y = self.pitch.CENTER_Y - self.pitch.GOAL_HEIGHT / 2
        GOAL_MAX_Y = self.pitch.CENTER_Y + self.pitch.GOAL_HEIGHT / 2
        return x > self.pitch.FIELD_WIDTH and GOAL_MIN_Y <= y <= GOAL_MAX_Y

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
