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

class OffensiveScenarioShotSingleAgent(gymnasium.Env):
    """
    Gym environment simulating a 1v1 football offensive scenario with shooting mechanics.

    The attacker can move and attempt shots towards the goal.
    The defender tries to regain possession by following the ball.
    The environment manages:
      - Movement of attacker and defender in normalized pitch coordinates.
      - Ball movement during dribbling and shooting phases.
      - Shooting simulation including direction, power, and shot progress.
      - Reward computation for positioning, successful shots, possession loss, and out-of-bounds.
      - Episode termination on goals, possession loss, or max steps.

    Key attributes added for shooting state management:
      - is_shooting: bool flag indicating if a shot is currently in progress.
      - shot_direction: numpy array representing the unit vector direction of the shot.
      - shot_power: float scalar representing the speed of the shot.
      - shot_position: starting position (normalized) of the shot.
      - shot_progress: float [0,1] tracking shot progression along the trajectory.
    """
    def __init__(self, pitch, max_steps=240, fps=24):
        """
        Initialize the environment.

        Args:
            pitch: Pitch object with dimensions and rendering utilities.
            max_steps (int): Maximum number of steps per episode.
            fps (int): Frames per second for simulation timing.
        """

        # Initialize the parent class
        # This sets up the environment with the necessary metadata and configurations
        super().__init__()

        # Store pitch for position normalization and rendering
        self.pitch = pitch

        # Action space: (movement_x_y, shoot_flag, shot_power, shot_direction)
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # movement x,y
            spaces.Discrete(2),  # shoot flag 0 or 1
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # shot power [0,1]
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # shot direction vector x,y
        ))


        # Observation space: normalized player positions, ball position, and shooting state
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),  # 7 original + 1 is_shooting flag + 1 shot_progress float
            dtype=np.float32
        )

        # Initialize player and ball entities
        self.attacker = PlayerAttacker()
        self.defender = PlayerDefender()
        self.ball = Ball()

        # Initialize environment state variables
        self.has_possession = True
        self.done = False
        self.steps = 0
        self.max_steps = max_steps # Standard 24 FPS * 10 seconds

        # Grid parameters for reward shaping (CELL_SIZE is in meters from pitch.py)
        self.num_cells_x = self.pitch.num_cells_x
        self.num_cells_y = self.pitch.num_cells_y
        self.cell_size = self.pitch.CELL_SIZE

        # Simulation Parameters
        self.fps = fps                          # Frames per second for rendering (standard 24 FPS)
        self.time_per_step = 1.0 / self.fps     # Time per step in seconds 

        # Reward grid initialized as None; built on first use
        self.reward_grid = None

        # Shooting state variables
        self.is_shooting = False            # Flag indicating if a shot is in progress
        self.shot_direction = None          # numpy array unit vector for shot direction
        self.shot_power = 0.0               # Float representing shot speed in m/s
        self.shot_position = None           # Normalized coordinates where shot started
        self.shot_progress = 0.0            # Progress along shot path from 0 (start) to 1 (end)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        The attacker and defender are placed at their starting positions, the ball is centered,
        and the game state and shooting state are initialized.
        """
        # Reset the random seed if provided (important for reproducibility)
        super().reset(seed=seed)

        # Reset positions of players and ball (normalized coordinates)
        self.attacker.reset_position(*normalize(60, 40))
        self.defender.reset_position(*normalize(110, 40))
        self.ball.position = normalize(60, 40)

        # Reset main environment state flags and counters
        self.has_possession = True
        self.done = False
        self.steps = 0

        # Reset shooting state
        self.is_shooting = False
        self.shot_direction = None
        self.shot_power = 0.0
        self.shot_position = None
        self.shot_progress = 0.0

        # Return initial observation and info dictionary as required by Gym API
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (tuple): Tuple containing:
                - movement_action (np.array): Continuous 2D vector in [-1,1] for movement.
                - shoot_flag (int): 0 or 1, 1 means initiate shot.
                - shot_power (np.array): array with one float in [0,1] indicating desired shot power.
                - shot_direction (np.array): 2D vector in [-1,1] indicating desired shot direction.

        Returns:
            obs (np.array): Observation after step.
            reward (float): Reward obtained this step.
            done (bool): Whether episode ended.
            truncated (bool): Gym API, unused here (False).
            info (dict): Additional info, empty here.
        """

        # Unpack action tuple
        movement_action, shoot_flag, shot_power_arr, shot_direction = action

        # Convert shot_power from array to scalar float
        shot_power = float(shot_power_arr[0])

        # Logic when no shot is currently active
        if not self.is_shooting:
            # Move attacker according to movement_action
            self.attacker.move_with_action(
                action=movement_action,
                time_per_step=self.time_per_step,
                x_range=self.pitch.X_MAX - self.pitch.X_MIN,
                y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
            )

            # Update ball position to follow attacker (dribbling)
            self._update_ball_position(movement_action)

            # If agent requests to shoot, initiate shooting
            if shoot_flag == 1:
                # Calculate actual shot parameters with player skill and precision adjustment
                shot_quality, actual_shot_direction, actual_shot_power = self.attacker.shoot(
                    desired_direction=shot_direction,
                    desired_power=shot_power
                )

                # Initialize shooting state variables
                self.is_shooting = True
                self.shot_direction = actual_shot_direction
                self.shot_power = actual_shot_power
                self.shot_position = self.ball.position  # current ball position as shot origin
                self.shot_progress = 0.0  # reset shot progress

        else:
            # Shot is in progress
            # Keep attacker fixed at the shot start position
            self.attacker.position = self.shot_position

            # Move the ball autonomously along shot trajectory
            self._update_ball_position(None)  # action ignored during shot

        # Defender always moves towards the ball
        ball_x, ball_y = self.ball.position
        def_x, def_y = self.defender.get_position()
        direction = np.array([ball_x - def_x, ball_y - def_y])

        # Normalize direction vector or zero if at same position
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        else:
            direction = np.array([0.0, 0.0])

        # Move defender towards the ball
        self.defender.move_with_action(
            direction,
            time_per_step=self.time_per_step,
            x_range=self.pitch.X_MAX - self.pitch.X_MIN,
            y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
        )

        # Compute reward
        reward = self._compute_reward()

        # Check if shot finished
        if self.is_shooting:
            # Convert ball position to meters for bounds checking
            pitch_width = self.pitch.X_MAX - self.pitch.X_MIN
            pitch_height = self.pitch.Y_MAX - self.pitch.Y_MIN
            ball_x_m = ball_x * pitch_width + self.pitch.X_MIN
            ball_y_m = ball_y * pitch_height + self.pitch.Y_MIN

            # End shooting phase if ball out of pitch or beyond goal line
            if (ball_x_m > self.pitch.FIELD_WIDTH or ball_x_m < 0 or
                ball_y_m < 0 or ball_y_m > self.pitch.FIELD_HEIGHT):
                # Reset shooting state
                self.is_shooting = False
                self.shot_direction = None
                self.shot_power = 0.0
                self.shot_position = None
                self.shot_progress = 0.0
                # Could add additional logic here for final shot reward/penalty

        # Step counter and episode termination
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}


    def _update_ball_position(self, action):
        """
        Update the ball position based on the current state.

        - If a shot is in progress (self.is_shooting == True), move the ball along the shot trajectory.
        - Otherwise, make the ball follow the attacker with a slight offset to simulate dribbling.
        """

        if self.is_shooting:
            # Compute pitch dimensions in meters
            pitch_width = self.pitch.X_MAX - self.pitch.X_MIN
            pitch_height = self.pitch.Y_MAX - self.pitch.Y_MIN

            # Calculate ball displacement for this step (normalized coordinates)
            velocity_x = (self.shot_direction[0] * self.shot_power * self.time_per_step) / pitch_width
            velocity_y = (self.shot_direction[1] * self.shot_power * self.time_per_step) / pitch_height

            # Update ball position by moving it along the shot direction vector
            new_ball_x = self.ball.position[0] + velocity_x
            new_ball_y = self.ball.position[1] + velocity_y

            # Update ball position
            self.ball.position = (new_ball_x, new_ball_y)

            # Optionally, update shot progress here if needed (or elsewhere)
            self.shot_progress += np.linalg.norm([velocity_x, velocity_y])

        else:
            # Normal dribbling: ball follows attacker with small offset
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
    
    def _is_goal(self, x, y):
        """
        Check if the ball is in the net
        """
        GOAL_MIN_Y = self.pitch.CENTER_Y - self.pitch.GOAL_HEIGHT / 2
        GOAL_MAX_Y = self.pitch.CENTER_Y + self.pitch.GOAL_HEIGHT / 2
        return x > self.pitch.FIELD_WIDTH and GOAL_MIN_Y <= y <= GOAL_MAX_Y
    
    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball,
        possession status, shooting flag, and shot progress.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = normalize(*self.attacker.get_position())
        def_x, def_y = normalize(*self.defender.get_position())
        ball_x, ball_y = normalize(*self.ball.position)

        # Possession status: 1 if attacker has possession, 0 otherwise
        possession = 1.0 if self.has_possession else 0.0

        # Shooting flag: 1 if shooting in progress, else 0
        is_shooting = 1.0 if self.is_shooting else 0.0

        # Shot progress: normalized progress [0,1], or 0 if no shot
        shot_progress = self.shot_progress if self.is_shooting else 0.0

        return np.array([
            att_x, att_y,
            def_x, def_y,
            ball_x, ball_y,
            possession,
            is_shooting,
            shot_progress
        ], dtype=np.float32)
    
    def _compute_reward(self):
        """
        Compute the reward for the current environment step.

        The reward considers several factors:
        - A small time penalty to encourage active play.
        - Position-based reward from a predefined grid to encourage progression and good positioning.
        - Penalty for being out of bounds.
        - Large positive reward for scoring a goal.
        - Penalty for losing ball possession to the defender.
        - Additional one-time bonus reward when a shot is just initiated, 
        encouraging shooting from advantageous positions and directions.

        Returns:
            float: The computed reward value for the current step.
        """

        reward = 0.0  # Initialize reward accumulator

        # Get attacker's normalized position and convert to meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        y_m = att_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

        # Apply a small time penalty to discourage inactivity
        reward -= 0.1

        # Add position-based reward from the reward grid
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += pos_reward

        # Terminate episode if attacker is out of bounds (strong penalty)
        if pos_reward <= -4.0:
            self.done = True
            return reward

        # Check if a goal has been scored and reward accordingly
        if self._is_goal(x_m, y_m):
            self.done = True
            reward += 5.0
            return reward

        # Penalize losing possession of the ball to the defender
        if self._check_possession_loss():
            reward -= 1.0
            self.done = True
            return reward

        # Provide an additional one-time shooting bonus when a shot is just started
        # This is determined by checking if shooting is active and the shot progress is near zero
        if self.is_shooting and self.shot_progress < 0.01:
            # Convert ball position to meters
            ball_x, ball_y = self.ball.position
            ball_x_m = ball_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
            ball_y_m = ball_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

            # Bonus for shooting from a good position
            shoot_pos_reward = self._get_position_reward(ball_x_m, ball_y_m)
            reward += 0.5 * shoot_pos_reward

            # Compute goal direction vector
            goal_direction = np.array([1.0, 0.5]) - np.array([ball_x, ball_y])
            goal_direction /= np.linalg.norm(goal_direction)

            # Use shot direction if available, else default to goal direction
            shot_dir = self.shot_direction if self.shot_direction is not None else goal_direction

            # Compute angle between shot direction and goal direction (in radians)
            alignment = np.clip(np.dot(shot_dir, goal_direction), -1, 1)
            angle = np.arccos(alignment)  # angle between 0 and pi

            # Normalize angle to [0, 1]
            normalized_angle = angle / np.pi

            # Calculate reward linearly from +0.5 (perfect alignment) to -0.5 (opposite)
            angle_reward = 0.5 - normalized_angle

            reward += angle_reward
        
        return reward

    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass