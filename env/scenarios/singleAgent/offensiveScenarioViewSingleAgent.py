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

class OffensiveScenarioViewSingleAgent(gymnasium.Env):
    """
    Gym environment simulating a 1v1 football offensive scenario with visual perception constraints.

    The attacker can move and attempt shots towards the goal, but is now limited by a visual field:
    shots are only allowed if the desired direction is within the attacker's field of view (FOV).
    The defender tries to regain possession by following the ball.

    The environment manages:
      - Movement of attacker and defender in normalized pitch coordinates.
      - Ball movement during dribbling and shooting phases.
      - Shooting simulation including direction, power, and visual constraints.
      - Reward computation for positioning, successful shots, invalid attempts, possession loss, and out-of-bounds.
      - Episode termination on goals, possession loss, or time expiration.

    Key visual constraint features:
      - Field of view is defined by a tunable angle and range, set per player.
      - Shots in directions outside the FOV are invalidated automatically.
      - Shooting behavior depends on player vision, not only position.

    Shooting state attributes:
      - is_shooting (bool): Flag indicating if a shot is currently in progress.
      - shot_direction (np.ndarray): Unit vector representing the direction of the shot.
      - shot_power (float): Scalar value for shot speed in meters per second.
      - shot_position (list): Normalized coordinates where the shot started.
      - shot_just_started (bool): Flag used to trigger one-time rewards or logic at shot start.
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

        # Action space: (movement_x_y, shot_flag, shot_power, shot_direction)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),  # x,y movement, shot_flag, shot_power, shot_dir x,y
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )


        # Observation space: normalized player positions, ball position, and shooting state
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),  # 7 original + 1 is_shooting flag
            dtype=np.float32
        )

        # Initialize player and ball entities
        self.attacker = PlayerAttacker()
        self.defender = PlayerDefender()
        self.ball = Ball()

        # Attacker starts with the ball
        self.ball.set_owner(self.attacker)  

        # Initialize environment state variables
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
        self.shot_just_started = False      # Flag to track if a shot just started (for one-time bonuses)


    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        The attacker and defender are placed at their starting positions, the ball is centered,
        and the game state and shooting state are initialized.
        """
        # Reset the random seed if provided (important for reproducibility)
        super().reset(seed=seed)

        # Reset positions of players and ball (normalized coordinates)
        self.attacker.reset_position(normalize(60, 40))
        self.defender.reset_position(normalize(110, 40))
        self.ball.position = normalize(60, 40)

        # Reset attacker state
        self.attacker.last_action_direction = np.array([1.0, 0.0])

        # Reset ball state
        self.ball.owner = self.attacker  # Attacker starts with the ball

        # Reset main environment state flags and counters
        self.done = False
        self.steps = 0

        # Reset shooting state
        self.is_shooting = False
        self.shot_direction = None
        self.shot_power = 0.0
        self.shot_position = None
        self.shot_just_started = False  # Reset for new episode

        # Return initial observation and info dictionary as required by Gym API
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (np.array): Array containing:
                - movement_action (x, y): 2D vector in [-1, 1] for attacker movement.
                - shot_flag: Float in [0, 1], interpreted as 1 if > 0.5.
                - shot_power: Float in [0, 1], indicating desired shot strength.
                - shot_direction: 2D vector in [-1, 1], indicating desired shot direction.

        Returns:
            obs (np.array): Current observation after the step.
            reward (float): Reward value computed for this step.
            done (bool): Whether the episode has ended.
            truncated (bool): Always False (not used).
            info (dict): Empty dictionary for compatibility.
        """

        # Unpack and format action components
        movement_action = action[0:2]
        shot_flag = 1 if action[2] > 0.5 else 0
        shot_power = float(action[3])
        shot_direction = action[4:6]

        # Attempt a shot if valid (not already shooting and attacker has the ball)
        if not self.is_shooting and shot_flag and self.ball.owner is self.attacker:
            shot_quality, actual_direction, actual_power = self.attacker.shoot(
                desired_direction=shot_direction,
                desired_power=shot_power
            )

            # Only initiate the shot if it's valid (within field of view)
            if actual_power > 0.0:
                self.is_shooting = True
                self.shot_direction = actual_direction
                self.shot_power = actual_power
                self.shot_position = self.ball.position
                self.shot_just_started = True
                self.ball.set_owner(None)

                # Convert real velocity to normalized velocity and set on the ball
                pitch_width = self.pitch.X_MAX - self.pitch.X_MIN
                pitch_height = self.pitch.Y_MAX - self.pitch.Y_MIN
                velocity_real = actual_direction * actual_power
                velocity_norm = np.array([
                    velocity_real[0] / pitch_width,
                    velocity_real[1] / pitch_height
                ])
                self.ball.set_velocity(velocity_norm)

        # Save attempted movement direction for reward logic
        self.attempted_movement_direction = movement_action.copy()

        # Move attacker if direction is visible
        if self.attacker.is_direction_visible(movement_action):
            self.attacker.move_with_action(
                action=movement_action,
                time_per_step=self.time_per_step,
                x_range=self.pitch.X_MAX - self.pitch.X_MIN,
                y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
            )

            # Update ball position only if attacker still owns it
            if self.ball.owner is self.attacker:
                self._update_ball_position(movement_action)

        # Update ball movement
        if self.is_shooting or self.ball.owner is None:
            # Ball moves autonomously only after shot
            self.ball.update(self.time_per_step)

        # Move defender toward current ball position
        ball_x, ball_y = self.ball.position
        def_x, def_y = self.defender.get_position()
        direction = np.array([ball_x - def_x, ball_y - def_y])
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else np.array([0.0, 0.0])

        self.defender.move_with_action(
            direction,
            time_per_step=self.time_per_step,
            x_range=self.pitch.X_MAX - self.pitch.X_MIN,
            y_range=self.pitch.Y_MAX - self.pitch.Y_MIN
        )

        # Compute reward (all logic handled internally)
        reward = self._compute_reward(shot_flag=shot_flag)

        # Shot termination logic (goal scored, ball out, etc.)
        if self.is_shooting:
            pitch_width = self.pitch.X_MAX - self.pitch.X_MIN
            pitch_height = self.pitch.Y_MAX - self.pitch.Y_MIN
            ball_x_m = ball_x * pitch_width + self.pitch.X_MIN
            ball_y_m = ball_y * pitch_height + self.pitch.Y_MIN
            ball_velocity = np.linalg.norm(self.ball.velocity)

            # Check if the ball is completely out of bounds, scored a goal, or possession lost
            ball_out = self._is_ball_completely_out(ball_x_m, ball_y_m)
            goal = self._is_goal(ball_x_m, ball_y_m)
            possession_lost = self._check_possession_loss()
            ball_stopped = ball_velocity < 0.01

            # End shot and possibly episode
            if goal or ball_out or possession_lost or ball_stopped:
                self.is_shooting = False
                self.shot_direction = None
                self.shot_power = 0.0
                self.shot_position = None

                if goal or ball_out or possession_lost:
                    self.done = True

        # End episode if maximum steps reached
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}



    def _update_ball_position(self, action=None):
        """
        Update the ball's position and velocity based on the current game state.

        - If a shot is in progress (self.is_shooting is True), set the ball's velocity
        according to the shot direction and power (converted to normalized units)
        only once at the start of the shot, then update position applying friction.
        - If the ball is possessed by the attacker, position the ball slightly ahead
        of the attacker in the movement direction to simulate dribbling, and set velocity to zero.
        - If the ball is possessed by the defender, place the ball exactly at the defender's position,
        and set velocity to zero.
        - If the ball is free (no owner), update the position and velocity based on current velocity
        applying friction.

        Args:
            action (np.ndarray, optional): The 2D movement direction vector of the attacker, normalized.
                                        Used only when dribbling to position the ball ahead of the player.
        """
        if self.is_shooting:
            pitch_width = self.pitch.X_MAX - self.pitch.X_MIN
            pitch_height = self.pitch.Y_MAX - self.pitch.Y_MIN

            # Set ball velocity only once at the start of the shot
            if self.shot_just_started:
                # Calculate real shot velocity vector in meters per second
                velocity_real = self.shot_direction * self.shot_power

                # Convert real velocity to normalized velocity (unit per second)
                velocity_norm = np.array([
                    velocity_real[0] / pitch_width,
                    velocity_real[1] / pitch_height
                ])

                # Assign normalized velocity to the ball
                self.ball.set_velocity(velocity_norm)

                # Reset the flag so velocity is not reset every update
                self.shot_just_started = False

            # Update ball position and velocity (apply friction) every step
            self.ball.update(self.time_per_step)

        elif self.ball.owner is self.attacker:
            # Default movement direction is zero
            direction = np.array([0.0, 0.0])

            # If an action is provided, normalize it for direction
            if action is not None:
                direction = np.array([action[0], action[1]])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm

            offset = 0.01  # Small forward offset in normalized coordinates to simulate dribbling

            # Position the ball slightly ahead of the attacker in movement direction
            new_pos = self.attacker.position + direction * offset
            self.ball.set_position(new_pos)

            # While controlled by attacker, ball velocity is zero
            self.ball.set_velocity([0.0, 0.0])

        elif self.ball.owner is self.defender:
            # Position ball exactly at defender's location
            self.ball.set_position(self.defender.position)

            # While controlled by defender, ball velocity is zero
            self.ball.set_velocity([0.0, 0.0])

        else:
            # Ball is free: update position and velocity using physics with friction
            self.ball.update(self.time_per_step)


    def _check_possession_loss(self):
        """
        Check if the defender has stolen the ball from the attacker.
        The defender steals the ball if they are within a certain distance threshold
        and their tackling ability is sufficient.
        If the ball is stolen, update the ball owner accordingly.
        """
        # Get real positions in meters
        def_x, def_y = self.defender.get_position()
        def_x = def_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        def_y = def_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN
        ball_x = self.ball.position[0] * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        ball_y = self.ball.position[1] * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

        threshold = 0.5  # meters
        distance = np.sqrt((def_x - ball_x) ** 2 + (def_y - ball_y) ** 2)

        if distance < threshold:
            if np.random.rand() < self.defender.tackling:
                # Update ball owner to defender (possession lost)
                self.ball.set_owner(self.defender)
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

        position_reward_scale = 4.0
        y_center_penalty_scale = 2.0


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

                    x_reward = -0.5 * position_reward_scale + position_reward_scale * x_norm # range [-2, 2] if position_reward_scale = 4.0
                    y_penalty = -0.5 * y_center_penalty_scale * abs(y_norm - 0.5) * 2 # range [-1, 0] if y_center_penalty_scale = 1.0

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
        margin = 1.0  # 1.0 meters margin for goal area

        GOAL_MIN_Y = self.pitch.CENTER_Y - self.pitch.GOAL_HEIGHT / 2
        GOAL_MAX_Y = self.pitch.CENTER_Y + self.pitch.GOAL_HEIGHT / 2
        return x > self.pitch.FIELD_WIDTH + margin and GOAL_MIN_Y <= y <= GOAL_MAX_Y

    def _is_ball_completely_out(self, ball_x_m, ball_y_m):
        """
        Simple check if ball is outside the real field plus margin, using denormalized coordinates.

        Args:
            ball_x_m (float): Ball's x coordinate in meters.
            ball_y_m (float): Ball's y coordinate in meters.
            pitch: Pitch instance containing field dimensions and constants.

        Returns:
            bool: True if ball is outside field + margin, False otherwise
        """

        margin_m = 2.5  # 2.5 meters margin for out of bounds

        # Check if ball outside real field + margin
        if (ball_x_m < 0 - margin_m or
            ball_x_m > self.pitch.FIELD_WIDTH + margin_m or
            ball_y_m < 0 - margin_m or
            ball_y_m > self.pitch.FIELD_HEIGHT + margin_m):
            return True

        return False

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
        possession = 1.0 if self.ball.owner is self.attacker else 0.0

        # Shooting flag: 1 if shooting in progress, else 0
        is_shooting = 1.0 if self.is_shooting else 0.0

        return np.array([
            att_x, att_y,
            def_x, def_y,
            ball_x, ball_y,
            possession,
            is_shooting
        ], dtype=np.float32)
    
    def _compute_reward(self, shot_flag=None):
        """
        Compute the reward for the current environment step.

        The reward considers several factors:
        - A small time penalty to encourage active play.
        - Position-based reward from a predefined grid to encourage progression and good positioning.
        - Penalty for being out of bounds.
        - Large positive reward for scoring a goal.
        - Penalty for losing ball possession to the defender.
        - Additional one-time bonus reward at the start of a shot,
        encouraging shooting from advantageous positions and directions.
        - Continuous reward during the shot to encourage accurate trajectory and
        progress towards the goal.
        - Final reward at the end of the shot based on shot success and position.
        Args:
            shot_flag (int, optional): Flag indicating if a shot is being attempted.
        Returns:
            float: The computed reward value for the current step.
        """

        reward = 0.0  # Initialize reward accumulator

        # POSITION-BASED REWARD LOGIC

        # Get attacker's normalized position and convert to meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        y_m = att_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN

        # Add position-based reward from the reward grid (encourages good positioning)
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += pos_reward  # Scale down to avoid large swings

        # TERMINAL CONDITIONS REWARD LOGIC

        # Penalize for being out of bound
        if self._is_ball_completely_out(x_m, y_m):
            reward -= 5.0
            return reward

        # Check if a goal has been scored and reward accordingly
        if self._is_goal(x_m, y_m):
            reward += 15.0
            return reward

        # Penalize losing possession of the ball to the defender
        if self._check_possession_loss():
            reward -= 3.0
            return reward

        # SHOOT REWARD LOGIC

        # Penalize if the attacker tries to shoot but is not the owner of the ball
        if shot_flag and self.ball.owner != self.attacker:
            reward -= 2.0
        
        # Penalize if attacker tried to shoot but the shot was invalid (e.g., direction not in FOV)
        # Shot_flag is True if the attacker attempted a shot, but it was not valid since the shot was not initiated
        # This means the shot was not started due to invalid direction or power
        elif shot_flag and not self.is_shooting and self.ball.owner is self.attacker:
            reward -= 1.0  # Penalty for attempting a shot in the wrong direction

        # Convert ball position to meters for shot-related calculations
        ball_x, ball_y = self.ball.position
        ball_x_m = ball_x * (self.pitch.X_MAX - self.pitch.X_MIN) + self.pitch.X_MIN
        ball_y_m = ball_y * (self.pitch.Y_MAX - self.pitch.Y_MIN) + self.pitch.Y_MIN


        # START OF SHOT: REWARD LOGIC
        # Provide an additional one-time shooting bonus only at the very beginning of the shot
        # self.shot_just_started is True only if the attacker is the owner of the ball when the shot is initiated
        if self.is_shooting and self.shot_just_started:

            self.shot_just_started = False  # Reset flag after one-time bonus

            # Bonus reward for starting a shot
            reward += 5.0

            # Bonus for shooting from a good position on the field
            shot_pos_reward = self._get_position_reward(ball_x_m, ball_y_m)
            reward += 7.5 * shot_pos_reward

            # Compute goal direction vector from ball position
            goal_direction = np.array([1.0, 0.5]) - np.array([ball_x, ball_y])
            goal_direction /= np.linalg.norm(goal_direction)

            # Use shot direction if available, else default to goal direction
            shot_dir = self.shot_direction if self.shot_direction is not None else goal_direction

            # Compute cosine similarity (alignment) between shot direction and goal direction
            alignment = np.clip(np.dot(shot_dir, goal_direction), -1, 1)
            alignment = (alignment + 1) / 2.0  # Normalize to [0, 1]

            p = 3  # Power factor for scaling the bonus

            angle_reward = alignment ** p # Skew towards higher values

            # scale to add penalty for misalignment
            angle_reward = 2 * angle_reward - 1  # Scale to [-1, 1]

            # Add angle reward to the total reward
            reward += angle_reward


        # DURING SHOT: CONTINUOUS REWARD LOGIC
        # Provide continuous reward during the shot based on shot direction and power
        #elif self.is_shooting and not self.shot_just_started:

            # Calculate the distance from the ball to the goal in normalized coordinates
            #goal_dir = np.array([1.0, 0.5]) - np.array(self.ball.position)
            #goal_dir /= np.linalg.norm(goal_dir)
            #shot_alignment = np.dot(self.shot_direction, goal_dir)

            # Reward for being aligned with the goal direction
            #reward += 2.0 * self.shot_power * max(0, shot_alignment)

        # END OF SHOT: REWARD LOGIC
        # Provide a reward boost when the shot is finished (ball stopped, out or possession lost)
        if self.is_shooting:
            ball_velocity_norm = np.linalg.norm(self.ball.velocity)

            # Terminal conditions
            ball_completely_out = self._is_ball_completely_out(ball_x_m, ball_y_m)
            goal_scored = self._is_goal(ball_x_m, ball_y_m)
            possession_lost = self._check_possession_loss()
            ball_stopped = ball_velocity_norm < 0.01

            terminal_condition = ball_completely_out or goal_scored or possession_lost

            # Apply penalties
            if ball_completely_out:
                reward -= 5.0
            elif possession_lost:
                reward -= 2.0
            elif ball_stopped and not terminal_condition:
                reward -= 1.0  # shot stopped early but not due to terminal outcome

        # FOV REWARD LOGIC

        # Penalize attempted movement outside the field of view
        # This encourages the agent to only attempt movement in visible directions.
        # Reward bonus if the attacker is moving in a valid direction within FOV
        if hasattr(self, "attempted_movement_direction"):
            direction = self.attempted_movement_direction
            if np.linalg.norm(direction) > 0:
                if self.attacker.is_direction_visible(direction):
                    reward += 0.2  # small positive reward
                else:
                    reward -= 0.1  # existing penalty for bad direction

        return reward


    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass