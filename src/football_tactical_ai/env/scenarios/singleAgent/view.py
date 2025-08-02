import numpy as np
from gymnasium import spaces
from football_tactical_ai.env.scenarios.singleAgent.base_offensive import BaseOffensiveScenario
from football_tactical_ai.helpers.helperFunctions import normalize

class OffensiveScenarioViewSingleAgent(BaseOffensiveScenario):
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
        super().__init__(pitch=pitch, max_steps=max_steps, fps=fps)

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
        # Reset the random seed if provided
        # By doing this, we use the reset method from the parent class so 
        # the ball and players are initialized correctly
        super().reset(seed=seed)

        # Reset shooting state
        self.is_shooting = False
        self.shot_direction = None
        self.shot_power = 0.0
        self.shot_position = None
        self.shot_just_started = False  # Reset for new episode

        self.attacker.last_action_direction = np.array([1.0, 0.0])  # Reset last action direction

        # Return initial observation and info dictionary as required by Gym API
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (np.ndarray): Action vector containing:
                - movement_x_y: 2D vector for player movement in [-1, 1]
                - shot_flag: continuous float in [0, 1] indicating if a shot is being attempted
                - shot_power: continuous float in [0, 1] for shot power
                - shot_direction: 2D vector in [-1, 1] for shot direction
        Returns:
            obs (np.ndarray): Current observation of the environment.
            reward (float): Reward for the current step.
            terminated (bool): True if the episode has ended due to a goal or possession loss.
            truncated (bool): True if the episode has ended due to reaching max steps.
            info (dict): Additional information about the environment state.
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
                desired_power=shot_power,
                enable_fov=True  # Enable FOV check for shooting
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
                pitch_width = self.pitch.x_max - self.pitch.x_min
                pitch_height = self.pitch.y_max - self.pitch.y_min
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
            
            # Attacker movement
            self._apply_attacker_action(action)

            # Update ball position only if attacker still owns it
            if self.ball.owner is self.attacker:
                self._update_ball_position(movement_action)

        # Update ball movement
        if self.is_shooting or self.ball.owner is None:
            # Ball moves autonomously only after shot
            self.ball.update(self.time_per_step)

        # Defender always moves towards the ball
        self._apply_defender_ai()

        # Compute reward (all logic handled internally)
        reward, terminated = self.compute_reward(shot_flag=shot_flag)

        # Check if shot finished (only if currently shooting)
        if self.is_shooting:

            # Get current ball position in normalized coordinates
            ball_x, ball_y = self.ball.position

            # Convert ball position to meters and compute velocity norm
            pitch_width = self.pitch.x_max - self.pitch.x_min
            pitch_height = self.pitch.y_max - self.pitch.y_min
            ball_x_m = ball_x * pitch_width + self.pitch.x_min
            ball_y_m = ball_y * pitch_height + self.pitch.y_min
            ball_velocity_norm = np.linalg.norm(self.ball.velocity)

            # Check if ball is completely out of bounds
            ball_completely_out = self._is_ball_completely_out(ball_x_m, ball_y_m)

            # Check if ball in goal
            goal_scored = self._is_goal(ball_x_m, ball_y_m)

            # Check if possession lost
            possession_lost = self._check_possession_loss()

            # Check if ball stopped moving (velocity is near zero)
            ball_stopped = ball_velocity_norm < 0.01

            # If any terminal condition met, end shot
            if goal_scored or possession_lost or ball_completely_out or ball_stopped:
                self.is_shooting = False
                self.shot_direction = None
                self.shot_power = 0.0
                self.shot_position = None

        # Increment step counter
        self._t += 1

        # Build return tuple
        obs = self._get_obs()
        if not terminated:
            terminated = self._check_termination()
        
        truncated = self._t >= self.max_steps

        # Check for NaN values in observation or reward
        if np.any(np.isnan(obs)) or np.isnan(reward):
            print("[WARNING] NaN detected in observation or reward. Resetting episode.")
            obs = np.nan_to_num(obs, nan=0.0)
            reward = -50.0  # Penality for NaN state
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {}

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

    
    def compute_reward(self, shot_flag=None):
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
        terminated = False  # Initialize termination flag

        # TIME PENALTY LOGIC

        # Apply a small time penalty to encourage active play
        reward -= 0.02  # Small penalty for each step to encourage efficiency

        # POSITION-BASED REWARD LOGIC

        # Get attacker's normalized position and convert to meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        y_m = att_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        # Add position-based reward from the reward grid (encourages good positioning)
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += pos_reward  # Scale the position reward to encourage good positioning

        # TERMINAL CONDITIONS REWARD LOGIC

        # Check if a goal has been scored and reward accordingly
        if self._is_goal(x_m, y_m):
            reward += 5.0
            terminated = True  # End episode on goal
            return reward, terminated

        # Penalize losing possession of the ball to the defender
        if self._check_possession_loss():
            reward -= 1.0
            terminated = True # End episode on possession loss
            return reward, terminated

        # SHOOT REWARD LOGIC

        # Penalize if the attacker tries to shoot but is not the owner of the ball
        if shot_flag and self.ball.owner != self.attacker:
            reward -= 0.5
        
        # Penalize if attacker tried to shoot but the shot was invalid (e.g., direction not in FOV)
        # Shot_flag is True if the attacker attempted a shot, but it was not valid since the shot was not initiated
        # This means the shot was not started due to invalid direction or power
        elif shot_flag and not self.is_shooting and self.ball.owner is self.attacker:
            reward -= 0.25  # Penalty for attempting a shot in the wrong direction

        # Convert ball position to meters for shot-related calculations
        ball_x, ball_y = self.ball.position
        ball_x_m = ball_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        ball_y_m = ball_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min


        # START OF SHOT: REWARD LOGIC
        # Provide an additional one-time shooting bonus only at the very beginning of the shot
        # self.shot_just_started is True only if the attacker is the owner of the ball when the shot is initiated
        if self.is_shooting and self.shot_just_started:

            self.shot_just_started = False  # Reset flag after one-time bonus

            # Bonus reward for starting a shot
            reward += 2.5

            # Bonus for shooting from a good position on the field
            shot_pos_reward = self._get_position_reward(ball_x_m, ball_y_m)
            reward += 3.0 * shot_pos_reward

            # Compute goal direction vector from ball position
            goal_direction = np.array([1.0, 0.5]) - np.array([ball_x, ball_y])
            norm = np.linalg.norm(goal_direction)
            if norm < 1e-6:
                goal_direction = np.array([1.0, 0.0])  # fallback
            else:
                goal_direction /= norm

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
            reward += angle_reward # Scale the angle reward to encourage good shooting angles


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
                reward -= 1.5
                terminated = True  # End episode on ball out
            elif possession_lost:
                reward -= 1.0
                terminated = True
            elif ball_stopped and not terminal_condition:
                reward -= 0.25  # shot stopped early but not due to terminal outcome

        # FOV REWARD LOGIC

        # Penalize attempted movement outside the field of view
        # This encourages the agent to only attempt movement in visible directions.
        # Reward bonus if the attacker is moving in a valid direction within FOV
        if hasattr(self, "attempted_movement_direction"):
            direction = self.attempted_movement_direction
            if np.linalg.norm(direction) > 0:
                if self.attacker.is_direction_visible(direction):
                    reward += 0.25  # small positive reward
                else:
                    reward -= 0.1  # penalty for bad direction

        return reward, terminated
    
    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball,
        possession status, shooting flag, and shot progress.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        ball_x, ball_y = self.ball.position

        # Possession status: 1 if attacker has possession, 0 otherwise
        possession = 1.0 if self.ball.owner is self.attacker else 0.0

        # Shooting flag: 1 if shooting in progress, else 0
        is_shooting = 1.0 if self.is_shooting else 0.0

        obs = np.array([
        att_x, att_y,
        def_x, def_y,
        ball_x, ball_y,
        possession,
        is_shooting
        ], dtype=np.float32)

        # Check for NaN or out-of-bounds values in the observation
        if np.any(np.isnan(obs)) or np.any(obs < 0.0) or np.any(obs > 1.0):
            print(f"[NaN or out-of-bounds OBS at step {self._t}]")
            print(f"att_pos: {(att_x, att_y)}, def_pos: {(def_x, def_y)}, ball_pos: {(ball_x, ball_y)}")
            print(f"poss: {possession}, is_shooting: {is_shooting}")
            print(f"OBS: {obs}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        return obs


    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass