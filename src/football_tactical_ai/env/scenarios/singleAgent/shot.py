import numpy as np
from gymnasium import spaces
from football_tactical_ai.env.scenarios.singleAgent.base_offensive import BaseOffensiveScenario
from football_tactical_ai.helpers.helperFunctions import normalize

class OffensiveScenarioShotSingleAgent(BaseOffensiveScenario):
    """
    Gym environment simulating a 1v1 football offensive scenario with shooting mechanics.

    The attacker can move and attempt shots towards the goal.
    The defender tries to regain possession by following the ball.
    The environment manages:
      - Movement of attacker and defender in normalized pitch coordinates.
      - Ball movement during dribbling and shooting phases.
      - Shooting simulation including direction and power.
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

        # Reset the random seed if provided
        # By doing this, we use the reset method from the parent class so 
        # the ball and players are initialized correctly
        super().reset(seed=seed)

        # Reset shooting state
        self.is_shooting = False
        self.shot_direction = None
        self.shot_power = 0.0
        self.shot_position = None
        self.shot_just_started = False

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
                enable_fov=False  # Disable FOV check for shooting
            )

            # Initiate the shot
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

        else:  
            # Attacker movement
            self._apply_attacker_action(action, enable_fov=False)

            # Update ball position to follow attacker (dribbling)
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

            # Check if ball stopped moving (velocity is zero)
            ball_stopped = ball_velocity_norm == 0

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
            # Set ball velocity only once at the start of the shot
            if self.shot_just_started:

                pitch_width = self.pitch.x_max - self.pitch.x_min
                pitch_height = self.pitch.y_max - self.pitch.y_min

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
        reward = 0.0
        terminated = False

        # BASE POSITIONAL REWARD
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        y_m = att_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        pos_reward = self._get_position_reward(x_m, y_m)
        reward += pos_reward

        # SMALL TIME PENALTY (to encourage active play)
        reward -= 0.01

        # GOAL
        if self._is_goal(x_m, y_m):
            reward += 7.5      
            terminated = True
            return reward, terminated

        # POSSESSION LOST
        if self._check_possession_loss():
            reward -= 2.5
            terminated = True
            return reward, terminated

        # INVALID SHOT ATTEMPT (not owning ball)
        if shot_flag and self.ball.owner != self.attacker:
            reward -= 0.25

        # BALL POSITION FOR SHOT LOGIC
        ball_x, ball_y = self.ball.position
        ball_x_m = ball_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        ball_y_m = ball_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        # START OF SHOT
        if self.is_shooting and self.shot_just_started:
            self.shot_just_started = False

            # SMALL shot start bonus
            reward += 0.15

            # Positional quality boost (scaled down)
            shot_pos_reward = self._get_position_reward(ball_x_m, ball_y_m)
            reward += 0.5 * shot_pos_reward

            # Alignment computation
            goal_direction = np.array([1.0, 0.5]) - np.array([ball_x, ball_y])
            goal_direction /= np.linalg.norm(goal_direction)

            shot_dir = self.shot_direction if self.shot_direction is not None else goal_direction
            alignment = np.clip(np.dot(shot_dir, goal_direction), -1, 1)
            alignment = (alignment + 1) / 2.0

            angle_reward = (alignment**3)
            angle_reward = 2*angle_reward - 1   # [-1,1]
            angle_reward *= 0.5                # scale down [-0.5, 0.5]

            reward += angle_reward

        # END OR INTERRUPTED SHOT
        if self.is_shooting:
            ball_velocity_norm = np.linalg.norm(self.ball.velocity)
            ball_out = self._is_ball_completely_out(ball_x_m, ball_y_m)
            goal_scored = self._is_goal(ball_x_m, ball_y_m)
            possession_lost = self._check_possession_loss()
            ball_stopped = ball_velocity_norm < 0.01

            if goal_scored:
                reward += 7.5      # consistent with goal reward
                terminated = True

            elif ball_out:
                reward -= 2.5      # moderate penalty
                terminated = True

            elif possession_lost:
                reward -= 2.5      # consistent with possession loss penalty
                terminated = True

            elif ball_stopped:
                reward -= 0.5      # mild penalty for weak shot

            # Reset shot state
            self.is_shooting = False
            self.shot_direction = None
            self.shot_power = 0.0
            self.shot_position = None

        return reward, terminated


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


    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass