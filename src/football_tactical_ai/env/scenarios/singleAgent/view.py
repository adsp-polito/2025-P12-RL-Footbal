import numpy as np
from gymnasium import spaces
from football_tactical_ai.env.scenarios.singleAgent.base_offensive import BaseOffensiveScenario
from football_tactical_ai.helpers.helperFunctions import normalize, denormalize

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

        # Goal center (meters)
        self.goal_x = self.pitch.x_max  
        self.goal_y = (self.pitch.y_min + self.pitch.y_max) / 2

        # Action space: (movement_x_y, shot_flag, shot_power, shot_direction)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),  # x,y movement, shot_flag, shot_power, shot_dir x,y
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: normalized player positions (x,y), ball position (x,y), and shooting state (direction x,y + power)
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
        self.shot_in_fov = False            # Flag indicating if the shot was initiated within FOV

        # LOGGING METRICS FOR EVALUATION ONLY
        self.valid_shot = None
        self.shot_distance = None
        self.shot_step = None
        self.shot_angle = None
        self.shot_power = None
        self.reward_components = None

        # FOV logging metrics
        self.fov_valid_movements = 0
        self.fov_invalid_movements = 0
        self.invalid_shot_fov = 0
        self.total_shots = 0
        self.valid_shots = 0
        self.valid_shot_ratio = 0.0


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
        self.shot_in_fov = False

        self.attacker.last_action_direction = np.array([1.0, 0.0])  # Reset last action direction

        # Reset logging metrics
        self.valid_shot = None
        self.shot_distance = None
        self.shot_step = None
        self.shot_angle = None
        self.shot_power = None
        self.reward_components = None
        self.shot_start_bonus = 0.0

        # Reset FOV metrics
        self.fov_valid_movements = 0
        self.fov_invalid_movements = 0
        self.invalid_shot_fov = 0
        self.total_shots = 0
        self.valid_shots = 0
        self.valid_shot_ratio = 0.0

        # Return initial observation and info dictionary as required by Gym API
        return self._get_obs(), {}
    


    def step(self, action):
        """
        Execute one simulation step in the environment.

        The attacker may move or attempt a shot, but both actions are subject
        to visual constraints. A shot can only start if the desired direction
        lies within the attacker's field of view (FOV). Movement is also
        conditioned by visibility: if the intended direction is not visible,
        the attacker will not move.

        Args:
            action (np.ndarray): Continuous action vector containing:
                - action[0:2]: movement direction in [-1, 1]
                - action[2]:    shot flag (continuous, thresholded at 0.5)
                - action[3]:    desired shot power in [0, 1]
                - action[4:6]:  desired shot direction in [-1, 1]

        Returns:
            observation (np.ndarray)
            reward (float)
            terminated (bool)
            truncated (bool)
            info (dict)
        """

        # Unpack action components
        movement = action[:2]
        shot_flag = (action[2] > 0.5)
        desired_power = float(action[3])
        desired_dir = action[4:6]

        # Default shot quality (used only when a shot is validly started)
        shot_quality = 0.0

        # Attempt to start a shot only if attacker has the ball and no shot is active
        if shot_flag and not self.is_shooting and self.ball.owner is self.attacker:
            shot_quality, actual_dir, actual_power = self.attacker.shoot(
                desired_direction=desired_dir,
                desired_power=desired_power,
                enable_fov=True            # Visual constraint specific to this scenario
            )

            # LOGGING: count total shot attempts
            self.total_shots += 1

            # Check FOV alignment 
            # It is used desired_dir to evaluate the intent of the agent
            self.shot_in_fov = self.attacker.is_direction_visible(desired_direction=desired_dir)

            # A shot starts only if it is within the FOV
            if self.shot_in_fov:
                self.is_shooting = True
                self.shot_direction = actual_dir
                self.shot_power = actual_power
                self.shot_position = self.ball.position.copy()
                self.shot_just_started = True
                self.ball.set_owner(None)

                # LOG METRICS
                self.valid_shot = True
                self.shot_power = float(actual_power)
                self.valid_shots += 1

                # Compute shot angle relative to the goal
                att_x, att_y = self.attacker.get_position()
                vec = np.array([self.goal_x - att_x, self.goal_y - att_y])
                vec /= np.linalg.norm(vec)
                self.shot_angle = float(np.arccos(np.clip(np.dot(actual_dir, vec), -1, 1)))

                # Compute shot distance
                pitch_w = self.pitch.x_max - self.pitch.x_min
                pitch_h = self.pitch.y_max - self.pitch.y_min
                bx = self.shot_position[0] * pitch_w + self.pitch.x_min
                by = self.shot_position[1] * pitch_h + self.pitch.y_min
                self.shot_distance = float(np.linalg.norm([self.goal_x - bx, self.goal_y - by]))

                # Convert to normalized ball velocity
                velocity_real = actual_dir * actual_power
                velocity_norm = np.array([velocity_real[0] / pitch_w,
                                        velocity_real[1] / pitch_h])
                self.ball.set_velocity(velocity_norm)
            else:
                # Shot was invalid due to FOV
                # Count how many invalid shots were due to FOV
                self.valid_shot = False
                self.invalid_shot_fov += 1
        
        # Update ratio of valid shots
        self.valid_shot_ratio = self.valid_shots / max(self.total_shots, 1) if self.total_shots > 0 else 0.0

        # Movement direction is stored for FOV reward logic
        self.attempted_movement_direction = movement.copy()

        # Apply movement only if direction is visible
        if np.linalg.norm(movement) > 0:
            if self.attacker.is_direction_visible(movement):    # FOV CHECK
                self.fov_valid_movements += 1
                self._apply_attacker_action(action, enable_fov=True)
                if self.ball.owner is self.attacker:
                    self._update_ball_position(movement)
            else:
                self.fov_invalid_movements += 1


        # Update ball physics: either free or in shooting trajectory
        if self.is_shooting or self.ball.owner is None:
            self.ball.update(self.time_per_step)

        # Defender AI moves toward the ball
        self._apply_defender_ai()

        # Compute reward for the current state transition
        reward = self.compute_reward(shot_flag=shot_flag, shot_quality=shot_quality, movement=movement)

        # Termination check performed after applying all state updates
        self._t += 1
        terminated = self._check_termination()
        truncated = self._t >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}



    
    def compute_reward(self, shot_flag=None, shot_quality=0.0, movement=None):
        """
        Compute the reward for the current step in the FOV-constrained shot scenario

        This version is based on the plain Shot scenario, with additional shaping
        to encourage:
        - moving inside the field of view (FOV)
        - shooting only when the target direction is visible

        Args:
            shot_flag (bool): Whether a shot was attempted this step
            shot_quality (float): Quality of the shot attempted (if any)
            movement (np.ndarray): 2D movement vector of the attacker (action space)

        Returns:
            reward (float): Computed reward for the current step
        """

        reward = 0.0

        # Small constant time-penalty for encouraging fast decision-making
        reward -= 0.01

        # REAL POSITIONS IN METERS (attacker + ball)
        att_x, att_y = self.attacker.get_position()
        x_m, y_m = denormalize(att_x, att_y)

        ball_x, ball_y = self.ball.position
        bx_m, by_m = denormalize(ball_x, ball_y)

        # BASE POSITIONAL REWARD (scaled down)
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += 0.5 * pos_reward

        # READINESS + MOVEMENT FOV SHAPING
        # Encourages moving TOWARD a better shooting position before shooting
        # ONLY applied when attacker has possession
        vec_to_goal = np.array([self.goal_x - x_m, self.goal_y - y_m])
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)

        # Movement direction from the action (attempted movement)
        if movement is None:
            movement = np.array([0.0, 0.0], dtype=float)
        movement = np.asarray(movement, dtype=float)
        movement_norm = movement / (np.linalg.norm(movement) + 1e-6)

        readiness = float(np.dot(movement_norm, vec_to_goal))   # ∈ [-1, +1]

        # Direction used for FOV visibility check:
        # if we have an explicit attempted_movement_direction, prefer that
        move_dir = movement_norm.copy()
        if hasattr(self, "attempted_movement_direction"):
            dir_vec = np.asarray(self.attempted_movement_direction, dtype=float)
            if np.linalg.norm(dir_vec) > 1e-6:
                move_dir = dir_vec / np.linalg.norm(dir_vec)

        # Is the attempted movement direction inside FOV?
        move_visible = self.attacker.is_direction_visible(move_dir)

        if self.ball.owner is self.attacker:
            if move_visible:
                # Same readiness term as Shot, but only when inside FOV
                reward += 0.05 * readiness
                # Small dense bonus for actually trying to move within FOV
                reward += 0.02 * np.linalg.norm(movement_norm)
            else:
                # Small penalty for trying to move outside FOV
                reward -= 0.01

        # BASE EVENT REWARDS (goal, possession loss, out-of-bounds)
        # BONUS FOR GOAL
        if self._is_goal(bx_m, by_m):
            reward += 15.0  # same scale as Shot

        # PENALTY FOR POSSESSION LOSS
        if self.possession_lost:
            reward -= 3.0

        # PENALTY FOR OUT OF BOUNDS (ball exits field)
        if self._is_ball_completely_out(bx_m, by_m):
            reward -= 3.0

        # SHOT VALIDITY + FOV-CONSTRAINED SHOT SHAPING
        shot_in_fov = getattr(self, "shot_in_fov", True)

        if shot_flag:
            # Standard invalid shot (attacker is not the owner of the ball  → cannot shoot)
            if self.ball.owner is not self.attacker:
                reward -= 0.25
            else:
                # Agent has the ball and attempts a shot
                if not shot_in_fov:
                    # Explicit penalty for shooting out of FOV
                    reward -= 0.25
                else:
                    # Small bonus for respecting the FOV constraint when shooting
                    reward += 0.2

        # START OF SHOT: QUALITY, ALIGNMENT, DISTANCE (with FOV modulation)
        if self.is_shooting and self.shot_just_started:

            # SHOT QUALITY REWARD
            reward += shot_quality          # in [0, 1], direct addition

            # Reward for initiating the shot
            self.shot_start_bonus = 0.3
            reward += self.shot_start_bonus

            # SOFT ALIGNMENT REWARD MODULATED BY FOV
            goal_dir = np.array([self.goal_x - bx_m, self.goal_y - by_m])
            gn = np.linalg.norm(goal_dir)
            if gn > 1e-6:
                goal_dir /= gn
            else:
                goal_dir = np.array([1.0, 0.0])

            shot_dir = (
                self.shot_direction
                if self.shot_direction is not None
                else goal_dir
            )

            alignment = float(np.clip(np.dot(shot_dir, goal_dir), -1, 1))

            # Compute the base alignment bonus as in the Shot scenario
            alignment_bonus = 0.0
            if alignment > 0.7:
                alignment_bonus = 0.4
            elif alignment > 0.4:
                alignment_bonus = 0.1
            elif alignment < 0.1:
                alignment_bonus = -0.2  # soft penalty

            # FOV modulation: strong effect if the shot is inside FOV,
            # attenuated effect if it is outside.
            if shot_in_fov:
                reward += alignment_bonus
            else:
                reward += 0.5 * alignment_bonus  # still matters, but less

            # DISTANCE TO GOAL REWARD
            dist = np.linalg.norm([self.goal_x - x_m, self.goal_y - y_m])
            max_dist = np.linalg.norm([
                self.pitch.x_max - self.pitch.x_min,
                0.5 * (self.pitch.y_max - self.pitch.y_min),
            ])

            dist_norm = 1 - dist / max_dist      # ∈ [0,1]
            dist_norm = np.clip(dist_norm, 0.0, 1.0)

            # Very soft shaping to avoid dominating the reward
            reward += 0.3 * dist_norm   # ∈ [0, 0.3]

            # LOGGING: TIME OF SHOT IN SECONDS
            if self.is_shooting:
                self.shot_step = self._t
            else:
                self.shot_step = None

        # FINAL EPISODE BONUS / PENALTY
        terminated_now = self._check_termination()
        if self._t == self.max_steps - 1 or terminated_now:
            if self.shot_start_bonus > 0:
                # Episode ends: at least one valid shot taken (under FOV constraints)
                reward += 5.0
            else:
                # Episode ends: no shots taken
                reward -= 5.0

        # LOGGING: LAST REWARD COMPONENTS
        self.reward_components = {
            "position": float(pos_reward),
            "shot_start_bonus": float(self.shot_start_bonus),
            "goal": 15.0 if self._is_goal(bx_m, by_m) else 0.0,
            "possession_lost": -3.0 if self.possession_lost else 0.0,
        }

        return reward


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
            pitch_width = self.pitch.x_max - self.pitch.x_min
            pitch_height = self.pitch.y_max - self.pitch.y_min

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

    def _check_termination(self):
        """
        Override: termination logic for the View scenario
        Includes also: ball stopped while shooting
        """

        # Convert ball pos to meters
        ball_x, ball_y = self.ball.position
        pw = self.pitch.x_max - self.pitch.x_min
        ph = self.pitch.y_max - self.pitch.y_min
        bx = ball_x * pw + self.pitch.x_min
        by = ball_y * ph + self.pitch.y_min

        # Base termination events
        if self._is_goal(bx, by):
            return True

        if self._is_ball_completely_out(bx, by):
            return True

        if self.ball.owner is self.defender:
            return True

        # EXTRA: ball stopped → end of shot
        if self.is_shooting and np.linalg.norm(self.ball.velocity) < 0.01:
            return True

        return False
    
    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball,
        possession status, shooting flag, and shot progress.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        ball_x, ball_y = self.ball.get_position()

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

        return obs

    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass