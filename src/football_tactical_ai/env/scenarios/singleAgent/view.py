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

        # LOGGING METRICS FOR EVALUATION ONLY
        self.last_valid_shot = None
        self.last_shot_distance = None
        self.last_time_to_shot = None
        self.last_shot_angle = None
        self.last_shot_power = None
        self.last_reward_components = None

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

        self.attacker.last_action_direction = np.array([1.0, 0.0])  # Reset last action direction

        # Reset logging metrics
        self.last_valid_shot = None
        self.last_shot_distance = None
        self.last_time_to_shot = None
        self.last_shot_angle = None
        self.last_shot_power = None
        self.last_reward_components = None
        self.last_shot_start_bonus = 0.0

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

            # A shot starts only if actual_power > 0 (direction inside FOV)
            if actual_power > 0.0:
                self.is_shooting = True
                self.shot_direction = actual_dir
                self.shot_power = actual_power
                self.shot_position = self.ball.position.copy()
                self.shot_just_started = True
                self.ball.set_owner(None)

                # LOG METRICS
                self.last_valid_shot = True
                self.last_shot_power = float(actual_power)
                self.valid_shots += 1

                # Compute shot angle relative to the goal
                att_x, att_y = self.attacker.get_position()
                vec = np.array([self.goal_x - att_x, self.goal_y - att_y])
                vec /= np.linalg.norm(vec)
                self.last_shot_angle = float(np.arccos(np.clip(np.dot(actual_dir, vec), -1, 1)))

                # Compute shot distance
                pitch_w = self.pitch.x_max - self.pitch.x_min
                pitch_h = self.pitch.y_max - self.pitch.y_min
                bx = self.shot_position[0] * pitch_w + self.pitch.x_min
                by = self.shot_position[1] * pitch_h + self.pitch.y_min
                self.last_shot_distance = float(np.linalg.norm([self.goal_x - bx, self.goal_y - by]))

                # Convert to normalized ball velocity
                velocity_real = actual_dir * actual_power
                velocity_norm = np.array([velocity_real[0] / pitch_w,
                                        velocity_real[1] / pitch_h])
                self.ball.set_velocity(velocity_norm)
            else:
                # Shot was invalid due to FOV
                self.last_valid_shot = False
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


    
    def compute_reward(self, shot_flag=None, shot_quality=0.0, movement=np.array([0.0, 0.0])):
        """
        Compute the scalar reward for the current timestep.

        The reward integrates positional incentives, visual constraints,
        penalties for invalid actions, and bonuses for well-formed shots.
        Termination events such as goals, out-of-bounds, and possession loss
        are handled elsewhere through the environment's termination function.

        Args:
            shot_flag (bool): Whether the agent attempted to shoot in this step.
            shot_quality (float): Shooting alignment/quality from PlayerAttacker.shoot().

        Returns:
            reward (float): Total reward for this timestep.
        """

        reward = 0.0

        # Small constant penalty for encouraging fast decision-making
        reward -= 0.01

        # Attacker real position in meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        y_m = att_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        # Ball real position in meters
        ball_x, ball_y = self.ball.position
        bx_m = ball_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        by_m = ball_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        # Positional reward based on spatial reward grid
        pos_reward = self._get_position_reward(x_m, y_m)
        reward += 0.5 * pos_reward

        # READINESS REWARD → encourages moving toward the goal before shooting
        vec_to_goal = np.array([self.goal_x - x_m, self.goal_y - y_m])
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)

        movement_norm = movement / (np.linalg.norm(movement) + 1e-6)

        readiness = np.dot(movement_norm, vec_to_goal)   # [-1,+1]

        reward += 0.05 * readiness # in [-0.05, +0.05]

        # BONUS FOR GOAL
        if self._is_goal(bx_m, by_m):
            reward += 7.5

        # PENALTY FOR POSSESSION LOSS
        if self.possession_lost_this_step:
            reward -= 2.5

        # PENALTY FOR OUT OF BOUNDS
        if self._is_ball_completely_out(bx_m, by_m):
            reward -= 2.5

        # INVALID SHOT ATTEMPT
        if shot_flag and self.ball.owner is not self.attacker:
            reward -= 0.25

        # Penalty for attempted shots that were invalid (e.g., outside FOV)
        elif shot_flag and not self.is_shooting:
            reward -= 0.25

        # Shot start bonuses and alignment shaping
        if self.is_shooting and self.shot_just_started:
            self.shot_just_started = False

            # Shot quality bonus (only active when a valid shot was initiated)
            reward += shot_quality  # in [0, +1]

            # Small encouragement to initiate shots
            self.last_shot_start_bonus = 0.5
            reward += self.last_shot_start_bonus
            
            # SHOT ALIGNMENT
            # Alignment reward
            goal_dir = np.array([self.goal_x - bx_m, self.goal_y - by_m])
            norm = np.linalg.norm(goal_dir)
            if norm > 1e-6:
                goal_dir /= norm
            else:
                goal_dir = np.array([1.0, 0.0])

            shot_dir = self.shot_direction if self.shot_direction is not None else goal_dir
            alignment = float(np.clip(np.dot(shot_dir, goal_dir), -1, 1))

            # Non-linear shaping
            sharpness = 4
            alignment_shaped = np.sign(alignment) * (abs(alignment) ** sharpness)

            reward += alignment_shaped  # in [-1, +1]

            # penalty for very poor alignment
            if alignment < 0.5:
                reward -= 1.0

            # DISTANCE TO GOAL REWARD
            dist = np.linalg.norm([self.goal_x - x_m, self.goal_y - y_m])
            max_dist = np.linalg.norm([
                self.pitch.x_max - self.pitch.x_min,
                0.5 * (self.pitch.y_max - self.pitch.y_min),
            ])

            # Normalized distance ∈ [0,1], 1 = near goal
            dist_norm = 1 - dist / max_dist
            dist_norm = np.clip(dist_norm, 0.0, 1.0)

            # Exponential shaping
            sharpness = 5
            dist_shaped = dist_norm ** sharpness

            # Map to [-1, +1]
            dist_scaled = 2 * dist_shaped - 1

            reward += dist_scaled # in [-1, +1]      

        # Reward or penalty depending on whether movement direction was visible
        # For each direction, so dense reward signal (low magnitude to avoid overpowering main rewards)
        if hasattr(self, "attempted_movement_direction"):
            direction = self.attempted_movement_direction
            if np.linalg.norm(direction) > 0:
                if self.attacker.is_direction_visible(direction):
                    reward += 0.05     # movement inside FOV is encouraged
                else:
                    reward -= 0.05      # movement outside FOV is discouraged

        # LOGGING: time-to-shot (distance / shot power)
        if self.is_shooting and self.last_shot_distance is not None and self.last_shot_power is not None:
            if self.last_shot_power > 0:
                self.last_time_to_shot = self.last_shot_distance / self.last_shot_power
            else:
                self.last_time_to_shot = None
        else:
            self.last_time_to_shot = None

        # FINAL EPISODE BONUS: reward if the episode ended after taking at least one shot
        if self._t == self.max_steps - 1 or self._check_termination():
            if self.last_shot_start_bonus > 0:
                # Episode ends with at least one shot attempted
                reward += 6.0
            else:
                # Episode ends without ever shooting
                reward -= 7.5

        self.last_reward_components = {
            "position": float(pos_reward),
            "shot_start_bonus": float(self.last_shot_start_bonus) if self.is_shooting else 0.0,
            "goal": 7.5 if self._is_goal(bx_m, by_m) else 0.0,
            "possession_lost": -2.5 if self._check_possession_loss() else 0.0,
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

        return obs

    def close(self):
            """
            Close the environment and release resources.
            """
            # No specific resources to release in this environment
            # but this method is here for compatibility with gym's API
            pass