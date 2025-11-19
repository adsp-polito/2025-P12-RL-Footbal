import numpy as np
from gymnasium import spaces
from football_tactical_ai.env.scenarios.singleAgent.base_offensive import BaseOffensiveScenario
from football_tactical_ai.helpers.helperFunctions import normalize, denormalize

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

        # Goal center (meters)
        self.goal_x = self.pitch.x_max  
        self.goal_y = (self.pitch.y_min + self.pitch.y_max) / 2



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
        self.shot_pos_rew = None            # Positional reward at shot initiation (used for reward)

        # LOGGING METRICS FOR EVALUATION ONLY
        self.valid_shot = None
        self.shot_distance = None
        self.shot_step = None
        self.shot_angle = None
        self.shot_power = None
        self.reward_components = None



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
        self.shot_pos_rew = None

        # Reset logging metrics
        self.valid_shot = None
        self.shot_distance = None
        self.shot_step = None
        self.shot_angle = None
        self.shot_power = None
        self.reward_components = None
        self.shot_start_bonus = 0.0

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

        # Unpack action
        movement = action[:2]
        shot_flag = (action[2] > 0.5)
        shot_power = float(action[3])
        shot_dir = action[4:6]

        shot_quality = 0.0  # Default shot quality

        # 1) SHOOT LOGIC
        if shot_flag and not self.is_shooting and self.ball.owner is self.attacker:

            # perform shot
            shot_quality, actual_dir, actual_power = self.attacker.shoot(
                desired_direction=shot_dir,
                desired_power=shot_power,
                enable_fov=False,
            )

            self.is_shooting = True
            self.shot_direction = actual_dir
            self.shot_power = actual_power
            self.shot_position = self.ball.position.copy()
            self.shot_just_started = True
            self.ball.set_owner(None)

            # LOG metrics
            self.valid_shot = True
            self.shot_power = actual_power

            # shot angle
            att_x, att_y = self.attacker.get_position()
            vec = np.array([self.goal_x - att_x, self.goal_y - att_y])
            vec /= np.linalg.norm(vec)
            self.shot_angle = float(
                np.arccos(np.clip(np.dot(actual_dir, vec), -1, 1))
            )

            # shot distance
            pitch_w = self.pitch.x_max - self.pitch.x_min
            pitch_h = self.pitch.y_max - self.pitch.y_min
            bx = self.shot_position[0] * pitch_w + self.pitch.x_min
            by = self.shot_position[1] * pitch_h + self.pitch.y_min
            self.shot_pos_rew = self._get_position_reward(bx, by)   # in [-0.03, 0.07]

            # shot initial velocity
            vel_real = actual_dir * actual_power
            vel_norm = np.array([vel_real[0]/pitch_w, vel_real[1]/pitch_h])
            self.ball.set_velocity(vel_norm)

        else:
            # normal movement + dribbling
            self._apply_attacker_action(action, enable_fov=False)
            self._update_ball_position(movement)

        # 2) BALL PHYSICS
        if self.is_shooting or self.ball.owner is None:
            self.ball.update(self.time_per_step)

        # 3) DEFENDER AI
        self._apply_defender_ai()

        # 4) REWARD
        reward = self.compute_reward(shot_flag=shot_flag, shot_quality=shot_quality, movement=movement)

        # 5) TERMINATION
        terminated = self._check_termination()
        self._t += 1
        truncated = self._t >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}
        
    

        
    def compute_reward(self, shot_flag=None, shot_quality=0.0, movement=None):
        """
        Compute the reward for the current step.
        Balances movement (advancement), decision-making (when to shoot),
        and shot quality (direction, angle, and landing position).
        """

        reward = 0.0

        # SMALL TIME PENALTY ==> encourages fast decision-making
        reward -= 0.01

        # REAL POSITIONS IN METERS (attacker and ball)
        att_x, att_y = self.attacker.get_position()
        x_m, y_m = denormalize(att_x, att_y)

        ball_x, ball_y = self.ball.position
        bx_m, by_m = denormalize(ball_x, ball_y)

        # POSITIONAL REWARD
        min_r, max_r = -0.03, 0.07
        pos_reward = self._get_position_reward(x_m, y_m,
                                            min_reward=min_r,
                                            max_reward=max_r)
        reward += pos_reward

        # READINESS ==> encourages moving toward the goal direction
        if movement is None:
            movement = np.array([0.0, 0.0], dtype=float)
        else:
            movement = np.asarray(movement, dtype=float)

        vec_to_goal = np.array([self.goal_x - x_m, self.goal_y - y_m], dtype=float)
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)

        if np.linalg.norm(movement) > 1e-6:
            movement_norm = movement / np.linalg.norm(movement)
        else:
            movement_norm = np.array([0.0, 0.0], dtype=float)

        readiness = float(np.dot(movement_norm, vec_to_goal))  # ∈ [-1, 1]

        # Scale readiness more strongly when positional reward is low
        pos_norm = np.clip((pos_reward - min_r) / (max_r - min_r), 0.0, 1.0)
        readiness_weight = 1.0 - pos_norm

        if self.ball.owner is self.attacker:
            reward += readiness_weight * 0.05 * readiness

        # EVENT-BASED REWARDS ==> goals, possession loss, ball out
        if self._is_goal(bx_m, by_m) and self.ball.get_owner() is not self.attacker:
            reward += 10.0

        if self.possession_lost:
            reward -= 2.5

        if self._is_ball_completely_out(bx_m, by_m):
            reward -= 2.5

        if shot_flag and self.ball.owner is not self.attacker:
            reward -= 0.25
            self.shot_start_bonus = 0.0

        # SHOOTING REWARD
        if self.is_shooting and self.shot_just_started:

            reward += shot_quality

            # Shot initiation bonus
            self.shot_start_bonus = 0.3
            reward += self.shot_start_bonus

            # Directions
            goal_dir = np.array([self.goal_x - bx_m, self.goal_y - by_m], dtype=float)
            goal_dir /= (np.linalg.norm(goal_dir) + 1e-6)

            shot_dir = (
                self.shot_direction if self.shot_direction is not None else goal_dir.copy()
            )
            shot_dir /= (np.linalg.norm(shot_dir) + 1e-6)

            # Angle shot→goal
            cos_angle = np.clip(np.dot(shot_dir, goal_dir), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # GAUSSIAN ANGLE REWARD
            # More reward as angle ==> 0°
            sigma = np.radians(28.0)
            angle_reward = float(np.exp(-(angle ** 2) / (2.0 * sigma ** 2)))
            reward += 5.0 * angle_reward

            # SOFT PENALTY FOR BAD ANGLES 
            bad_angle_penalty = (angle / np.pi) ** 2      # ∈ [0, 1]
            reward -= 2.0 * bad_angle_penalty

            # DISTANCE-BASED SHAPING (closer = better)
            dist = float(np.linalg.norm([self.goal_x - x_m, self.goal_y - y_m]))

            
            max_dist = float(np.linalg.norm([
                self.pitch.x_max - self.pitch.x_min,
                0.5 * (self.pitch.y_max - self.pitch.y_min),
            ]))
            dist_norm = 1.0 - dist / (max_dist + 1e-6)
            dist_norm = float(np.clip(dist_norm, 0.0, 1.0))

            dist_reward = -0.2 + dist_norm * 0.8
            reward += dist_reward

            # LOGGING METRICS
            self.shot_distance = dist
            self.shot_step = self._t

        # FINAL EPISODE BONUS ==> use ball landing and final shot angle
        terminated_now = self._check_termination()
        if self._t == self.max_steps - 1 or terminated_now:

            if getattr(self, "shot_start_bonus", 0.0) > 0.0:

                # Base final bonus
                reward += 3.0 + 5.0 * (self.shot_pos_rew * 10.0)

                # Vertical landing accuracy (where the ball ends)
                dy_final = abs(by_m - self.goal_y)
                goal_half_h = self.pitch.goal_width / 2.0
                dy_norm = dy_final / (goal_half_h + 1e-6)

                center_bonus = float(1.0 - np.clip(dy_norm, 0.0, 1.0))
                reward += 3.0 * center_bonus

                goal_dir = np.array([self.goal_x - bx_m, self.goal_y - by_m], dtype=float)
                goal_dir /= (np.linalg.norm(goal_dir) + 1e-6)

                # Final angular reward (how good was shot direction)
                if self.shot_direction is not None and np.linalg.norm(self.shot_direction) > 1e-6:
                    final_shot_dir = self.shot_direction / np.linalg.norm(self.shot_direction)

                    final_cos = np.clip(np.dot(final_shot_dir, goal_dir), -1.0, 1.0)
                    final_angle = np.arccos(final_cos)

                    final_angle_reward = np.exp(-(final_angle**2) / (2 * np.radians(30)**2))
                    reward += 2.0 * final_angle_reward

                    final_penalty = (final_angle / np.pi) ** 2
                    reward -= 1.5 * final_penalty

            else:
                reward -= 7.5    # episode ended with no shot

        # LOGGING (for analysis and debugging)
        self.reward_components = {
            "position": float(pos_reward),
            "shot_start_bonus": float(getattr(self, "shot_start_bonus", 0.0)),
            "goal": 10.0 if self._is_goal(bx_m, by_m) else 0.0,
            "possession_lost": -2.5 if self.possession_lost else 0.0,
            "time_penalty": -0.01,
        }

        return float(reward)






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

    
    
    
    def _check_termination(self):
        """
        Override: termination logic for the Shot scenario.
        Includes also: ball stopped while shooting.
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