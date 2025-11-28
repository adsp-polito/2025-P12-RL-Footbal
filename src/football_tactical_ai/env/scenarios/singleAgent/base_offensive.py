"""
base_offensive.py

Shared logic for every SINGLE-AGENT offensive scenario (Move, Shot, View)

Child classes MUST override
    • compute_reward(self) -> float

Child classes MAY override
    • _check_termination(self) -> bool
    • after_step(self, action)         
"""
import gymnasium as gym
import numpy as np

from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.players.playerAttacker import PlayerAttacker
from football_tactical_ai.players.playerDefender import PlayerDefender
from football_tactical_ai.helpers.helperFunctions import normalize


class BaseOffensiveScenario(gym.Env):
    """
    Base class for single-agent offensive scenarios in football tactical AI
    Provides common functionality for managing the ball, attacker, and defender,
    as well as episode control and observation/action spaces
    """

    def __init__(self, *, pitch: Pitch, max_steps: int = 360, fps: int = 24) -> None:

        super().__init__()

        # Core objects
        self.pitch: Pitch = pitch
        self.ball: Ball = Ball()
        self.attacker: PlayerAttacker = PlayerAttacker()
        self.defender: PlayerDefender = PlayerDefender()

        # Initialize additional attributes specific to this scenario
        self.num_cells_x = self.pitch.num_cells_x
        self.num_cells_y = self.pitch.num_cells_y
        self.cell_size   = self.pitch.cell_size
        self.reward_grid = None

        # Episode control
        self.max_steps = max_steps
        self.fps = fps
        self.time_per_step = 1.0 / fps
        self._t = 0  # step counter

        # OBSERVATION SPACE: attacker(2) + defender(2) + ball(2) + possession(1) = 7
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)

        # ACTION SPACE: dx, dy in [-1, 1]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Possession loss tracking
        self.possession_lost = False


    def reset(self, *, seed: int | None = None):
        """
        Reset the environment to its initial state, with controlled randomization
        to avoid overfitting while keeping scenarios realistic.
        """
        
        # Set random seed for reproducibility
        super().reset(seed=seed)

        self._t = 0  # reset step counter

        # RANDOMIZED POSITIONS (in metres)
        # Attacker: 20m before → 20m after midfield (40–80m)
        att_x_m = np.random.uniform(40, 80)
        att_y_m = np.random.uniform(20, 60)

        # Defender: full defensive half (60–120m)
        def_x_m = np.random.uniform(60, 120)
        def_y_m = np.random.uniform(20, 60)

        # RESET BALL & PLAYERS
        # Reset attacker & defender positions (normalized)
        self.attacker.reset_position(normalize(att_x_m, att_y_m))
        self.defender.reset_position(normalize(def_x_m, def_y_m))

        # Ball belongs to attacker
        self.ball.set_owner(self.attacker)

        # Ball placed at attacker’s feet:
        # small forward offset in normalized space (same as in dribbling logic)
        offset = 0.01
        direction = np.array([1.0, 0.0])  # default facing right
        ball_pos = normalize(att_x_m, att_y_m) + direction * offset

        self.ball.position[:] = ball_pos
        self.ball.velocity[:] = 0.0

        return self._get_obs(), {}


    def step(self, action):

        # 1) Attacker movement
        self._apply_attacker_action(action, enable_fov=False)

        # 2) Simple defender AI
        self._apply_defender_ai()

        # 3) Ball follows owner / updates physics
        self._update_ball_position(action)

        # 4) Increment step counter
        self._t += 1

        # 5) Build return tuple
        obs = self._get_obs()
        reward, terminated = self.compute_reward()
        if not terminated:
            terminated = self._check_termination()
        truncated = self._t >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _apply_attacker_action(self, action, enable_fov=True) -> None:
        """
        Convert continuous action into attacker movement in normalised space

        We pass field dimensions in metres because PlayerAttacker expects
        them to scale max_speed → normalized delta
        """
        field_width_m  = self.pitch.x_max - self.pitch.x_min
        field_height_m = self.pitch.y_max - self.pitch.y_min

        self.attacker.move_with_action(
            action=action[:2],
            time_per_step=self.time_per_step,
            x_range=field_width_m,
            y_range=field_height_m,
            enable_fov=enable_fov  # Enable FOV for attacker 
        )

        self.last_action = action

    def _apply_defender_ai(self) -> None:
        """
        Very naive chase logic: defender moves towards ball
        """

        # Store old owner to detect possession changes
        old_owner = self.ball.owner

        # Get ball and defender positions
        ball_x, ball_y = self.ball.position
        def_x, def_y = self.defender.get_position()

        # Calculate direction vector from defender to ball
        direction = np.array([ball_x - def_x, ball_y - def_y], dtype=np.float32)

        # Normalize direction vector to unit length
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)
        else:
            direction = np.array([0.0, 0.0])

        # Convert to metres, as PlayerDefender expects field dimensions
        field_width_m  = self.pitch.x_max - self.pitch.x_min
        field_height_m = self.pitch.y_max - self.pitch.y_min

        # Move defender towards attacker
        self.defender.move_with_action(
            direction,
            time_per_step=self.time_per_step,
            x_range=field_width_m,
            y_range=field_height_m,
            enable_fov=False  # Disable FOV for defender AI
        )

        # CHECK TACKLE EVENT
        if old_owner is self.attacker:

            # Calculate distance between defender and ball in metres
            def_x, def_y = self.defender.get_position()
            def_x_m = def_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
            def_y_m = def_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

            ball_x = self.ball.position[0] * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
            ball_y = self.ball.position[1] * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

            distance = np.sqrt((def_x_m - ball_x)**2 + (def_y_m - ball_y)**2)

            # Probabilistic tackle
            if distance < 0.5 and np.random.rand() < self.defender.tackling:
                self.ball.set_owner(self.defender)
                self.possession_lost = True
            else:
                self.possession_lost = False
        else:
            self.possession_lost = False

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

        margin_m = 1.0  # 1.0 meters margin for out of bounds

        # Check if ball outside real field + margin
        if (ball_x_m < 0 - margin_m or
            ball_x_m > self.pitch.width + margin_m or
            ball_y_m < 0 - margin_m or
            ball_y_m > self.pitch.height + margin_m):
            
            return True

        return False

    def _is_goal(self, x, y):
        """
        Check if the ball is in the net
        """
        GOAL_MIN_Y = self.pitch.center_y - self.pitch.goal_width / 2
        GOAL_MAX_Y = self.pitch.center_y + self.pitch.goal_width / 2
        return x > self.pitch.width and GOAL_MIN_Y <= y <= GOAL_MAX_Y

    def _check_possession_loss(self):
        """
        Returns True if possession was lost in THIS step only.
        """
        return self.possession_lost



    def _update_ball_position(self, action=None):
        """
        Update the ball's position based on the current owner.
            - If the attacker owns the ball, position it slightly ahead of the attacker
              in the direction of the last action to simulate dribbling.
            - If the defender owns the ball, position it exactly on the defender.
            - If the ball is free, update its position according to its velocity.
        """
        # If the attacker owns the ball, position it slightly ahead of the attacker
        if self.ball.owner is self.attacker:
            direction = np.array([0.0, 0.0])
            
            # If an action is provided, use it to determine the direction
            if action is not None:
                direction = np.array([action[0], action[1]])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
            
            offset = 0.01  # small offset ahead of attacker to simulate dribbling
            
            # Position ball slightly in front of the attacker based on movement direction
            self.ball.position = self.attacker.position + direction * offset
            self.ball.velocity.fill(0)

        # If the defender owns the ball, position it exactly on the defender
        elif self.ball.owner is self.defender:
            self.ball.position = self.defender.position.copy()
            self.ball.velocity.fill(0)
        
        # If the ball is free (no owner), update its position based on velocity and time step
        else:
            self.ball.update(self.time_per_step)




    def _build_reward_grid(self, min_reward=-0.03, max_reward=0.07):
        """
        Elliptical Gaussian reward grid centered on the goal.

        The reward is highest near the goal center and decreases smoothly
        as elliptical distance increases. This creates a central 'hot zone'
        and discourages lateral paths.

        - Peak reward at goal center (x = pitch.width, y = pitch.center_y)
        - Radially smooth, but vertically tighter (elliptical)
        - Out-of-play cells get min_reward
        """

        pitch = self.pitch

        # Center of reward "hot zone"
        cx = pitch.width - 11.0  # slightly before goal line (penalty spot is the reference)
        cy = pitch.center_y

        # Ellipse radii (in meters)
        # a = horizontal spread, b = vertical tightness
        a = 50.0      # how wide reward spreads horizontally
        b = 35.0      # how narrow reward spreads vertically

        # Best try a = 50, b = 50

        grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

        for i in range(pitch.num_cells_x):
            for j in range(pitch.num_cells_y):

                # Cell center in meters
                cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
                cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

                # OUT OF PLAY → hard penalty
                if cell_x < 0 or cell_x > pitch.width or cell_y < 0 or cell_y > pitch.height:
                    grid[i, j] = min_reward
                    continue

                # Elliptical distance
                dx = cell_x - cx
                dy = cell_y - cy

                dist_ell = np.sqrt((dx*dx) / (a*a) + (dy*dy) / (b*b))

                # Elliptical Gaussian
                reward_raw = np.exp(-(dist_ell**2))

                # Scale to [min_reward, max_reward]
                reward_val = min_reward + (max_reward - min_reward) * reward_raw

                if not np.isfinite(reward_val):
                    reward_val = min_reward

                grid[i, j] = reward_val

        self.reward_grid = grid



    def _get_position_reward(self, x, y, min_reward=-0.03, max_reward=0.07):
        """
        Returns the reward from the reward grid for the attacker's position.
        Builds the grid on first access.
        """
        if self.reward_grid is None:
            self._build_reward_grid(min_reward=min_reward, max_reward=max_reward)

        cell_x = int((x - self.pitch.x_min) / self.cell_size)
        cell_y = int((y - self.pitch.y_min) / self.cell_size)

        cell_x = np.clip(cell_x, 0, self.num_cells_x - 1)
        cell_y = np.clip(cell_y, 0, self.num_cells_y - 1)

        return self.reward_grid[cell_x, cell_y]
    
    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball, and possession status.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        ball_x, ball_y = self.ball.position

        # Possession status: 1 if attacker has possession, 0 otherwise
        possession = 1.0 if self.ball.owner is self.attacker else 0.0

        return np.array([att_x, att_y, def_x, def_y, ball_x, ball_y, possession], dtype=np.float32)
    
    def _check_termination(self) -> bool:
        """
        Determines if the episode should terminate:
        - If goal is scored
        - If ball is completely out of bounds
        - If attacker loses possession to the defender
        - (Max steps handled separately by `truncated`)
        """
        ball_x, ball_y = self.ball.position
        pitch_width = self.pitch.x_max - self.pitch.x_min
        pitch_height = self.pitch.y_max - self.pitch.y_min

        # Convert ball to meters
        ball_x_m = ball_x * pitch_width + self.pitch.x_min
        ball_y_m = ball_y * pitch_height + self.pitch.y_min

        goal = self._is_goal(ball_x_m, ball_y_m)
        out_of_bounds = self._is_ball_completely_out(ball_x_m, ball_y_m)
        lost_possession = (self.ball.owner is self.defender)

        return goal or out_of_bounds or lost_possession

    # Mandatory / optional methods to override
    def compute_reward(self) -> float:         # must override
        raise NotImplementedError