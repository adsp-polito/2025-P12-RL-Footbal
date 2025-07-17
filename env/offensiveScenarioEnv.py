import gymnasium as gym
import numpy as np

from players.playerBase import Player
from env.ball import Ball
from helpers.visuals import render_state
from env.pitch import CELL_SIZE

# Rendering settings
FPS = 24

# === Pitch bounds and normalization constants ===
X_MIN = -5
X_MAX = 125  # Total pitch width = 130
Y_MIN = -5
Y_MAX = 85   # Total pitch height = 90
PITCH_WIDTH = X_MAX - X_MIN
PITCH_HEIGHT = Y_MAX - Y_MIN

# Goal specs (in absolute coordinates)
GOAL_WIDTH = 7.32
GOAL_X_MIN = 120  
GOAL_Y_MIN = (Y_MAX+Y_MIN)//2 - GOAL_WIDTH / 2    # Y_MAX + Y_MIN = 80 because y starts at -5 and end at 85 (85+(-5)=80)
GOAL_Y_MAX = (Y_MAX+Y_MIN)//2 + GOAL_WIDTH / 2

class OffensiveScenarioEnv(gym.Env):
    """
    A simplified RL environment representing an offensive football scenario on half field.

    The field includes buffer space around the official 120x80 pitch, 
    extending X: [-5, 125], Y: [-5, 85] for reward and boundary logic.
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ACTION SPACE
        self.action_space = gym.spaces.Discrete(9)

        # OBSERVATION SPACE: normalized (px, py, has_ball, bx, by)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Entities
        self.player = Player(player_id=0)
        self.ball = Ball()
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose a fixed starting point (in absolute field coordinates)
        # The extended pitch spans from x ∈ [55, 130], y ∈ [-5, 85] in meters

        # random position within the extended half field
        #start_x = np.random.uniform(55, 130)  # meters
        #start_y = np.random.uniform(-5, 85)  # meters

        # Initial Position 
        start_x = 65
        start_y = 40

        # Safety check: ensure the start point is within the extended pitch limits
        if not (55 <= start_x <= 130) or not (-5 <= start_y <= 85):
            raise ValueError(f"Start position out of bounds: ({start_x}, {start_y})")

        # Normalize absolute coordinates to [0, 1] range
        norm_x = (start_x - X_MIN) / PITCH_WIDTH
        norm_y = (start_y - Y_MIN) / PITCH_HEIGHT

        # Reset the player and ball positions
        self.player.reset(start_pos=(norm_x, norm_y))

        self.ball.reset()
        self.ball.set_position(self.player.get_position())
        self.player.has_ball = True
        self.ball.owner_id = self.player.player_id

        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        # Check if the episode is done
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Step and smooth movement
        self.player.step(action)
        self.player.update_position()

        # Ball follows player
        if self.player.has_ball:
            offset = self.player.get_ball_offset()
            self.ball.set_position(self.player.get_position() + offset)

        # Convert normalized to absolute coordinates
        px, py = self.ball.get_position()
        abs_x = px * PITCH_WIDTH + X_MIN
        abs_y = py * PITCH_HEIGHT + Y_MIN

        # Initialize reward and info
        reward = 0.0
        info = {}

        # Terminal conditions
        # Out of bounds (If x < 60 or x > 130, y < 0 or y > 80)
        if abs_x < PITCH_WIDTH//2 or abs_x > PITCH_WIDTH or abs_y < 0 or abs_y > PITCH_HEIGHT:
            reward = -10.0
            self.done = True
            info["out_of_bounds"] = True

        # Goal scored
        elif abs_x >= GOAL_X_MIN and GOAL_Y_MIN < abs_y < GOAL_Y_MAX:
            reward = 5.0
            self.done = True
            info["goal"] = True

        # Proximity to goal
        else:
            # Distance-based reward shaping (scaled to stay << 1.0)
            goal_center = np.array([GOAL_X_MIN, (Y_MAX + Y_MIN) / 2])
            ball_pos = np.array([abs_x, abs_y])
            dist_to_goal = np.linalg.norm(ball_pos - goal_center)

            # Compute max possible distance from goal
            max_distance = np.linalg.norm(np.array([X_MIN, Y_MIN]) - goal_center)

            # Reward ∈ [0, proximity_scale]
            proximity_scale = 0.2  # Max shaping reward if right at goal center
            proximity_reward = proximity_scale * (1.0 - (dist_to_goal / max_distance))
            proximity_reward = max(proximity_reward, 0.0)

            reward = proximity_reward
            info["proximity_reward"] = reward

        return self._get_obs(), reward, self.done, False, info

    def _get_obs(self):
        px, py = self.player.get_position()
        bx, by = self.ball.get_position()
        return np.array([px, py, 1.0 if self.player.has_ball else 0.0, bx, by], dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            render_state({
                "player": self.player,
                "ball": self.ball,
                "opponents": []
            }, show_grid=True, show_cell_ids=False)

    def close(self):
        pass