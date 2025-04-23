import gymnasium as gym
import numpy as np

from RLEnvironment.player import Player
from RLEnvironment.ball import Ball
from helpers.visuals import render_state
from RLEnvironment.pitch import CELL_SIZE

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
GOAL_X_MIN = 119  # 1 meter before end of official pitch (still within field)
GOAL_Y_MIN = 40 - GOAL_WIDTH / 2
GOAL_Y_MAX = 40 + GOAL_WIDTH / 2

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
        start_x = 62.5  # meters (within the extended half field)
        start_y = 42.5  # meters (center vertically)

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
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Step and smooth movement
        self.player.step(action)
        self.player.update_position()

        # Ball follows player
        if self.player.has_ball:
            offset = self.player.get_ball_offset()
            self.ball.set_position(self.player.get_position() + offset)

        # Convert from normalized to absolute coordinates
        px, py = self.ball.get_position()
        abs_x = px * PITCH_WIDTH + X_MIN
        abs_y = py * PITCH_HEIGHT + Y_MIN

        reward = 0.0
        info = {}

        if abs_x < X_MIN or abs_x > X_MAX or abs_y < Y_MIN or abs_y > Y_MAX:
            reward = -1.0
            self.done = True
            info["out_of_bounds"] = True

        elif abs_x >= GOAL_X_MIN and GOAL_Y_MIN <= abs_y <= GOAL_Y_MAX:
            reward = 1.0
            self.done = True
            info["goal"] = True

        else:
            # Shaped reward (from middle of field at 60m)
            progress = (abs_x - 60) / (120 - 60)
            reward = max(progress, 0)
            info["progress_reward"] = reward

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