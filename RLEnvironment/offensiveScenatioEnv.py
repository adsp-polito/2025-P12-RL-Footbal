import gymnasium as gym
import numpy as np

from RLEnvironment.player import Player
from RLEnvironment.ball import Ball
from RLEnvironment.pitch import render_state, CELL_SIZE as PITCH_CELL_SIZE

# Global constants
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
HALF_FIELD_X = FIELD_WIDTH // 2
CENTER_Y = FIELD_HEIGHT // 2
CELL_SIZE = 5  # Must match pitch.py

# Check CELL_SIZE match to avoid grid alignment errors
assert CELL_SIZE == PITCH_CELL_SIZE, f"CELL_SIZE mismatch: {CELL_SIZE} (env) != {PITCH_CELL_SIZE} (pitch)"

# Goal dimensions (match visual pitch)
GOAL_WIDTH = 7.32  # Standard goal width in meters
GOAL_Y_MIN = CENTER_Y - GOAL_WIDTH / 2  # Lower Y of goal
GOAL_Y_MAX = CENTER_Y + GOAL_WIDTH / 2  # Upper Y of goal
GOAL_X_MIN = FIELD_WIDTH - 1  # Ball must cross at least this X to count as goal

# Environment description
class OffensiveScenarioEnv(gym.Env):
    """
    A minimal RL environment for offensive football scenarios on half field.
    One player must move toward the goal and attempt to score.
    Inspired by the mouse-and-cheese metaphor: the agent (mouse) must reach the goal (cheese).
    """

    # Metadata for Gymnasium
    metadata = {"render_modes": ["human"], "render_fps": 24}

    # Initialize the environment
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ACTION SPACE
        # 0 = idle, 1-8 = 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.action_space = gym.spaces.Discrete(9)

        # OBSERVATION SPACE
        # [player_x, player_y, has_ball, ball_x, ball_y] → normalized
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # INITIALIZE PLAYER AND BALL
        self.player = Player(player_id=0, team_id=0, side="left")
        self.ball = Ball()

        self.done = False  # Episode end flag

    # Reset the environment to its initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset positions (player starts center-left)
        self.player.reset()
        self.player.set_position((HALF_FIELD_X + 5, CENTER_Y))

        self.ball.reset()
        self.ball.set_position(self.player.get_position())

        # Possession
        self.player.has_ball = True
        self.ball.owner_id = self.player.player_id

        self.done = False
        return self._get_obs(), {}

    # Step the environment with the given action
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Move player
        self.player.step(action)

        # Move ball with player if possessed
        if self.player.has_ball:
            self.ball.set_position(self.player.get_position())

        # Check for ball possession
        reward = 0.0
        x, y = self.ball.get_position()

        # Goal scored
        if x >= GOAL_X_MIN and GOAL_Y_MIN <= y <= GOAL_Y_MAX:
            reward = 1.0
            self.done = True
            info = {"goal": True}

        # Ball out of bounds
        elif x > FIELD_WIDTH or y < 0 or y > FIELD_HEIGHT:
            reward = -1.0
            self.done = True
            info = {"out_of_bounds": True}

        else:
            # Reward shaping: progress toward goal line
            reward = (x - HALF_FIELD_X) / (FIELD_WIDTH - HALF_FIELD_X)
            info = {}

        return self._get_obs(), reward, self.done, False, info

    # Get the current observation
    def _get_obs(self):
        """Return normalized observation vector"""
        px, py = self.player.get_position()
        bx, by = self.ball.get_position()
        return np.array([
            px / FIELD_WIDTH,
            py / FIELD_HEIGHT,
            1.0 if self.player.has_ball else 0.0,
            bx / FIELD_WIDTH,
            by / FIELD_HEIGHT
        ], dtype=np.float32)

    # Render the environment
    def render(self):
        """Visualize the current state using the pitch rendering utilities"""
        if self.render_mode == "human":
            render_state({
                "player": self.player,
                "ball": self.ball,
                "opponents": []
            }, show_grid=True, show_cell_ids=False)

    def close(self):
        pass