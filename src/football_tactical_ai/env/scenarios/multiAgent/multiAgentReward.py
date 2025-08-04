from matplotlib.style import context
import numpy as np
from typing import Dict
from football_tactical_ai.players.playerBase import BasePlayer
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.helpers.helperFunctions import denormalize

# Memory of last grid rewards for each agent
_last_grid_reward = {}


def get_reward(agent_id: str,
               player: BasePlayer,
               ball: Ball,
               pitch: Pitch,
               reward_grid: np.ndarray,
               context: Dict = None) -> float:
    """
    General reward dispatcher by role.

    Args:
        agent_id (str): The agent's unique identifier.
        player (BasePlayer): Player instance (att, def, gk).
        ball (Ball): Ball instance.
        pitch (Pitch): Pitch instance.
        reward_grid (np.ndarray): Grid with position-based reward for this role.
        context (Dict): Action result context (e.g. {'shot': True}, {'tackle': success}).
    """
    role = player.get_role()

    # Convert normalized player position to meters
    x_norm, y_norm = player.get_position()
    x_m = denormalize(x_norm, pitch.x_min, pitch.x_max)
    y_m = denormalize(y_norm, pitch.y_min, pitch.y_max)

    # Retrieve reward from spatial grid
    pos_reward = get_position_reward_from_grid(
        pitch, reward_grid, x_m, y_m, agent_id
    )

    # Dispatch logic by role
    if role == "ATT":
        return attacker_reward(agent_id, player, ball, pitch, pos_reward, context)
    elif role == "DEF":
        return defender_reward(agent_id, player, ball, pitch, pos_reward, context)
    elif role == "GK":
        return goalkeeper_reward(agent_id, player, ball, pitch, pos_reward, context)
    else:
        return 0.0


def get_position_reward_from_grid(pitch: Pitch,
                                  reward_grid: np.ndarray,
                                  x_m: float,
                                  y_m: float,
                                  agent_id: str,
                                  idle_penalty: float = -0.25) -> float:
    """
    Get reward from grid, penalize if agent is stuck on same reward value.
    """
    i = int((x_m - pitch.x_min) / pitch.cell_size)
    j = int((y_m - pitch.y_min) / pitch.cell_size)

    if not (0 <= i < reward_grid.shape[0] and 0 <= j < reward_grid.shape[1]):
        return -5.0

    current_reward = reward_grid[i, j]

    # Check if same as previous
    if agent_id in _last_grid_reward:
        if np.isclose(current_reward, _last_grid_reward[agent_id], atol=1e-6):
            return idle_penalty

    _last_grid_reward[agent_id] = current_reward
    return current_reward

def attacker_reward(agent_id, player, ball, pitch, pos_reward, context):
    """
    Advanced attacker logic: encourage movement, shooting, goal scoring, and positioning.
    """
    reward = 0.0
    reward += pos_reward
    reward -= 0.02  # time penalty

    if context.get("possession_lost", False):
        reward -= 1.0

    # Shooting logic
    if context.get("shot_attempted", False):
        reward += 0.25

    # Reward for scoring a goal
    if context.get("goal_scored", False):
        print("[INFO] Goal scored!")
        reward += 10.0
    else:
        reward -= 0.1  # Small penalty for not scoring

    # Scaled reward by shot quality (0 to 1)
    shot_quality = context.get("shot_quality")
    if shot_quality is not None:
        reward += 2.5 * shot_quality

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Penalize if shot was attempted but not by the owner
    if context.get("not_owner_shot_attempt", False):
        reward -= 0.5

    # Angle reward (dot product with goal direction)
    alignment = context.get("shot_alignment")
    if alignment is not None:
        # alignment in [0, 1], skewed to reward higher values
        angle_reward = (2 * (alignment ** 3)) - 1  # range [-1, 1]
        reward += angle_reward

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward


def defender_reward(agent_id, player, ball, pitch, pos_reward, context):
    """
    Simple defender logic: reward good positioning and tackle success.
    """
    reward = 0.0
    reward += pos_reward
    reward -= 0.02  # time penalty

    if context.get("tackle_success", False):
        reward += 10.0

    return reward


def goalkeeper_reward(agent_id, player, ball, pitch, pos_reward, context):
    """
    Goalkeeper logic: reward staying in position and saving goals.
    """
    reward = 0.0
    reward += pos_reward

    if context.get("save_success", False):
        reward += 10.0

    if context.get("goal_conceded", False):
        reward -= 5.0

    return reward
