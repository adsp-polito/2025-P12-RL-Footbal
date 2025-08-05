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
    Args:
        pos_reward (float): Position-based reward from grid.
        context (Dict): Action result context (e.g. {'start_shot_bonus': True}, {'goal': scored}).
    Returns:
        float: Computed reward for the attacker.
    """

    # Initialize reward
    reward = 0.0
    # Add positional reward
    reward += pos_reward
    # Time penalty
    reward -= 0.02

    # Penalize if possession is lost
    if context.get("possession_lost", False):
        reward -= 1.0

    # Individual reward if player scored
    if context.get("goal_scored", False):
        reward += 10.0 

    # Team reward if any teammate scored
    if context.get("goal_team") == player.team:
        reward += 8.0

    # Penalize if agent kicked the ball out
    if context.get("ball_out_by") == agent_id:
        reward -= 2.5

    # FOV + Shot logic unchanged

    # Bonus for starting a shot and positional quality
    if context.get("start_shot_bonus", False):
        reward += 2.5
        reward += context.get("shot_positional_quality", 0.0)

    # Scaled reward by shot quality (0 to 1)
    if context.get("shot_quality") is not None:
        reward += 2.5 * context.get("shot_quality")

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Penalize if shot was attempted but not by the owner
    if context.get("not_owner_shot_attempt", False):
        reward -= 0.5

    # Angle reward (dot product with goal direction)
    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += (2 * (alignment ** 3)) - 1  # [-1, 1]

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward


def defender_reward(agent_id, player, ball, pitch, pos_reward, context):
    """
    Defender logic:
    - Reward good positioning (from grid)
    - Bonus for successful tackle
    - Penalty for time and goals conceded (indirect)
    """
    reward = 0.0
    reward += pos_reward
    reward -= 0.02  # time penalty

    # Tackle success
    if context.get("tackle_success", False):
        reward += 10.0

    # Reward if ball is out of bounds
    if context.get("ball_out_by") is not None:
        reward += 2.5
        if context.get("ball_out_by") == agent_id:
            reward += 1.0

    # Penalty if attacker scores (indirect defensive failure)
    if context.get("goal_team") and context.get("goal_team") != player.team:
        reward -= 5.0  # team conceded

    # Penalize if shot was attempted but not by the owner
    if context.get("not_owner_shot_attempt", False):
        reward -= 0.5

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward

def goalkeeper_reward(agent_id, player, ball, pitch, pos_reward, context):
    """
    Goalkeeper logic:
    - Reward for staying in correct area (grid)
    - Bonus for successful save
    - Penalty for conceding
    - Penalty for being too far from goal center (via grid)
    """
    reward = 0.0
    reward += pos_reward

    # Save success
    if context.get("save_success", False):
        reward += 10.0

    # Reward for successful dive
    if context.get("dive_success", False):
        reward += 5.0

    # Bonus for deflection
    if context.get("deflected", False):
        deflection_power = context.get("deflection_power", 0.0)
        reward += 2.0 * deflection_power  # scale based on power

    # Reward if ball is out of bounds
    if context.get("ball_out_by") is not None:
        reward += 2.5
        if context.get("ball_out_by") == agent_id:
            reward += 1.0

    # Penalty if attacker scores (indirect defensive failure)
    if context.get("goal_scored") and context.get("goal_team") != player.team:
        reward -= 5.0  # team conceded

    # Penalize if shot was attempted but not by the owner
    if context.get("not_owner_shot_attempt", False):
        reward -= 0.5

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward