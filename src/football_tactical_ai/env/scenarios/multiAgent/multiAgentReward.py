from matplotlib.style import context
import numpy as np
from typing import Dict
from football_tactical_ai.players.playerBase import BasePlayer
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.helpers.helperFunctions import denormalize

def get_reward(player: BasePlayer,
               ball: Ball,
               pitch: Pitch,
               reward_grid: np.ndarray,
               context: Dict = None) -> float:
    """
    General reward dispatcher by role.

    Args:
        player (BasePlayer): Player instance (att, def, gk).
        ball (Ball): Ball instance.
        pitch (Pitch): Pitch instance.
        reward_grid (np.ndarray): Grid with position-based reward for this role.
        context (Dict): Action result context (e.g. {'shot': True}, {'tackle': success}).
    """

    role = player.get_role()
    agent_id = player.get_agent_id()

    # Convert normalized player position to meters
    x_norm, y_norm = player.get_position()
    norm_xy = denormalize(x_norm, y_norm)

    # Retrieve reward from spatial grid
    pos_reward = get_position_reward_from_grid(
        pitch, reward_grid, norm_xy[0], norm_xy[1]
    )

    # Group roles into macro categories
    attacker_roles = {"ATT", "LW", "RW", "CF", "LCF", "RCF", "SS"}
    defender_roles = {"DEF", "LCB", "RCB", "CB"}
    goalkeeper_roles = {"GK"}

    # Dispatch logic
    if role in attacker_roles:
        return attacker_reward(agent_id, player, pos_reward, context)
    elif role in defender_roles:
        return defender_reward(agent_id, player, pos_reward, context)
    elif role in goalkeeper_roles:
        return goalkeeper_reward(agent_id, player, pos_reward, context)
    else:
        # Unknown role → neutral reward
        return 0.0



def get_position_reward_from_grid(pitch: Pitch,
                                  reward_grid: np.ndarray,
                                  x_m: float,
                                  y_m: float) -> float:
    """
    Get reward from grid, safely handling boundary cases.
    """
    i = int((x_m - pitch.x_min) / pitch.cell_size)
    j = int((y_m - pitch.y_min) / pitch.cell_size)

    # Clamp indices to stay within grid bounds
    i = max(0, min(i, reward_grid.shape[0] - 1))
    j = max(0, min(j, reward_grid.shape[1] - 1))

    return reward_grid[i, j]


def attacker_reward(agent_id, player, pos_reward, context):
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
        reward -= 2.5

    # Individual reward if player scored
    if context.get("goal_scored", False):
        reward += 10.0 

    # Team reward if any teammate scored
    if context.get("goal_team") == player.team:
        reward += 7.5

    # Penalize if agent kicked the ball out
    if context.get("ball_out_by") == agent_id:
        reward -= 2.5

    # Bonus for starting a pass and positional quality
    if context.get("start_pass_bonus", False):
        reward += 2.5

    # Scaled reward by pass quality 
    if context.get("pass_quality") is not None:
        reward += 2.0 * context.get("pass_quality") 

    # Penalize if pass was attempted but not by the owner
    if context.get("invalid_pass_attempt", False):
        reward -= 0.1

    # Bonus for completed pass
    if context.get("pass_completed", False):
        reward += 5.0  

    # Extra penalty if pass led to ball out
    if context.get("ball_out_by") == agent_id and context.get("pass_attempted", False):
        reward -= 1.0

    # Bonus for starting a shot and positional quality
    if context.get("start_shot_bonus", False):
        reward += 2.5
        reward += context.get("shot_positional_quality", 0.0)

    # Scaled reward by shot quality (0 to 1)
    if context.get("shot_quality") is not None:
        reward += 2.5 * context.get("shot_quality")

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.1

    # Penalize if shot was attempted but not by the owner
    if context.get("invalid_shot_attempt", False):
        reward -= 0.1

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


def defender_reward(agent_id, player, pos_reward, context):
    """
    Defender logic:
    - Reward good positioning (from grid)
    - Bonus for successful tackle
    - Penalty for time and goals conceded (indirect)
    """
    reward = 0.0
    reward += pos_reward
    reward -= 0.02  # time penalty

    # Tackle success (stealing the ball/stopping opponent)
    if context.get("tackle_success", False):
        reward += 10.0

    # Fake tackle penalty
    if context.get("fake_tackle", False):
        reward -= 0.1

    # Reward if ball is out of bounds
    if context.get("ball_out_by") is not None:
        reward += 2.5
        if context.get("ball_out_by") == agent_id:
            reward += 1.0

    # Reward for interception success (take possession or block pass)
    if context.get("interception_success", False):
        reward += 8.0

    # Penalty if goal is conceded
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 10.0

    # Penalize if shot was attempted but not by the owner
    if context.get("invalid_shot_attempt", False):
        reward -= 0.1

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.1

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward

def goalkeeper_reward(agent_id, player, pos_reward, context):
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
    if context.get("blocked", False):
        reward += 10.0

    # Reward for dive even if not successful
    # This encourages goalkeepers to attempt saves
    if context.get("dive_score") is not None and not context.get("wasted_dive", False):
        reward += context.get("dive_score", 0.0) * 5.0  # scale dive success

    # Penalty for wasted dive
    if context.get("wasted_dive", False):
        reward -= 0.1

    # Bonus for deflection
    if context.get("deflected", False):
        reward += 5.0

    # Reward if ball is out of bounds
    if context.get("ball_out_by") is not None:
        reward += 2.5
        if context.get("ball_out_by") == agent_id:
            reward += 1.0

    # Penalty if goal is conceded
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 10.0

    # Penalize if shot was attempted but not by the owner
    if context.get("invalid_shot_attempt", False):
        reward -= 0.1

    # Penalize bad shot direction
    if context.get("invalid_shot_direction", False):
        reward -= 0.1

    # Field of view visibility
    if context.get("fov_visible") is True:
        reward += 0.25
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward