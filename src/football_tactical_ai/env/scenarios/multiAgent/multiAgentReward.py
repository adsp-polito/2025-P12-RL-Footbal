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
        return attacker_reward(agent_id, player, pos_reward, ball, context)
    elif role in defender_roles:
        return defender_reward(agent_id, player, pos_reward, ball, context)
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


def attacker_reward(agent_id, player, pos_reward, ball, context):
    """
    Attacker reward function:
    - Prioritizes goals and successful passes.
    - Encourages useful positioning and meaningful attempts.
    - Penalizes possession loss and out-of-play.
    """

    reward = 0.0

    # BASE: positioning + time penalty
    reward += pos_reward                     # small dense feedback
    reward -= 0.005                           # time malus (avoid stalling)

    # BALL CHASING
    if ball is not None and ball.get_owner() is None:
        # Distance from player to ball (in meters)
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Inverse distance reward (closer = better)
        # Positive reward up to ~20 meters, then decays to 0
        reward += max(0.0, 0.1 - 0.005 * dist_to_ball)    # range [0, 0.1]

    # POSSESSION / OUT
    if context.get("possession_lost", False):
        reward -= 0.5                       # losing the ball
    if context.get("ball_out_by") == agent_id:
        reward -= 1.0                       # kicking ball out

    # GOALS
    if context.get("goal_scored", False):
        reward += 8.0                       # scoring
    if context.get("goal_team") == player.team:
        reward += 6.0                       # team goal bonus (shared reward)

    # PASSING
    if context.get("start_pass_bonus", False):
        reward += 0.2                        # small bonus for attempting pass

    if context.get("pass_quality") is not None:
        reward += 0.5 * context["pass_quality"]   # scaled by quality (0–0.5)

    if context.get("invalid_pass_attempt", False):
        reward -= 0.05                      

    if context.get("pass_completed", False):
        # Reward both passer and receiver
        if "pass_to" in context:             # receiver
            reward += 0.2
        elif "pass_from" in context:         # passer
            reward += 0.4
        # Small team bonus for cooperation
        reward += 0.2

    # SHOOTING
    if context.get("start_shot_bonus", False):
        reward += 0.5                        # reward intent
        reward += context.get("shot_positional_quality", 0.0)

    if context.get("shot_quality") is not None:
        reward += 0.5 * context["shot_quality"]   # scaled 0–0.5

    if context.get("invalid_shot_direction", False):
        reward -= 0.05

    if context.get("invalid_shot_attempt", False):
        reward -= 0.05

    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += alignment ** 2 - 0.25  # shaping [-0.25, 0.75]

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.1
    elif context.get("fov_visible") is False:
        reward -= 0.05

    return reward

def defender_reward(agent_id, player, pos_reward, ball, context):
    """
    Defender reward function:
    - Prioritizes preventing goals.
    - Rewards interceptions, tackles, and forcing ball out.
    - Penalizes conceding goals or useless actions.
    """

    reward = 0.0

    # BASE
    reward += pos_reward            # dense shaping
    reward += 0.002                 # small time survival bonus

    # BALL CHASING (less aggressive than attackers)
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Smaller incentive (up to ~20 meters)
        reward += max(0.0, 0.1 - 0.005 * dist_to_ball)   # range [0, 0.1]

    # DEFENSIVE ACTIONS
    if context.get("tackle_success", False):
        reward += 0.8          # strong bonus for winning ball
    if context.get("interception_success", False):
        reward += 0.7            # slightly less than tackle

    # FAKE / FAILED ACTIONS
    if context.get("fake_tackle", False):
        reward -= 0.1
    if context.get("invalid_shot_attempt", False):
        reward -= 0.05
    if context.get("invalid_shot_direction", False):
        reward -= 0.05

    # BALL OUT
    if context.get("ball_out_by") is not None:
        reward += 0.3             # forcing out = good
        if context.get("ball_out_by") == agent_id:
            reward += 0.2         # if you caused it

    # GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 6.0        # conceding is worst outcome

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.1
    elif context.get("fov_visible") is False:
        reward -= 0.05

    return reward


def goalkeeper_reward(agent_id, player, pos_reward, context):
    """
    Goalkeeper reward function:
    - Prioritizes saves and keeping clean sheet.
    - Encourages positioning inside goal area.
    - Penalizes conceding goals.
    """

    reward = 0.0

    # BASE
    reward += pos_reward          # grid shaping
    reward += 0.002               # small time survival bonus

    # SAVES
    if context.get("blocked", False):
        reward += 1.5            # successful save
    if context.get("deflected", False):
        reward += 0.8            # save via deflection
    if context.get("dive_score") is not None and not context.get("wasted_dive", False):
        reward += 0.5 * context["dive_score"]  # scaled [0, 0.5]

    # FAILED ACTIONS
    if context.get("wasted_dive", False):
        reward -= 0.1
    if context.get("invalid_shot_attempt", False):
        reward -= 0.05
    if context.get("invalid_shot_direction", False):
        reward -= 0.05

    # BALL OUT
    if context.get("ball_out_by") is not None:
        reward += 0.3           # putting ball out = acceptable

    # GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 6.0        # conceding goal

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.1
    elif context.get("fov_visible") is False:
        reward -= 0.05

    return reward
