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
    Attacker reward function:
    - Prioritizes goals and successful passes.
    - Encourages useful positioning and meaningful attempts.
    - Penalizes possession loss and out-of-play.
    """

    reward = 0.0

    # BASE: positioning + time penalty
    reward += pos_reward                     # small dense feedback
    reward -= 0.02                           # time malus (avoid stalling)

    # POSSESSION / OUT
    if context.get("possession_lost", False):
        reward -= 5.0                        # losing the ball
    if context.get("ball_out_by") == agent_id:
        reward -= 7.5                        # kicking ball out

    # GOALS
    if context.get("goal_scored", False):
        reward += 25.0                       # scoring
    if context.get("goal_team") == player.team:
        reward += 20.0                       # team goal bonus (shared reward)

    # PASSING
    if context.get("start_pass_bonus", False):
        reward += 2.5                        # small bonus for attempting pass

    if context.get("pass_quality") is not None:
        reward += 5.0 * context["pass_quality"]   # scaled by quality (0–5)

    if context.get("invalid_pass_attempt", False):
        reward -= 0.5                        

    if context.get("pass_completed", False):
        # Reward both passer and receiver
        if "pass_to" in context:             # passer
            reward += 12.0
        elif "pass_from" in context:         # receiver
            reward += 10.0
        # Small team bonus for cooperation
        reward += 2.0

    if context.get("ball_out_by") == agent_id and context.get("pass_attempted", False):
        reward -= 3.0                        # attempted pass → out

    # SHOOTING
    if context.get("start_shot_bonus", False):
        reward += 3.0                        # reward intent
        reward += context.get("shot_positional_quality", 0.0)

    if context.get("shot_quality") is not None:
        reward += 5.0 * context["shot_quality"]   # scaled 0–5

    if context.get("invalid_shot_direction", False):
        reward -= 0.5

    if context.get("invalid_shot_attempt", False):
        reward -= 0.5

    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += (2 * (alignment ** 3)) - 1  # shaping [-1, 1]

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.5
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward

def defender_reward(agent_id, player, pos_reward, context):
    """
    Defender reward function:
    - Prioritizes preventing goals.
    - Rewards interceptions, tackles, and forcing ball out.
    - Penalizes conceding goals or useless actions.
    """

    reward = 0.0

    # BASE
    reward += pos_reward          # dense shaping
    reward += 0.01                # small time survival bonus

    # DEFENSIVE ACTIONS
    if context.get("tackle_success", False):
        reward += 15.0            # strong bonus for winning ball
    if context.get("interception_success", False):
        reward += 12.0            # slightly less than tackle

    # FAKE / FAILED ACTIONS
    if context.get("fake_tackle", False):
        reward -= 0.5
    if context.get("invalid_shot_attempt", False):
        reward -= 0.5
    if context.get("invalid_shot_direction", False):
        reward -= 0.5

    # BALL OUT
    if context.get("ball_out_by") is not None:
        reward += 5.0             # forcing out = good
        if context.get("ball_out_by") == agent_id:
            reward += 2.0         # if you caused it

    # GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 25.0        # conceding is worst outcome

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.5
    elif context.get("fov_visible") is False:
        reward -= 0.1

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
    reward += 0.01                # small time survival bonus

    # SAVES
    if context.get("blocked", False):
        reward += 15.0            # successful save
    if context.get("deflected", False):
        reward += 10.0            # save via deflection
    if context.get("dive_score") is not None and not context.get("wasted_dive", False):
        reward += context["dive_score"] * 7.5   # scaled by dive quality

    # FAILED ACTIONS
    if context.get("wasted_dive", False):
        reward -= 0.5
    if context.get("invalid_shot_attempt", False):
        reward -= 0.5
    if context.get("invalid_shot_direction", False):
        reward -= 0.5

    # BALL OUT
    if context.get("ball_out_by") is not None:
        reward += 3.0             # putting ball out = acceptable

    # GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 25.0        # conceding goal

    # FIELD OF VIEW
    if context.get("fov_visible") is True:
        reward += 0.5
    elif context.get("fov_visible") is False:
        reward -= 0.1

    return reward
