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
    Attacker reward function.
    
    Main principles:
    1. Positioning → dense shaping from reward grid + small time penalty
    2. Ball pursuit → reward for moving close to free ball
    3. Possession / Ball-out → penalize losing ball or sending it out
    4. Goals → large reward for scoring; medium for team goals
    5. Passing → rewarded for both passer and receiver
       - Blind pass (outside FOV) is allowed, but gets smaller reward
       - Pass inside FOV is preferred, larger reward
    6. Shooting → stricter rules:
       - Shots outside FOV are invalid and penalized
       - Valid shots inside FOV get rewards based on quality, alignment, and position
    7. FOV visibility → general shaping: being oriented towards play is positive
    """

    reward = 0.0

    # BASE: positioning + small time penalty
    reward += pos_reward
    
    reward -= 0.005  # time malus to avoid stalling

    # BALL CHASING: encourage attacker to chase free ball
    # small reward up to ~10 meters away because the attacker should be proactive only when he is close enough
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Reward decays exponentially with distance
        # At dist=0  -> reward=0.01
        # At dist≈5m -> reward≈0.003
        # At dist≈10m -> reward≈0.0001 (practically zero)
        reward += 0.01 * np.exp(-0.5 * dist_to_ball)

    # POSSESSION / OUT OF PLAY
    if context.get("possession_lost", False):
        reward -= 0.25

    if context.get("ball_out_by") == agent_id:
        reward -= 0.5

    # GOALS
    if context.get("goal_scored", False):
        reward += 8.0  # scorer bonus
    if context.get("goal_team") == player.team:
        reward += 5.0  # shared team bonus

    # PASSING
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.25  # reward for attempting pass

        # consider pass quality if available only one time
        if context.get("pass_quality") is not None:
            reward += 0.5 * context["pass_quality"] # scaled [0, 0.5] (0 if pass is invalid)

    if context.get("invalid_pass_attempt", False):
        reward -= 0.4  # passing without ball or outside FOV

    if context.get("pass_completed", False):
        # Reward passer and receiver symmetrically
        if "pass_to" in context or "pass_from" in context:
            if context.get("invalid_pass_direction", False):
                # Blind pass that worked → smaller bonus
                reward += 1.0
            else:
                # Normal visible pass → larger bonus
                reward += 2.0
        # Extra small team cooperation bonus
        reward += 1.5


    # SHOOTING
    if context.get("start_shot_bonus", False):
        reward += 0.5
        reward += context.get("shot_positional_quality", 0.0)

        # consider shot quality if available only one time
        if context.get("shot_quality") is not None:
            reward += 0.5 * context["shot_quality"] # scaled [0, 0.5]

    if context.get("invalid_shot_attempt", False):
        reward -= 0.3  # shooting without ball or outside FOV
    if context.get("invalid_shot_direction", False):
        reward -= 0.2

    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += alignment ** 2 - 0.25  # shaping from -0.25 to +0.75

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

    return reward
