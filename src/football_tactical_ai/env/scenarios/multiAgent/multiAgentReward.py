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
               context: Dict = None,
               pass_pending: Dict = None) -> float:
    """
    General reward dispatcher by role.

    Args:
        player (BasePlayer): Player instance (att, def, gk)
        ball (Ball): Ball instance
        pitch (Pitch): Pitch instance
        reward_grid (np.ndarray): Grid with position-based reward for this role
        context (Dict): Action result context (e.g. {'shot': True}, {'tackle': success})
        pass_pending (Dict): Info about ongoing pass, if any
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
        return attacker_reward(agent_id, player, pos_reward, ball, context, pass_pending)
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





def attacker_reward(agent_id, player, pos_reward, ball, context, pass_pending):
    """
    Attacker reward function (enhanced for realistic pass reception).

    Main principles:
    1. Positioning → dense shaping from reward grid + small time penalty
    2. Ball pursuit → small reward for chasing a free ball
    3. Possession / Ball-out → penalties for losing or sending out the ball
    4. Goals → large positive reward for scoring
    5. Passing → rewarded for both passer and receiver
       - Receiver is now strongly encouraged to stay stable and wait for the ball
    6. Shooting → reward based on alignment and quality
    """

    if not hasattr(attacker_reward, "consecutive_passes"):
        attacker_reward.consecutive_passes = 0

    reward = 0.0

    # 1) POSITIONING BASE REWARD
    reward += pos_reward

    # 2) BALL CHASING: small positive incentive
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])
        reward += 0.0015 * np.exp(-0.4 * dist_to_ball)

    # 3) POSSESSION AND BALL OUT
    if context.get("possession_lost", False):
        reward -= 0.25
        attacker_reward.consecutive_passes = 0
    if context.get("ball_out_by") == agent_id:
        reward -= 0.5
        attacker_reward.consecutive_passes = 0

    # 4) GOALS
    if context.get("goal_scored", False):
        reward += 8.0
        attacker_reward.consecutive_passes = 0
    if context.get("goal_team") == player.team:
        reward += 5.0
        attacker_reward.consecutive_passes = 0

    # 5) PASSING BEHAVIOR
    # PASS INITIATION
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.4
        if context.get("pass_quality") is not None:
            reward += 0.4 * context["pass_quality"]

    if context.get("invalid_pass_attempt", False):
        reward -= 0.3

    # PASS IN FLIGHT (no velocity-based shaping)
    is_pass_target = (
        pass_pending is not None and
        pass_pending.get("active", False) and
        pass_pending.get("to") == agent_id
    )

    if is_pass_target and not context.get("pass_completed", False):
        # Get positions (in meters)
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Reward proximity to ball (stronger when close)
        reward += 0.6 * np.exp(-0.25 * dist)  # up to +0.6 when < 2m

        # Light penalty when too far (>10m)
        if dist > 10:
            reward -= 0.15

        # Light positional stability bonus (close and consistent)
        if dist < 5:
            reward += 0.2 * np.exp(-0.15 * dist)

        # Penalize moving away from the ball trajectory
        vx, vy = ball.get_velocity()
        if np.linalg.norm([vx, vy]) > 1e-6:
            ball_dir = np.array([vx, vy]) / np.linalg.norm([vx, vy])
            player_dir = np.array([x_p - x_b, y_p - y_b])
            player_dir /= (np.linalg.norm(player_dir) + 1e-6)
            alignment = np.dot(ball_dir, player_dir)
            if alignment < 0:
                reward -= 0.2 * abs(alignment)  # smaller penalty than before

        # Final patience shaping (bonus for staying in “good zone”)
        if dist < 3:
            reward += 0.3


    # PASS COMPLETION
    if context.get("pass_completed", False):

        if context.get("pass_from") == agent_id:
            if context.get("invalid_pass_direction", False):
                reward += 0.8
            else:
                reward += 1.0
            reward += 0.2  # teamwork

        if context.get("pass_to") == agent_id:
            reward += 1.8  # slightly higher bonus for successful reception

        # Team bonus
        reward += 0.25

        # Decay for long passing chains
        attacker_reward.consecutive_passes += 1
        decay = max(0.4, 1.0 - 0.08 * attacker_reward.consecutive_passes)
        reward *= decay

    # 6) SHOOTING BEHAVIOR
    if context.get("start_shot_bonus", False):
        reward += 0.5
        reward += context.get("shot_positional_quality", 0.0)
        if context.get("shot_quality") is not None:
            reward += 0.4 * context["shot_quality"]
        attacker_reward.consecutive_passes = 0

    if context.get("invalid_shot_attempt", False):
        reward -= 0.3
    if context.get("invalid_shot_direction", False):
        reward -= 0.2

    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += alignment ** 2 - 0.25

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
