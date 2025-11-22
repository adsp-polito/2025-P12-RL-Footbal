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
        return attacker_reward(agent_id, player, pos_reward, ball, context, pass_pending, pitch)
    elif role in defender_roles:
        return defender_reward(agent_id, player, pos_reward, ball, context, pitch)
    elif role in goalkeeper_roles:
        return goalkeeper_reward(agent_id, player, pos_reward, context, pitch)
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





def attacker_reward(agent_id, player, pos_reward, ball, context, pass_pending, pitch):
    """
    Attacker reward function (cleaned, fixed, and rebalanced).

    Key principles:
    - Dense positional shaping from reward grid.
    - Encourage forward progression and goal-oriented play.
    - Reward efficient short passing, discourage long pass chains.
    - Reward shooting attempts and goal scoring.
    - Penalize possession loss, bad passes, bad shots.
    - Maintain per-agent internal state (pass chains, first-shot marker).
    """

    # -------------------------------------------------------------
    # 0. INTERNAL STATE (PER-AGENT)
    # -------------------------------------------------------------
    if not hasattr(attacker_reward, "consecutive_passes"):
        attacker_reward.consecutive_passes = {}
    if not hasattr(attacker_reward, "shot_started"):
        attacker_reward.shot_started = {}
    if not hasattr(attacker_reward, "prev_ball_x"):
        attacker_reward.prev_ball_x = {}

    # Initialize state for this attacker
    if agent_id not in attacker_reward.consecutive_passes:
        attacker_reward.consecutive_passes[agent_id] = 0
    if agent_id not in attacker_reward.shot_started:
        attacker_reward.shot_started[agent_id] = False
    if agent_id not in attacker_reward.prev_ball_x:
        bx_m, _ = denormalize(*ball.get_position())
        attacker_reward.prev_ball_x[agent_id] = bx_m

    reward = 0.0

    # 1. POSITIONAL SHAPING AND MOVEMENT
    # If this attacker is the intended pass receiver → must stay still
    is_pass_target = (
        pass_pending is not None
        and pass_pending.get("active", False)
        and pass_pending.get("to") == agent_id
    )

    if not is_pass_target:
        reward += pos_reward  # dense positional shaping

    # Small time penalty to discourage stalling
    reward -= 0.02

    # 2. BALL CHASING (when ball is free)
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist = np.linalg.norm([x_b - x_p, y_b - y_p])
        reward += 0.05 * np.exp(-0.4 * dist)

    # 3. POSSESSION EVENTS
    if context.get("possession_lost", False):
        reward -= 2.5
        attacker_reward.consecutive_passes[agent_id] = 0

    if context.get("ball_out_by") == agent_id:
        reward -= 2.5
        attacker_reward.consecutive_passes[agent_id] = 0

    # 4. PASSING BEHAVIOR 

    # (A) Valid pass attempt starts
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.1
        if context.get("pass_quality") is not None:
            reward += 0.2 * context["pass_quality"]

    # (B) Successful pass completion
    elif context.get("pass_completed", False):
        attacker_reward.consecutive_passes[agent_id] += 1
        c = attacker_reward.consecutive_passes[agent_id]

        # passer reward
        if context.get("pass_from") == agent_id:
            reward += 0.15

        # receiver reward
        if context.get("pass_to") == agent_id:
            reward += 0.10

        # minimal teamwork bonus
        reward += 0.05

        # Chain shaping: reward only the first 2 passes
        if not attacker_reward.shot_started[agent_id]:
            if c <= 2:
                reward += 0.05
            else:
                reward -= 0.10 * (c - 2)

    # (C) Bad pass attempts
    if context.get("invalid_pass_attempt", False):
        reward -= 0.25

    if context.get("pass_failed", False):
        reward -= 0.25

    # 5. BALL PROGRESSION BONUS (TOWARD GOAL)
    bx_norm, _ = ball.get_position()
    bx_m, _ = denormalize(bx_norm, 0.0)
    prev_bx = attacker_reward.prev_ball_x[agent_id]

    # +1 if team A (attacking right), -1 if team B (attacking left)
    dir_sign = 1.0 if player.team == "A" else -1.0

    dx = (bx_m - prev_bx) * dir_sign
    attacker_reward.prev_ball_x[agent_id] = bx_m

    if dx > 0:
        reward += 0.02 * dx

    # 6. SHOOTING BEHAVIOR
    if context.get("start_shot_bonus", False) and not context.get("invalid_shot_attempt", False):
        reward += 0.25
        reward += 0.25 * context.get("shot_positional_quality", 0.0)
        if context.get("shot_quality") is not None:
            reward += 0.25 * context["shot_quality"]

        # reset pass chain for this agent → shot started
        attacker_reward.consecutive_passes[agent_id] = 0
        attacker_reward.shot_started[agent_id] = True

    # Bad shots
    if context.get("invalid_shot_attempt", False):
        reward -= 0.25
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Reward angular alignment (smooth scaling)
    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += 0.1 * (alignment**2 - 0.25)

    # 7. SHOT OPPORTUNITY BONUS (proximity to goal)
    x_p, _ = denormalize(*player.get_position())
    goal_x = pitch.width if player.team == "A" else 0.0
    dist_goal = abs(goal_x - x_p)

    shot_zone_bonus = 0.01 / (1.0 + np.exp(0.3 * (dist_goal - 20)))

    if ball.get_owner() == agent_id:
        reward += shot_zone_bonus

        # Impatience penalty near goal if not shooting
        if dist_goal < 18 and not context.get("start_shot_bonus", False):
            reward -= 0.25 * (1 + attacker_reward.consecutive_passes[agent_id] / 2)
        if dist_goal < 11 and not context.get("start_shot_bonus", False):
            reward -= 0.5 * (1 + attacker_reward.consecutive_passes[agent_id] / 2)

    # 8. GOALS
    if context.get("goal_scored", False):
        reward += 10.0
        attacker_reward.consecutive_passes[agent_id] = 0

    elif context.get("goal_team") == player.team:
        reward += 3.0
        attacker_reward.consecutive_passes[agent_id] = 0

    # 9. DEFENSIVE POSSESSION PENALTY
    if ball.get_owner() is not None:
        owner_id = ball.get_owner()
        if owner_id.startswith("def") or owner_id.startswith("gk"):
            reward -= 0.01

    return reward




def defender_reward(agent_id, player, pos_reward, ball, context, pitch):
    """
    Defender reward function (refined, balanced, and shaped for learnability).

    Principles:
    - Encourage compact defensive positioning.
    - Reward blocking, tackling, interceptions.
    - Reward being between the ball and the goal (key defensive geometry).
    - Penalize going out of the pitch and conceding goals.
    - Small positive incentives for ball pressure.
    """

    reward = 0.0

    # 0. TIME PENALTY
    reward -= 0.01

    # 1. BASE POSITIONAL REWARD
    reward += pos_reward

    # Convert normalized player position to meters
    x_d, y_d = denormalize(*player.get_position())

    # 2. BALL CHASING (free ball pressure)
    if ball is not None and ball.get_owner() is None:
        x_b, y_b = denormalize(*ball.get_position())
        dist = np.linalg.norm([x_b - x_d, y_b - y_d])
        reward += max(0.0, 0.05 - 0.005 * dist)

    # 3. DEFENSIVE ACTIONS (tackles, interceptions)
    if context.get("tackle_success", False):
        # Stronger reward when near own goal
        goal_x = 0.0 if player.team == "A" else pitch.width
        dist_goal = abs(goal_x - x_d)
        zone_factor = np.exp(-0.1 * dist_goal)
        reward += 0.8 + 0.3 * zone_factor

    if context.get("interception_success", False):
        reward += 0.7

    # 4. CLEARANCES / BALL OUT
    if context.get("ball_out_by") == agent_id:
        reward += 0.3
        if not context.get("goal_scored", False):
            reward += 3.0  # safe defensive outcome

    # 5. BAD / USELESS ACTIONS
    if context.get("fake_tackle", False):
        reward -= 0.05
    if context.get("invalid_shot_attempt", False):
        reward -= 0.05
    if context.get("invalid_shot_direction", False):
        reward -= 0.05

    # 6. BALL–GOAL LINE SHAPING
    # Encourages the defender to stand between the ball and the goal,
    # even when no tackle/interception happens.
    if ball is not None:
        x_b, y_b = denormalize(*ball.get_position())

        # own goal location
        if player.team == "A":
            goal_x, goal_y = 0.0, pitch.center_y
        else:
            goal_x, goal_y = pitch.width, pitch.center_y

        # vectors goal→ball and goal→defender
        v_gb = np.array([x_b - goal_x, y_b - goal_y])
        v_gd = np.array([x_d - goal_x, y_d - goal_y])

        dist_gb = np.linalg.norm(v_gb) + 1e-6

        # projection on the goal→ball line (0 = near goal, 1 = near ball)
        proj = np.dot(v_gb, v_gd) / (dist_gb**2)

        # distance from the defensive line path (cross product magnitude)
        cross = (
            abs(v_gb[0] * v_gd[1] - v_gb[1] * v_gd[0]) / dist_gb
        )

        # If defender is between goal and ball → reward proximity to line
        if 0.0 <= proj <= 1.0:
            line_bonus = 0.05 * np.exp(-0.1 * cross)
            reward += line_bonus

    # 7. GOALS CONCEDED
    if context.get("goal_scored", False) and context.get("goal_team") != player.team:
        reward -= 7.5

    return reward







def goalkeeper_reward(agent_id, player, pos_reward, context, pitch):
    """
    Goalkeeper reward function (refined and balanced).

    Design principles:
    - Encourage correct positioning and defensive stability.
    - Reward effective saves, deflections, and safe clearances.
    - Penalize unnecessary dives or poor positioning.
    - Strong penalty for conceding goals.
    """

    reward = 0.0

    # 1. BASE POSITIONING AND TIME SURVIVAL
    reward += pos_reward          # spatial shaping (staying in goal area)
    reward += 0.005               # small step survival bonus

    # 2. DEFENSIVE ACTIONS
    if context.get("blocked", False):
        reward += 2.5             # direct save — best outcome
    elif context.get("deflected", False):
        reward += 1.25            # deflection still good
    elif context.get("dive_score") is not None and not context.get("wasted_dive", False):
        # Reward proportional to dive effectiveness
        reward +=  0.5 * context["dive_score"]

    # 3. CLEARANCES AND BALL OUT
    # Good if prevents danger (even if it concedes a corner)
    if context.get("ball_out_by", None) is not None:
        reward += 0.25
        if context.get("ball_out_by") == agent_id:
            reward += 0.5         # bonus for forcing ball out
        if not context.get("goal_scored", False):
            reward += 2.5         # safe clearance bonus

    # 4. FAILED OR UNNECESSARY ACTIONS
    if context.get("wasted_dive", False):
        reward -= 0.2             # penalize bad anticipation
    if context.get("invalid_shot_attempt", False):
        reward -= 0.1
    if context.get("invalid_shot_direction", False):
        reward -= 0.1

    # 6. GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 8.0         # strongest penalty

    return reward

