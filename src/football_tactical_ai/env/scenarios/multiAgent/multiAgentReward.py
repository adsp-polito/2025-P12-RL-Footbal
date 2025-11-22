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
    Attacker reward function (cleaned and rebalanced).

    Logic:
    - Dense positional shaping from reward grid.
    - Incentivize forward movement and proactive play.
    - Reward smart passing, discourage endless or lateral passes.
    - Reward shooting and goal scoring, penalize bad shots or passive possession.
    - Reset pass chains after shots or possession loss.
    """

    if not hasattr(attacker_reward, "consecutive_passes"):
        attacker_reward.consecutive_passes = 0
    if not hasattr(attacker_reward, "prev_x"):
        attacker_reward.prev_x = {}
    if not hasattr(attacker_reward, "shot_started"):
        attacker_reward.shot_started = False

    reward = 0.0

    # Check if the player is waiting for a pass (receiver locked)
    is_pass_target = (
        pass_pending is not None
        and pass_pending.get("active", False)
        and pass_pending.get("to") == agent_id
    )

    # 1. POSITIONING AND MOVEMENT
    if not is_pass_target:
        # Base positional reward from grid
        reward += pos_reward

    # Time penalty to encourage quicker actions
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
        attacker_reward.consecutive_passes = 0

    if context.get("ball_out_by") == agent_id:
        reward -= 2.5
        attacker_reward.consecutive_passes = 0

    # 4. PASSING BEHAVIOR

    # Reward valid pass attempts
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.5  # small bonus for starting a pass
        if context.get("pass_quality") is not None:
            reward += context["pass_quality"]

    elif context.get("pass_completed", False):
        attacker_reward.consecutive_passes += 1

        # Reward passer
        if context.get("pass_from") == agent_id:
            if context.get("invalid_pass_direction", False):
                reward += 0.25  # less reward for wrong direction
            else:
                reward += 0.5  # normal passer reward

        # Reward receiver
        if context.get("pass_to") == agent_id:
            reward += 0.25

        # Small synergy bonus for teamwork
        reward += 0.10

        # Diminishing reward on consecutive passes and penalize excessive ball circulation
        # The agent receives a small positive reward for short, effective pass sequences 
        # (1–2 consecutive successful passes) to encourage coordinated buildup play
        #
        # However, as the number of consecutive passes increases without a shot attempt,
        # the total effect gradually decreases and eventually becomes negative
        #
        # Formula:
        #   chain_effect = (positive_decay_term) - (negative_penalty_term)
        #
        #   The two parts combined ensure that:
        #     - Pass 1–2 → slightly positive reward
        #     - Pass 3 → roughly neutral (≈ 0)
        #     - Pass ≥4 → negative, discouraging excessive circulation
        # This way, the agent learns to prefer fast, purposeful combinations
        # that lead to shooting opportunities instead of infinite back-and-forth passes

        if not attacker_reward.shot_started:
            chain_effect = (
                0.5 / (1.0 + np.log1p(attacker_reward.consecutive_passes))
                - 0.1 * max(0, attacker_reward.consecutive_passes - 2)
            )
            reward += chain_effect

    # Penalize failed or invalid pass attempts
    if context.get("invalid_pass_attempt", False):
        reward -= 0.25

    # Reset pass chain if a shot starts
    if context.get("start_shot_bonus", False):
        attacker_reward.consecutive_passes = 0
        attacker_reward.shot_started = True

    # 5. SHOOTING BEHAVIOR
    if context.get("start_shot_bonus", False) and not context.get("invalid_shot_attempt", False):
        reward += 0.25  # base shooting reward
        reward += 0.25 * context.get("shot_positional_quality", 0.0)
        if context.get("shot_quality") is not None:
            reward += 0.25 * context["shot_quality"]

    # Penalize bad or misaligned shots
    if context.get("invalid_shot_attempt", False):
        reward -= 0.25
    if context.get("invalid_shot_direction", False):
        reward -= 0.25

    # Reward alignment with goal (squared for smoother scaling)
    alignment = context.get("shot_alignment")
    if alignment is not None:  
        reward += 0.1 * (alignment ** 2 - 0.25) # from -0.025 to +0.075

    # 6. SHOT OPPORTUNITY BONUS
    x_p, _ = denormalize(*player.get_position())
    goal_x = pitch.width if player.team == "A" else 0.0
    dist_goal = abs(goal_x - x_p)

    # Small bonus for entering a realistic shooting area
    shot_zone_bonus = 0.01 / (1.0 + np.exp(0.3 * (dist_goal - 20))) # from ~0.01 at 10m to ~0.001 at 30m

    if ball is not None and ball.get_owner() == agent_id:
        reward += shot_zone_bonus

        # Impatience penalty near the goal if not shooting
        if dist_goal < 18 and not context.get("start_shot_bonus", False):
            reward -= 0.25 * (1 + attacker_reward.consecutive_passes / 2)
        if dist_goal < 11 and not context.get("start_shot_bonus", False):
            reward -= 0.5 * (1 + attacker_reward.consecutive_passes / 2)

    # 7. GOALS
    if context.get("goal_scored", False):
        reward += 7.5
        attacker_reward.consecutive_passes = 0
    elif context.get("goal_team") == player.team:
        reward += 2.5
        attacker_reward.consecutive_passes = 0

    # 8. DEFENSIVE POSSESSION PENALTY
    if ball is not None and ball.get_owner() is not None:
        owner_id = ball.get_owner()
        if owner_id.startswith("def") or owner_id.startswith("gk"):
            reward -= 0.1

    return reward

def defender_reward(agent_id, player, pos_reward, ball, context, pitch):
    """
    Defender reward function (refined and balanced).

    Core principles:
    - Encourage compact positioning and staying near defensive line.
    - Reward key defensive actions (tackle, interception, clearance).
    - Penalize conceding goals or losing structure.
    - Small incentives for chasing free balls and delaying play.
    """

    reward = 0.0

    # 1. BASE POSITIONING AND SURVIVAL
    reward += pos_reward              # dense positional shaping

    # 2. BALL CHASING (incentive to contest loose balls)
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist = np.linalg.norm([x_b - x_p, y_b - y_p])
        reward += max(0.0, 0.5 - 0.005 * dist)  # up to +0.5 when close

    # 3. DEFENSIVE ACTIONS
    if context.get("tackle_success", False):
        # Stronger reward when close to goal (critical zone)
        x_p, _ = denormalize(*player.get_position())
        goal_x = 0.0 if player.team == "A" else pitch.width
        dist_goal = abs(goal_x - x_p)
        zone_factor = np.exp(-0.1 * dist_goal)  # higher near goal
        reward += 0.8 + 0.3 * zone_factor

    if context.get("interception_success", False):
        reward += 0.7

    # 4. CLEARANCE AND BALL OUT
    # Forcing the ball out of play is generally positive if it prevents danger
    if context.get("ball_out_by") == agent_id:
        reward += 0.3
        if not context.get("goal_scored", False):  # safe outcome
            reward += 0.2

    # 5. FAILED OR USELESS ACTIONS
    if context.get("fake_tackle", False):
        reward -= 0.25
    if context.get("invalid_shot_attempt", False):
        reward -= 0.1
    if context.get("invalid_shot_direction", False):
        reward -= 0.1

    # 6. GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 8.0  # conceding is the worst outcome

    # 7. POSITIONAL DISCIPLINE
    # Slight penalty if too far from defensive line
    x_p, _ = denormalize(*player.get_position())
    goal_x = 0.0 if player.team == "A" else pitch.width
    dist_from_goal = abs(goal_x - x_p)
    if dist_from_goal > 25:  # far outside defensive zone
        reward -= 0.1 * (dist_from_goal - 25) / 10.0  # gradual penalty

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

