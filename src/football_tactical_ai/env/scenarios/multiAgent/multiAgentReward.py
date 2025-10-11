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



def attacker_reward(agent_id, player, pos_reward, ball, context, pass_pending, pitch):
    """
    Attacker reward function (balanced for tactical learning).

    Main design principles:
    1. Positioning → dense shaping from reward grid (only when not receiving a pass)
    2. Time penalty → small negative cost to encourage proactive play
    3. Ball pursuit → small positive incentive for chasing free balls
    4. Possession & ball-out → penalties for losing or sending out the ball
    5. Passing → rewarded for valid passes and successful receptions
    6. Shooting → reward proportional to shot quality and alignment
    7. Goals → large positive reward for scoring or contributing
    """

    if not hasattr(attacker_reward, "consecutive_passes"):
        attacker_reward.consecutive_passes = 0
    if not hasattr(attacker_reward, "prev_x"):
        attacker_reward.prev_x = {}

    reward = 0.0

    # Determine if player is currently the target of a pass
    is_pass_target = (
        pass_pending is not None
        and pass_pending.get("active", False)
        and pass_pending.get("to") == agent_id
    )

    # 1) POSITIONING BASE REWARD
    # Apply only if player is not currently the pass target
    if not is_pass_target:
        reward += pos_reward

        # ADDITIONAL PROGRESSION INCENTIVE
        # Encourage advancing toward the opponent's goal when having the ball
        # (Track movement along X-axis since the grid already encodes good Y behavior)
        if ball is not None and ball.get_owner() == agent_id:
            x_now, _ = denormalize(*player.get_position())
            x_prev = attacker_reward.prev_x.get(agent_id, x_now)
            dx = x_now - x_prev if player.team == "A" else x_prev - x_now

            # Reward positive forward movement, slight penalty for retreating
            reward += 0.05 * np.tanh(dx * 0.4)
            attacker_reward.prev_x[agent_id] = x_now

    # 2) TIME PENALTY
    reward -= 0.005

    # 3) BALL CHASING
    # encourages moving toward free ball
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])
        reward += 0.03 * np.exp(-0.4 * dist_to_ball)  # up to +0.03 when very close

    # 4) POSSESSION AND BALL OUT
    if context.get("possession_lost", False):
        reward -= 0.25
        attacker_reward.consecutive_passes = 0

    if context.get("ball_out_by") == agent_id:
        reward -= 0.4
        attacker_reward.consecutive_passes = 0

    # 5) GOALS
    if context.get("goal_scored", False):
        reward += 7.5
        attacker_reward.consecutive_passes = 0

    if context.get("goal_team") == player.team:
        reward += 2.5
        attacker_reward.consecutive_passes = 0

    # 6) PASSING BEHAVIOR
    # Valid pass attempt
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.25
        if context.get("pass_quality") is not None:
            reward += 0.25 * context["pass_quality"]

    # Invalid pass attempt → light penalty
    if context.get("invalid_pass_attempt", False):
        reward -= 0.2

    # 7) RECEIVER BEHAVIOR
    # Encourage moving towards ball if currently the pass target and pass not yet completed
    if is_pass_target and not context.get("pass_completed", False):
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())

        # Compute distance to ball
        dist = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Track previous distance per agent
        if not hasattr(attacker_reward, "prev_dist"):
            attacker_reward.prev_dist = {}
        prev_dist = attacker_reward.prev_dist.get(agent_id, dist)

        # Distance change (positive if moving closer)
        delta = prev_dist - dist

        # Reward for approaching the ball
        if delta > 0:
            reward += 0.15 * np.tanh(delta)

        # Penalty for moving away from the ball
        elif delta < 0:
            reward -= 0.15 * np.tanh(-delta)

        # Base reward for being close to the ball
        reward += 0.25 * np.exp(-0.4 * dist)

        # Store distance for next step
        attacker_reward.prev_dist[agent_id] = dist


    # 8) PASS COMPLETION (both passer and receiver)
    if context.get("pass_completed", False):

        # PASSER REWARD
        if context.get("pass_from") == agent_id:
            if context.get("invalid_pass_direction", False):
                reward += 0.5  # lower bonus for suboptimal direction
            else:
                reward += 1.0  # normal passer bonus

        # RECEIVER REWARD
        if context.get("pass_to") == agent_id:
            reward += 0.5  # same reward regardless of direction

        # TEAM SYNERGY BONUS
        reward += 0.25

        # LOGARITHMIC DECAY FOR PASS CHAINS (limit overpassing)
        attacker_reward.consecutive_passes += 1
        chain_bonus = 0.5 / (1.0 + np.log1p(attacker_reward.consecutive_passes))
        reward += chain_bonus

        # Apply small decay after the second pass to discourage excessive looping
        if attacker_reward.consecutive_passes > 2:
            reward -= 0.5 * np.tanh(attacker_reward.consecutive_passes / 2)


    # 9) SHOOTING BEHAVIOR
    if context.get("start_shot_bonus", False):
        reward += 0.5  # slightly higher base to promote shooting
        reward += 0.5 * context.get("shot_positional_quality", 0.0)
        if context.get("shot_quality") is not None:
            reward += 0.25 * context["shot_quality"]
        attacker_reward.consecutive_passes = 0

    if context.get("invalid_shot_attempt", False):
        reward -= 0.25
    if context.get("invalid_shot_direction", False):
        reward -= 0.10

    # Alignment between player direction and goal direction
    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += 0.1 * (alignment ** 2 - 0.25)

    # 10) SHOT OPPORTUNITY BONUS
    # Encourage shooting when entering a realistic shooting zone,
    # but avoid excessive reward just for being close to goal.

    x_p, _ = denormalize(*player.get_position())
    goal_x = pitch.width if player.team == "A" else 0.0
    dist_goal = abs(goal_x - x_p)

    # Moderate logistic bonus: small encouragement near the box (~20m)
    shot_zone_bonus = 0.01 / (1.0 + np.exp(0.3 * (dist_goal - 20)))

    if ball is not None and ball.get_owner() == agent_id:
        # Add small positional encouragement
        reward += shot_zone_bonus

        # If close enough (inside penalty area) and not shooting → light penalty
        if dist_goal < 18 and not context.get("start_shot_bonus", False):
            reward -= 0.05  # slight impatience
        if dist_goal < 11 and not context.get("start_shot_bonus", False):
            reward -= 0.1  # stronger impatience very close to goal

    # 11) DEFENSIVE POSSESSION PENALTY
    if ball is not None and ball.get_owner() is not None:
        owner_id = ball.get_owner()
        if owner_id.startswith("def") or owner_id.startswith("gk"):
            reward -= 0.1  # attackers lose reward if defenders have the ball


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
    reward += 0.005                 # small time survival bonus

    # BALL CHASING
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
    reward += 0.005               # small time survival bonus

    # SAVES
    if context.get("blocked", False):
        reward += 2.5            # successful save
    if context.get("deflected", False):
        reward += 1.25            # save via deflection
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
        reward += 0.5           # putting ball out = acceptable

    # GOALS CONCEDED
    if context.get("goal_scored", False):
        if context.get("goal_team") != player.team:
            reward -= 8.0        # conceding goal

    return reward
