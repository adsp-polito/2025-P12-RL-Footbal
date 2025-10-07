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
    Attacker reward function (balanced version with pass chain decay).

    Main principles:
    1. Positioning → dense shaping from reward grid + small time penalty
    2. Ball pursuit → small reward for chasing a free ball (close distance)
    3. Possession / Ball-out → penalties for losing ball or sending it out
    4. Goals → large reward for scoring; medium for team goals
    5. Passing → rewarded for both passer and receiver
       - Blind pass (outside FOV) gets smaller reward
       - Pass inside FOV is preferred, larger reward
       - Consecutive passes decay progressively to avoid infinite loops
       - Receiver is rewarded for stabilizing and waiting for the ball
    6. Shooting → stricter rules:
       - Shots outside FOV are invalid and penalized
       - Valid shots inside FOV get rewards based on quality, alignment, and position
    """

    # GLOBAL COUNTER: tracks consecutive completed passes across all agents
    # Resets on shot, goal, ball out, or loss of possession
    if not hasattr(attacker_reward, "consecutive_passes"):
        attacker_reward.consecutive_passes = 0

    reward = 0.0

    # 1) POSITIONING BASE REWARD
    reward += pos_reward       # dense spatial shaping
    #reward -= 0.005            # small time penalty to prevent stalling

    # 2) BALL CHASING: encourage approaching the free ball
    if ball is not None and ball.get_owner() is None:
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist_to_ball = np.linalg.norm([x_b - x_p, y_b - y_p])
        # Reward decays exponentially with distance
        # At dist=0  -> reward=0.0015
        # At dist≈5m -> reward≈0.0005
        # At dist≈10m -> reward≈0.0001 (almost zero)
        reward += 0.0015 * np.exp(-0.4 * dist_to_ball)

    # 3) POSSESSION AND BALL OUT
    if context.get("possession_lost", False):
        reward -= 0.25
        attacker_reward.consecutive_passes = 0  # reset chain
    if context.get("ball_out_by") == agent_id:
        reward -= 0.5
        attacker_reward.consecutive_passes = 0  # reset chain

    # 4) GOALS
    if context.get("goal_scored", False):
        reward += 8.0                 # scorer bonus
        attacker_reward.consecutive_passes = 0
    if context.get("goal_team") == player.team:
        reward += 5.0                 # shared team goal bonus
        attacker_reward.consecutive_passes = 0

    # 5) PASSING BEHAVIOR
    # Reward logic:
    # - Start of pass → small encouragement to attempt
    # - Invalid pass → penalty
    # - Pass in flight → strong shaping for receiver (move toward the ball)
    # - Completed pass → moderate bonus for passer, strong for receiver
    # - Receiver learns to stabilize when targeted by a pass

    # PASS INITIATION (valid attempt)
    if context.get("start_pass_bonus", False) and not context.get("invalid_pass_attempt", False):
        reward += 0.4
        if context.get("pass_quality") is not None:
            reward += 0.4 * context["pass_quality"]

    # INVALID PASS ATTEMPT (no ball / outside FOV)
    if context.get("invalid_pass_attempt", False):
        reward -= 0.3

    # PASS IN FLIGHT (receiver shaping)
    if context.get("pass_to") == agent_id and not context.get("pass_completed", False):
        # Receiver should move toward the ball, not away from it
        x_p, y_p = denormalize(*player.get_position())
        x_b, y_b = denormalize(*ball.get_position())
        dist = np.linalg.norm([x_b - x_p, y_b - y_p])

        # Reward approaching the ball (strong shaping)
        if dist < 12.0:
            reward += 0.6 * np.exp(-0.12 * dist)   # up to +0.6 when very close
        else:
            reward -= 0.15                         # mild penalty if far

        # Penalize moving away from ball trajectory
        vx, vy = ball.get_velocity()
        if np.linalg.norm([vx, vy]) > 1e-6:
            ball_dir = np.array([vx, vy]) / np.linalg.norm([vx, vy])
            player_dir = np.array([x_p - x_b, y_p - y_b])
            player_dir /= (np.linalg.norm(player_dir) + 1e-6)
            alignment = np.dot(ball_dir, player_dir)
            if alignment < 0:
                reward -= 0.4 * abs(alignment)  # stronger penalty for running away

        # BONUS FOR STABILIZING (waiting for the ball)
        # Encourage receiver to slow down if the ball is coming
        speed = np.linalg.norm(player.velocity)
        ball_speed = np.linalg.norm(ball.get_velocity())
        if ball_speed > 1e-6:  # only if ball is moving
            reward += 0.1 * np.exp(-0.4 * speed / (ball_speed + 1e-6))
            # Small penalty if moving too much compared to ball
            if speed > ball_speed:
                reward -= 0.05

        # Small penalty if increasing distance from the ball (moving away)
        ball_to_player = np.array([x_p - x_b, y_p - y_b])
        dist_change = np.dot(ball_to_player, player.velocity)
        if dist_change > 0:  # moving away
            reward -= 0.05

    # PASS COMPLETION
    if context.get("pass_completed", False):
        # Passer
        if context.get("pass_from") == agent_id:
            if context.get("invalid_pass_direction", False):
                reward += 0.8      # blind pass succeeded
            else:
                reward += 1.0      # good visible pass
            reward += 0.2          # small cooperative team bonus

        # Receiver
        if context.get("pass_to") == agent_id:
            reward += 1.5          # major bonus for successful reception

        # Global light team cooperation bonus
        reward += 0.25

        # DECAY LOGIC: reduce reward for consecutive passing loops
        attacker_reward.consecutive_passes += 1
        decay = max(0.3, 1.0 - 0.1 * attacker_reward.consecutive_passes)
        reward *= decay

    # 6) SHOOTING BEHAVIOR
    # Reward for shooting depends on quality and position
    # Invalid shots (outside FOV / without ball) are penalized
    if context.get("start_shot_bonus", False):
        reward += 0.5
        reward += context.get("shot_positional_quality", 0.0)
        if context.get("shot_quality") is not None:
            reward += 0.4 * context["shot_quality"]
        attacker_reward.consecutive_passes = 0  # reset chain after shot

    if context.get("invalid_shot_attempt", False):
        reward -= 0.3
    if context.get("invalid_shot_direction", False):
        reward -= 0.2

    # Alignment shaping (penalty for misalignment, reward for accurate aiming)
    alignment = context.get("shot_alignment")
    if alignment is not None:
        reward += alignment ** 2 - 0.25  # range [-0.25, +0.75]

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
