import numpy as np
from typing import Dict
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.players.playerBase import BasePlayer

def update_ball_state(ball: Ball,
                      players: Dict[str, BasePlayer],
                      actions: Dict[str, np.ndarray],
                      time_step: float,
                      dribble_offset: float = 0.01):
    """
    Update the ball's position based on ownership or free motion.

    - If the ball is possessed, simulate dribbling (offset in direction).
    - If the ball is free, apply physics (velocity + friction).
    
    Args:
        ball (Ball): The ball object.
        players (dict): All agents {agent_id: player object}.
        actions (dict): Actions taken by each agent this step.
        time_step (float): Time step duration.
        dribble_offset (float): Distance to place ball in front of owner.
    """

    owner_id = ball.get_owner()
    
    # Case 1: Ball is possessed
    if owner_id in players:
        player = players[owner_id]
        action = actions.get(owner_id, None)

        # Default direction: no movement
        direction = np.array([0.0, 0.0])

        if action is not None and len(action) >= 2:
            direction = np.array(action[:2])  # dx, dy
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = np.array([0.0, 0.0])

        # Place ball in front of player
        new_pos = np.array(player.get_position()) + direction * dribble_offset
        ball.set_position(new_pos)
        ball.set_velocity([0.0, 0.0])
        return

    # Case 2: Free ball physics
    ball.update(time_step)
