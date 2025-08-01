import numpy as np
from football_tactical_ai.env.scenarios.singleAgent.base_offensive import BaseOffensiveScenario
from football_tactical_ai.helpers.helperFunctions import normalize


# Coordinate System Design:
#
# This environment works in normalized coordinates within [0, 1] for X and Y.
# Player positions (attacker, defender) and the ball are stored and updated normalized.
# Movements are computed in this normalized space.
#
# Normalization:
# - The physical pitch spans X_MIN to X_MAX and Y_MIN to Y_MAX (in meters).
# - Positions are normalized between 0 and 1 with these limits.
#
# Rewards and rendering:
# - Rewards are computed in meters, denormalizing positions as needed.
# - Rendering denormalizes to draw correctly on the pitch in meters.
#
# Summary:
# 1. Environment stores positions in [0, 1].
# 2. Actions move positions in [0, 1].
# 3. Observations are returned normalized.
# 4. Rendering converts to meters.
# 5. Rewards are computed in meters after denormalizing.

class OffensiveScenarioMoveSingleAgent(BaseOffensiveScenario):
    """
    Single-agent offensive scenario for moving the attacker with continuous control.
    The attacker can move in the X and Y directions, while the defender tries to intercept.
    This scenario is designed to test the attacker's ability to maintain possession of the ball
    while avoiding the defender.
    Based on the BaseOffensiveScenario class, it implements the necessary methods
    to handle the environment's dynamics, rewards, and observations.
    """

    def __init__(self, pitch, max_steps=240, fps=24):

        # Initialize the parent class
        super().__init__(pitch=pitch, max_steps=max_steps, fps=fps)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        The attacker and defender are placed at their starting positions, the ball is centered,
        and the game state is initialized.
        """

        # Reset the random seed if provided
        # By doing this, we use the reset method from the parent class so 
        # the ball and players are initialized correctly
        super().reset(seed=seed)

        return self._get_obs(), {}

    def compute_reward(self):
        """
        Computes the reward for the current step based on the attacker's position.
        Looks up the reward grid, applies a time penalty, and checks for goals or possession loss.
        """
        reward = 0.0  # Initialize reward for this step

         # Apply small time penalty to encourage active movement
        reward -= 0.1

        # Convert attacker's position from normalized [0, 1] to meters
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        y_m = att_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min

        # Get reward based on position from reward grid
        reward +=  self._get_position_reward(x_m, y_m)

        # Check if a goal has been scored
        if self._is_goal(x_m, y_m):
            reward += 5.0
            return reward  

        # Check if possession was lost
        if self._check_possession_loss():
            reward -= 1.0
            return reward

        return reward

    def _check_termination(self) -> bool:
        att_x, att_y = self.attacker.get_position()
        x_m = att_x * (self.pitch.x_max - self.pitch.x_min) + self.pitch.x_min
        y_m = att_y * (self.pitch.y_max - self.pitch.y_min) + self.pitch.y_min
        return self._is_goal(x_m, y_m) or self._check_possession_loss()

    def _get_obs(self):
        """
        Get the current observation: normalized positions of attacker, defender, ball, and possession status.
        """
        # Normalize positions to [0, 1]
        att_x, att_y = self.attacker.get_position()
        def_x, def_y = self.defender.get_position()
        ball_x, ball_y = self.ball.position

        # Possession status: 1 if attacker has possession, 0 otherwise
        possession = 1.0 if self.ball.owner is self.attacker else 0.0

        return np.array([att_x, att_y, def_x, def_y, ball_x, ball_y, possession], dtype=np.float32)

    def close(self):
        """
        Close the environment and release resources.
        """
        # No specific resources to release in this environment
        # but this method is here for compatibility with gym's API
        pass
