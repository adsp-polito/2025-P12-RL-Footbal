import numpy as np
from pettingzoo import ParallelEnv
from typing import Dict, Any

from football_tactical_ai.players.playerAttacker import PlayerAttacker
from football_tactical_ai.players.playerDefender import PlayerDefender
from football_tactical_ai.players.playerGoalkeeper import PlayerGoalkeeper
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.configs.multiAgentEnvConfig import get_config
from football_tactical_ai.helpers.helperFunctions import normalize
from football_tactical_ai.env.scenarios.multiAgent.rewardGrids import (
    build_attacker_grid,
    build_defender_grid,
    build_goalkeeper_grid,
)

from football_tactical_ai.env.scenarios.multiAgent.multiAgentReward import get_reward
# physics.py da integrare se userai update_ball, collisioni, ecc.

# This is a multi-agent environment for a football tactical AI scenario
class FootballMultiEnv(ParallelEnv):
    """
    A multi-agent environment for football tactical AI scenarios.
    This environment simulates a simple football game with attackers, defenders, and a goalkeeper.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the multi-agent football environment.
        
        Args:
            config (dict, optional): Custom configuration dictionary. 
                                    If None, defaults will be used from get_config().
        """
        # Load configuration
        self.config = config or get_config()

        # Time and simulation parameters
        self.fps = self.config["fps"]
        self.time_step = self.config["time_step"]
        self.max_steps = self.config["max_steps"]

        # Pitch and dimensions
        self.pitch: Pitch = Pitch()
        self.x_range = self.pitch.width
        self.y_range = self.pitch.height

        # Agents and roles
        self.agents = ["att_0", "def_0", "gk_0"]
        self.possible_agents = self.agents[:]  # PettingZoo requirement

        # Instantiate players (IDs and roles)
        self.players = {
            "att_0": PlayerAttacker(agent_id="att_0", team="A", role="ATT"),
            "def_0": PlayerDefender(agent_id="def_0", team="B", role="DEF"),
            "gk_0": PlayerGoalkeeper(agent_id="gk_0", team="B", role="GK"),
        }

        # Ball and initial ownership
        self.ball = Ball()
        self.episode_step = 0

        # Reward grids for each role
        self.reward_grids = {
            "ATT": build_attacker_grid(self.pitch),
            "DEF": build_defender_grid(self.pitch),
            "GK":  build_goalkeeper_grid(self.pitch),
        }


    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.
        Returns:
            dict: Initial observations for all agents.
        """

        # Set random seed for reproducibility
        super().reset(seed=seed)

        # Reset the environment state
        self.agents = self.possible_agents[:]
        self.episode_step = 0
        self.ball.reset()
        self.ball.set_owner("att_0")  # Initial ball ownership

        # Reset positions
        self.players["att_0"].reset_position(normalize(60, 40))
        self.players["def_0"].reset_position(normalize(110, 40))
        self.players["gk_0"].reset_position(normalize(120, 40))


        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        return observations

    def _build_context(self, agent_id, player, shot_owner, shot_just_started, goal_owner, ball_out_by) -> dict:
        """
        Build the context dictionary for the current action.
        Args:
            agent_id (str): ID of the agent performing the action.
            player (BasePlayer): Player instance for the agent.
            shot_owner (str): ID of the player who attempted the shot.
            shot_just_started (bool): Whether the shot was just initiated.
            goal_owner (str): ID of the player who scored a goal, if any.
            ball_out_by (str): ID of the player who caused the ball to go out, if applicable.
        Returns:
            dict: Contextual information about the action taken.
        """
        
        goal_team = self.players[goal_owner].team if goal_owner else None

        return {
            "goal_scored": goal_owner == agent_id,
            "goal_team": goal_team,
            "ball_out_by": ball_out_by,
            "start_shot_bonus": (agent_id == shot_owner and shot_just_started),
            "possession_lost": self._check_possession_loss(agent_id),

            # default-safe entries for reward logic
            "shot_attempted": False,
            "shot_quality": None,
            "not_owner_shot_attempt": False,
            "invalid_shot_direction": False,
            "shot_alignment": None,
            "fov_visible": None,
            "tackle_success": False,
            "save_success": False,
            "shot_positional_quality": 0.0,
        }



    
    def _check_possession_loss(self, agent_id: str) -> bool:
        """
        Check if the attacker has lost possession to a defender or goalkeeper.
        """
        if not agent_id.startswith("att"):
            return False

        new_owner = self.ball.owner
        if new_owner == agent_id:
            return False

        # Check if the new owner is a defender or goalkeeper
        role = self.players.get(new_owner, None).get_role() if new_owner in self.players else None
        return role in {"DEF", "GK"}
    
    def _is_ball_completely_out(self, ball_x_m, ball_y_m):
        """
        Simple check if ball is outside the real field plus margin, using denormalized coordinates.

        Args:
            ball_x_m (float): Ball's x coordinate in meters.
            ball_y_m (float): Ball's y coordinate in meters.
            pitch: Pitch instance containing field dimensions and constants.

        Returns:
            bool: True if ball is outside field + margin, False otherwise
        """

        margin_m = 1.0  # 1.0 meters margin for out of bounds

        # Check if ball outside real field + margin
        if (ball_x_m < 0 - margin_m or
            ball_x_m > self.pitch.width + margin_m or
            ball_y_m < 0 - margin_m or
            ball_y_m > self.pitch.height + margin_m):
            return True

        return False


    def _is_goal(self, x, y):
        """
        Check if the ball is in the net
        """
        GOAL_MIN_Y = self.pitch.center_y - self.pitch.goal_width / 2
        GOAL_MAX_Y = self.pitch.center_y + self.pitch.goal_width / 2
        return x > self.pitch.width and GOAL_MIN_Y <= y <= GOAL_MAX_Y

    def _get_observation(self, agent_id: str):
        """
        Build the observation for a given agent.
        """
        player = self.players[agent_id]
        px, py = player.get_position()
        bx, by = self.ball.get_position()

        return np.array([px, py, bx, by], dtype=np.float32)