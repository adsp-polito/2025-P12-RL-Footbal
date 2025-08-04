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

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute a simulation step for each agent in the environment.

        Args:
            actions (dict): Mapping from agent ID to their continuous action vector.

        Returns:
            tuple:
                - observations: New observations after the step.
                - rewards: Reward per agent.
                - terminations: Terminal flag per agent (True if goal).
                - truncations: Truncation flag per agent (True if max steps reached).
                - infos: Additional info per agent (optional).
        """

        # Step counter
        self.episode_step += 1

        # Dictionary to store results from agent actions (used for contextual reward logic)
        self.last_action_results = {}

        # Apply each agent's action and store contextual result (e.g., shot, direction, etc.)
        for agent_id, action in actions.items():
            player = self.players[agent_id]
            result = player.execute_action(
                action,
                time_step=self.time_step,
                x_range=self.x_range,
                y_range=self.y_range
            )
            self.last_action_results[agent_id] = result or {}

        # Physics update (e.g., ball movement)
        self.ball.update(self.time_step)

        # Initialize output containers
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Compute reward for each agent using role-specific grid and contextual action result
        for agent_id in self.agents:
            player = self.players[agent_id]
            role = player.get_role()
            reward_grid = self.reward_grids.get(role)
            context = self.last_action_results.get(agent_id, {})

            rewards[agent_id] = get_reward(
                agent_id=agent_id,
                player=player,
                ball=self.ball,
                pitch=self.pitch,
                reward_grid=reward_grid,
                context=context,
            )

        # Check global termination conditions
        goal_scored = self._check_goal()
        timeout = self.episode_step >= self.max_steps

        # Apply termination and truncation flags
        for agent_id in self.agents:
            terminations[agent_id] = goal_scored
            truncations[agent_id] = timeout
            infos[agent_id] = {}

        # Get new observations
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        return observations, rewards, terminations, truncations, infos



    def _check_goal(self):
        # Example: simple goal condition
        ball_x, ball_y = self.ball.get_position()
        return ball_x >= 1.0 and 0.4 <= ball_y <= 0.6  # in front of goal center

    def _get_observation(self, agent_id: str):
        """
        Build the observation for a given agent.
        """
        player = self.players[agent_id]
        px, py = player.get_position()
        bx, by = self.ball.get_position()

        

        return np.array([px, py, bx, by], dtype=np.float32)