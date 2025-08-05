import numpy as np
from pettingzoo import ParallelEnv
from typing import Dict, Any
from gymnasium import spaces

from football_tactical_ai.players.playerAttacker import PlayerAttacker
from football_tactical_ai.players.playerDefender import PlayerDefender
from football_tactical_ai.players.playerGoalkeeper import PlayerGoalkeeper
from football_tactical_ai.env.scenarios.multiAgent.physics import update_ball_state
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


"""
Action space for each player:

- Attacker (ATT): [dx, dy, shoot_flag, power, dir_x, dir_y]
- Defender (DEF):  [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y]
- Goalkeeper (GK):    [dx, dy, dive_left, dive_right, shoot_flag, power, dir_x, dir_y]

All values are normalized:
- dx, dy ∈ [-1, 1]
- power ∈ [0, 1]
- flag > 0.5 → activate action
"""


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

        # Action spaces per agent
        self.action_spaces = {
            "att_0": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            "def_0": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            "gk_0":  spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        }

        # Observation spaces per agent
        self.observation_spaces = {
            agent_id: spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent_id in self.agents
        }

        # Roles mapping
        self.roles = {aid: self.players[aid].get_role() for aid in self.agents}

        # Ball and initial ownership
        self.ball = Ball()
        self.episode_step = 0

        # Reward grids for each role
        self.reward_grids = {
            "ATT": build_attacker_grid(self.pitch),
            "DEF": build_defender_grid(self.pitch),
            "GK":  build_goalkeeper_grid(self.pitch),
        }

    # PettingZoo interface methods (required for multi-agent environments)
    @property
    def observation_space(self):
        return self.observation_spaces

    @property
    def action_space(self):
        return self.action_spaces

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

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Advance the environment by one timestep using the actions of all agents.

        Args:
            actions (dict): Dictionary of actions for each agent.

        Returns:
            Observations, rewards, terminations, truncations, infos:
                - observations (dict): Observations for each agent.
                - rewards (dict): Rewards for each agent.
                - terminations (dict): Termination flags for each agent.
                - truncations (dict): Truncation flags for each agent.
                - infos (dict): Additional information for each agent.
        """

        # Step 0: Action validation
        if not isinstance(actions, dict):
            raise ValueError("Actions must be a dictionary mapping agent IDs to actions.")
        if not all(agent in actions for agent in self.agents):
            raise ValueError("All agents must provide an action.")

        # Increment episode step
        self.episode_step += 1

        # Initialize containers for observations, rewards, terminations, truncations, and infos
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Temporary context for actions
        temp_context = {}
        shot_owner = None
        shot_just_started = False

        # Step 1: Action execution + context saving
        for agent_id, action in actions.items():
            player = self.players[agent_id]
            context = player.execute_action(
                action=action,
                time_step=self.time_step,
                x_range=self.x_range,
                y_range=self.y_range,
                ball=self.ball
            )

            # Shot attempted by this agent
            if context.get("shot_attempted", False):
                shot_owner = agent_id
                shot_just_started = True

            temp_context[agent_id] = context

        # Step 2: Update ball possession if tackle/save successful
        for agent_id, context in temp_context.items():
            if context.get("tackle_success", False):
                self.ball.set_owner(agent_id)
            elif context.get("save_success", False) and context.get("new_owner") == agent_id:
                self.ball.set_owner(agent_id)

        # Step 3: Ball physics
        update_ball_state(self.ball, self.players, actions, self.time_step)

        # Step 4: Check goal / out
        ball_x, ball_y = self.ball.get_position(denormalized=True)
        goal_owner = shot_owner if self._is_goal(ball_x, ball_y) else None
        ball_out_by = shot_owner if self._is_ball_completely_out(ball_x, ball_y) else None

        # Step 5: Add global context for each agent
        for agent_id in self.agents:
            context = temp_context[agent_id]

            # If the goalkeeper deflected the ball, calculate deflection power
            deflection_power = 0.0
            if context.get("deflected", False):
                deflection_power = np.linalg.norm(self.ball.get_velocity())

            # Update context with global information
            context.update({
                "goal_scored": goal_owner == agent_id,
                "goal_team": self.players[goal_owner].team if goal_owner else None,
                "ball_out_by": ball_out_by,
                "start_shot_bonus": (agent_id == shot_owner and shot_just_started),
                "possession_lost": self._check_possession_loss(agent_id),
                "deflection_power": deflection_power
            })

            infos[agent_id] = context

        # Step 6: Reward calculation
        for agent_id in self.agents:
            role = self.players[agent_id].get_role()
            rewards[agent_id] = get_reward(self.players[agent_id], infos[agent_id], self.reward_grids[role])

        # Step 7: Observations
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

        # Step 8: Termination / Truncation
        for agent_id in self.agents:
            terminations[agent_id] = goal_owner is not None or ball_out_by is not None
            truncations[agent_id] = self.episode_step >= self.max_steps

        return observations, rewards, terminations, truncations, infos


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