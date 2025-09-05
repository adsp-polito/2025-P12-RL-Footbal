import numpy as np
from pettingzoo import ParallelEnv
from typing import Dict, Any
from gymnasium import spaces

from football_tactical_ai.players.playerAttacker import PlayerAttacker
from football_tactical_ai.players.playerDefender import PlayerDefender
from football_tactical_ai.players.playerGoalkeeper import PlayerGoalkeeper
from football_tactical_ai.env.scenarios.multiAgent.physics import update_ball_state
from football_tactical_ai.helpers.helperFunctions import denormalize
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
                                    If None, defaults will be loaded from get_config().
        """
        # Load configuration parameters
        self.config = config or get_config()
        self.fps = self.config["fps"]
        self.time_step = self.config["time_step"]
        self.max_steps = self.config["max_steps"]

        # Pitch setup
        self.pitch = Pitch()
        self.x_range = self.pitch.width
        self.y_range = self.pitch.height

        # Define agents (IDs)

        # Attacking team: 3 attackers (att_1, att_2, att_3)
        self.attacker_ids = [f"att_{i}" for i in range(1, 4)]

        # Defending team: 2 defenders (def_1, def_2)
        self.defender_ids = [f"def_{i}" for i in range(1, 3)]

        # Defending team: 1 goalkeeper (gk_1)
        self.gk_ids = ["gk_1"]

        # Complete agent list (PettingZoo requirement)
        self.agents = self.attacker_ids + self.defender_ids + self.gk_ids
        self.possible_agents = self.agents[:]

        # Instantiate players
        players = {}

        # Attackers
        players.update({
            aid: PlayerAttacker(agent_id=aid, team="A", role="ATT")
            for aid in self.attacker_ids
        })

        # Defenders
        players.update({
            did: PlayerDefender(agent_id=did, team="B", role="DEF")
            for did in self.defender_ids
        })

        # Goalkeeper
        players.update({
            "gk_1": PlayerGoalkeeper(agent_id="gk_1", team="B", role="GK")
        })

        self.players = players

        # Action spaces
        # Each role has a different action vector:
        # - Attacker: [dx, dy, shoot_flag, power, dir_x, dir_y] → shape (6,)
        # - Defender: [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
        # - Goalkeeper: [dx, dy, dive_left, dive_right, shoot_flag, power, dir_x, dir_y] → shape (8,)
        #
        # dx, dy ∈ [-1, 1] represent movement direction (normalized)
        # power ∈ [0, 1] represents shooting/tackling power
        # Flags are binary actions (activated if > 0.5)
        self.action_spaces = {}
        for aid in self.attacker_ids:
            self.action_spaces[aid] = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        for did in self.defender_ids:
            self.action_spaces[did] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        for gid in self.gk_ids:
            self.action_spaces[gid] = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # Observation spaces
        # Each agent observes:
        # [player_x, player_y, ball_x, ball_y] → shape (4,)
        # All values are normalized ∈ [0, 1]
        self.observation_spaces = {
            agent_id: spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent_id in self.agents
        }

        # Roles mapping
        self.roles = {aid: self.players[aid].get_role() for aid in self.agents}

        # Ball setup
        self.ball = Ball()
        self.episode_step = 0
        self.ball.set_owner("att_1")  # First attacker starts with possession

        # Reward grids
        # Each role has its own spatial reward grid
        self.reward_grids = {
            "ATT": build_attacker_grid(self.pitch),
            "DEF": build_defender_grid(self.pitch),
            "GK": build_goalkeeper_grid(self.pitch),
        }

        # Shot context
        self.shot_owner = None
        self.shot_just_started = False


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

        # Reset the environment state
        self.agents = self.possible_agents[:]
        self.episode_step = 0
        self.ball.reset()
        self.ball.set_owner("att_1")  # Initial ball ownership

        # Reset positions
        start_positions = {
            "att_1": (60, 40),
            "att_2": (60, 30),
            "att_3": (60, 50),
            "def_1": (100, 30),
            "def_2": (100, 50),
            "gk_1":  (120, 40)
        }

        for aid, player in self.players.items():
            if aid in start_positions:
                x, y = start_positions[aid]
                player.reset_position(normalize(x, y))

        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        # Reset shot context
        self._reset_shot_context()

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Advance the environment by one timestep using the actions of all agents.
        """

        # Step 0: Validate input
        if not isinstance(actions, dict):
            raise ValueError("Actions must be a dictionary mapping agent IDs to actions.")
        if not all(agent in actions for agent in self.agents):
            raise ValueError("All agents must provide an action.")

        self.episode_step += 1
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        temp_context = {}

        # Step 1: Execute actions and handle shooting
        for agent_id, action in actions.items():
            player = self.players[agent_id]
            context = player.execute_action(
                action=action,
                time_step=self.time_step,
                x_range=self.x_range,
                y_range=self.y_range,
                ball=self.ball
            )

             # Check if this agent attempted a shot
            if context.get("shot_attempted", False):
                # Verify if the shot is valid (agent owns the ball and has visibility)
                # set the context here, inside the method
                self._process_shot_attempt(agent_id, context)

            temp_context[agent_id] = context

        # Step 2: Handle possession changes (tackle/save/dive)
        for agent_id, context in temp_context.items():
            if context.get("tackle_success", False) or context.get("blocked", False):
                self.ball.set_owner(agent_id)
                self._reset_shot_context()
            if context.get("deflected", False):
                self._reset_shot_context()

        # Step 3: Ball movement
        update_ball_state(
            ball=self.ball,
            players=self.players,
            pitch=self.pitch,
            actions=actions,
            time_step=self.time_step,
            shot_context=self.shot_context
        )

        # Assign ball to a nearby player if unowned
        self._assign_ball_if_nearby()

        # Step 4: Check goal or out
        ball_x, ball_y = denormalize(*self.ball.get_position())
        goal_owner = self.shot_owner if self._is_goal(ball_x, ball_y) else None
        ball_out_by = self.shot_owner if self._is_ball_completely_out(ball_x, ball_y) else None

        # Step 5: Build agent info and shot context
        for agent_id in self.agents:
            context = temp_context[agent_id]
            deflection_power = np.linalg.norm(self.ball.get_velocity()) if context.get("deflected") else 0.0

            context.update({
                "goal_scored": goal_owner == agent_id,
                "goal_team": self.players[goal_owner].team if goal_owner else None,
                "ball_out_by": ball_out_by,
                "start_shot_bonus": agent_id == self.shot_owner and self.shot_just_started,
                "possession_lost": self._check_possession_loss(agent_id),
                "deflection_power": deflection_power
            })

            infos[agent_id] = context

        # Step 6: Reward calculation
        for agent_id in self.agents:
            role = self.players[agent_id].get_role()
            rewards[agent_id] = get_reward(
                player=self.players[agent_id],
                ball=self.ball,
                pitch=self.pitch,
                reward_grid=self.reward_grids[role],
                context=infos[agent_id]
    )

        # Step 7: Observations
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

        # Step 8: Termination / Truncation
        for agent_id in self.agents:
            terminations[agent_id] = goal_owner is not None or ball_out_by is not None
            truncations[agent_id] = self.episode_step >= self.max_steps

        return observations, rewards, terminations, truncations, infos
    
    def _reset_shot_context(self):
        """
        Reset the shot context after a shot attempt.
        """
        self.shot_context = {"shot_by": None, "direction": None, "power": 0.0}
        self.shot_owner = None
        self.shot_just_started = False

    def _process_shot_attempt(self, agent_id: str, context: dict):
        """
        Process a shot attempt by an agent.
        Args:
            agent_id (str): ID of the agent attempting the shot.
            context (dict): Context dictionary containing shot details.
        """
        if self.ball.get_owner() == agent_id and context.get("fov_visible", False):
            self.shot_context.update({
                "shot_by": agent_id,
                "direction": context.get("shot_direction"),
                "power": context.get("shot_power")
            })
            self.shot_owner = agent_id
            self.shot_just_started = True
            self.ball.set_owner(None)
        else:
            context["invalid_shot_attempt"] = True
            context["shot_power"] = 0.0
            context["shot_direction"] = None
            context["shot_attempted"] = False

    def _assign_ball_if_nearby(self, threshold: float = 0.004):
        if self.ball.get_owner() is not None:
            return  # Already owned

        ball_pos = np.array(self.ball.get_position())
        for agent_id, player in self.players.items():
            player_pos = np.array(player.get_position())
            distance = np.linalg.norm(player_pos - ball_pos)
            if distance < threshold:  # If player is close enough to the ball (0.004 normalized units ~0.5 meters)
                self.ball.set_owner(agent_id)
                break  # Only assign to the first one found

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