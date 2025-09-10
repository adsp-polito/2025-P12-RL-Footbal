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
from football_tactical_ai.configs.configMultiAgentEnv import get_config
from football_tactical_ai.helpers.helperFunctions import normalize
from football_tactical_ai.env.scenarios.multiAgent.rewardGrids import (
    build_attacker_grid,
    build_defender_grid,
    build_goalkeeper_grid,
)
from football_tactical_ai.env.scenarios.multiAgent.multiAgentReward import get_reward


"""
Action space for each player:

- Attacker (ATT): [dx, dy, pass_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
- Defender (DEF): [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
- Goalkeeper (GK): [dx, dy, dive_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)

All values are normalized:
- dx, dy ∈ [-1, 1] (movement direction)
- power ∈ [0, 1]   (kick strength)
- flag > 0.5 → action is triggered
"""


# This is a multi-agent environment for a football tactical AI scenario
class FootballMultiEnv(ParallelEnv):
    """
    A multi-agent environment for football tactical AI scenarios.
    This environment simulates a simple football game with attackers, defenders, and a goalkeeper.
    """

    # Environment metadata
    # This metadata is used by the PettingZoo library
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "football_multi_env",
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the multi-agent football environment.
        
        Args:
            config (dict, optional): Custom configuration dictionary. 
                                    If None, defaults will be loaded from get_config().
        """

        # Rendering mode (not implemented)
        self.render_mode = None

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

        # Attacker roles assignment
        if len(self.attacker_ids) == 3:
            attacker_roles = ["LW", "CF", "RW"]
        elif len(self.attacker_ids) == 2:
            attacker_roles = ["LCF", "RCF"]
        elif len(self.attacker_ids) == 1:
            attacker_roles = ["CF"]
        else:
            raise ValueError("Unsupported number of attackers")

        players.update({
            aid: PlayerAttacker(agent_id=aid, team="A", role=role)
            for aid, role in zip(self.attacker_ids, attacker_roles)
        })


        # Defender roles assignment
        if len(self.defender_ids) == 2:
            defender_roles = ["LCB", "RCB"]
        elif len(self.defender_ids) == 1:
            defender_roles = ["CB"]  # fallback generic central back
        else:
            raise ValueError("Unsupported number of defenders")

        players.update({
            did: PlayerDefender(agent_id=did, team="B", role=role)
            for did, role in zip(self.defender_ids, defender_roles)
        })


        # Goalkeeper (always 1 for now)
        players.update({
            "gk_1": PlayerGoalkeeper(agent_id="gk_1", team="B", role="GK")
        })

        self.players = players

        # Action spaces
        # Each role has a different action vector:
        # - Attacker: [dx, dy, pass_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
        # - Defender: [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
        # - Goalkeeper: [dx, dy, dive_flag, shoot_flag, power, dir_x, dir_y] → shape (7,)
        #
        # dx, dy ∈ [-1, 1] represent movement direction (normalized)
        # power ∈ [0, 1] represents shooting/tackling power
        # Flags are binary actions (activated if > 0.5)
        self.action_spaces = {}
        for aid in self.attacker_ids:
            self.action_spaces[aid] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        for did in self.defender_ids:
            self.action_spaces[did] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        for gid in self.gk_ids:
            self.action_spaces[gid] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation spaces
        # Each agent observes:
        # [self_x, self_y, self_has_ball] +
        # [ball_x, ball_y, ball_vx, ball_vy] +
        # [goal_x, goal_y] +
        # For each other player:
        #   [player_x, player_y, action_code, visible_flag, team_flag, has_ball_flag]
        #
        # action_code = one-hot or discrete integer (e.g., 0=idle, 1=move, 2=pass, 3=shoot, 4=tackle, ...)
        # visible_flag = 1 if in FOV, 0 otherwise
        # team_flag = 1 if teammate, 0 if opponent
        # has_ball_flag = 1 if this player has the ball

        obs_dim = (
            3 +   # self_x, self_y, self_has_ball
            4 +   # ball_x, ball_y, ball_vx, ball_vy
            2 +   # goal_x, goal_y
            (len(self.agents) - 1) * 6  # other players info
        )

        self.observation_spaces = {
            agent_id: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent_id in self.agents
        }

        # Roles mapping
        self.roles = {aid: self.players[aid].get_role() for aid in self.agents}

        # Ball setup
        self.ball = Ball()
        self.episode_step = 0
        self.ball.set_owner("att_1")  # First attacker starts with possession

        # Reward grids per agent
        self.reward_grids = {}
        for agent_id, player in self.players.items():
            role = player.get_role()
            if role in {"LW", "RW", "CF", "LCF", "RCF"}:
                self.reward_grids[agent_id] = build_attacker_grid(self.pitch, role=role)
            elif role in {"LCB", "RCB"}:
                self.reward_grids[agent_id] = build_defender_grid(self.pitch, role=role)
            elif role == "CB":  # fallback if only 1 defender
                self.reward_grids[agent_id] = build_defender_grid(self.pitch)  # no specific role
            elif role == "GK":
                self.reward_grids[agent_id] = build_goalkeeper_grid(self.pitch)
            else:
                raise ValueError(f"Unknown role {role} for agent {agent_id}")

        # Shot context
        self.shot_owner = None
        self.shot_just_started = False

        # Pass context
        self.pass_owner = None
        self.pass_just_started = False


    # PettingZoo interface methods (required for multi-agent environments)
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

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

        # Reset pass context
        self._reset_pass_context()

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Advance the environment by one timestep using the actions of all agents
        """

        # Step 0: Validate input
        if not isinstance(actions, dict):
            raise ValueError("Actions must be a dictionary mapping agent IDs to actions.")
        if not all(agent in actions for agent in self.agents):
            raise ValueError("All agents must provide an action.")

        self.episode_step += 1
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        temp_context = {}

        # Step 1: Execute actions (movement, pass, shot, tackle, dive)
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
                self._process_shot_attempt(agent_id, context)

            # Check if this agent attempted a pass
            if context.get("pass_attempted", False):

                # Verify if the pass is valid (agent owns the ball and has visibility)
                self._process_pass_attempt(agent_id, context)

            temp_context[agent_id] = context

        # Step 2: Handle possession changes (tackle/save/dive/deflection)
        for agent_id, context in temp_context.items():
            if context.get("tackle_success", False) or context.get("blocked", False):
                self.ball.set_owner(agent_id)
                self._reset_shot_context()
                self._reset_pass_context()

                # Mark interception if the player is DEF
                if self.players[agent_id].get_role() in {"DEF"}:
                    context["interception_success"] = True

            if context.get("deflected", False):
                self._reset_shot_context()
                self._reset_pass_context()

        # Step 3: Ball movement
        update_ball_state(
            ball=self.ball,
            players=self.players,
            pitch=self.pitch,
            actions=actions,
            time_step=self.time_step,
            shot_context=self.shot_context,
            pass_context=self.pass_context
        )

        # Assign ball to a nearby player if unowned
        new_owner = self._assign_ball_if_nearby()

        # Step 3b: Check for possession loss by attackers
        if new_owner:
            
            new_owner_team = self.players[new_owner].team

            if self.pass_owner and new_owner_team != self.players[self.pass_owner].team:
                temp_context[new_owner]["interception_success"] = True 
                self.pass_owner = None  # reset pass owner after interception

            elif self.pass_owner and new_owner_team == self.players[self.pass_owner].team and new_owner != self.pass_owner:
                temp_context[new_owner]["pass_completed"] = True
                self.pass_owner = None  # reset pass owner after successful pass


        # Step 4: Check goal or out
        ball_x, ball_y = denormalize(*self.ball.get_position())
        goal_owner = self.shot_owner if self._is_goal(ball_x, ball_y) else None
        ball_out_by = None
        if self._is_ball_completely_out(ball_x, ball_y):
            if self.shot_owner:
                ball_out_by = self.shot_owner
            elif self.pass_owner:
                ball_out_by = self.pass_owner
            elif self.ball.get_owner():
                ball_out_by = self.ball.get_owner()

        # Step 5: Build agent info and contextual updates
        for agent_id in self.agents:
            context = temp_context[agent_id]
            deflection_power = np.linalg.norm(self.ball.get_velocity()) if context.get("deflected") else 0.0

            if goal_owner:
                scorer_team = self.players[goal_owner].team
                # Assign goal to the opposite team if the scorer is DEF or GK
                role = self.players[goal_owner].get_role()
                if role in {"DEF", "GK"}:
                    goal_team = "A" if scorer_team == "B" else "B"
                else:
                    goal_team = scorer_team
            else:
                goal_team = None

            context.update({
                "goal_scored": goal_owner == agent_id,
                "goal_team": goal_team,
                "ball_out_by": ball_out_by,
                "start_shot_bonus": agent_id == self.shot_owner and self.shot_just_started,
                "start_pass_bonus": agent_id == self.pass_owner and self.pass_just_started,
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
                reward_grid=self.reward_grids[agent_id],  
                context=infos[agent_id]
            )
        # Step 7: Observations
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

        # Step 8: Termination / Truncation
        for agent_id in self.agents:
            terminations[agent_id] = goal_owner is not None or ball_out_by is not None
            truncations[agent_id] = self.episode_step >= self.max_steps

        # If ANY agent is done (terminated or truncated), end episode for ALL
        if any(terminations.values()) or any(truncations.values()):
            for agent_id in self.agents:
                terminations[agent_id] = True
                truncations[agent_id] = truncations[agent_id] or self.episode_step >= self.max_steps

        # Step 9: Cleanup for next step
        self._reset_shot_context()
        self._reset_pass_context()

        return observations, rewards, terminations, truncations, infos
    
    def _reset_shot_context(self):
        """
        Reset the shot context after a shot attempt.
        """
        self.shot_context = {"shot_by": None, "direction": None, "power": 0.0}
        self.shot_owner = None
        self.shot_just_started = False

    def _reset_pass_context(self):
        """
        Reset the pass context after a pass attempt.
        """
        self.pass_context = {"pass_by": None, "direction": None, "power": 0.0}
        self.pass_owner = None
        self.pass_just_started = False

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

    def _process_pass_attempt(self, agent_id: str, context: dict):
        """
        Process a pass attempt by an agent.
        Args:
            agent_id (str): ID of the agent attempting the pass.
            context (dict): Context dictionary containing pass details.
        """

        if self.ball.get_owner() == agent_id and context.get("fov_visible", False):
            self.pass_context.update({
                "pass_by": agent_id,
                "direction": context.get("pass_direction"),
                "power": context.get("pass_power")
            })
            self.pass_owner = agent_id
            self.pass_just_started = True
            self.ball.set_owner(None)  # ball now travels

        else:
            context["invalid_pass_attempt"] = True
            context["pass_power"] = 0.0
            context["pass_direction"] = None
            context["pass_attempted"] = False


    def _assign_ball_if_nearby(self, threshold: float = 0.004):
        if self.ball.get_owner() is not None:
            return None  # Already owned

        ball_pos = np.array(self.ball.get_position())
        for agent_id, player in self.players.items():
            player_pos = np.array(player.get_position())
            distance = np.linalg.norm(player_pos - ball_pos)

            if distance < threshold:
                self.ball.set_owner(agent_id)
                return agent_id  # return the new owner
        return None


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
    
    def get_render_state(self):
        """
        Export the current environment state for rendering.
        Returns:
            dict: {
                "players": {agent_id: player_copy},
                "ball": ball_copy
            }
        """
        return {
            "players": {agent: self.players[agent].copy() for agent in self.agents},
            "ball": self.ball.copy()
    }
    
    def _is_in_fov(self, observer, target):
        """
        Check if target is inside the field of view of observer,
        using the observer's last_action_direction as facing vector.
        """
        ox, oy = observer.get_position()
        tx, ty = target.get_position()

        # Direction the observer is facing (must be normalized)
        dir_x, dir_y = observer.last_action_direction
        if np.linalg.norm([dir_x, dir_y]) == 0:
            dir_x, dir_y = 1.0, 0.0  # default facing right if idle

        # Vector to target
        vec_x, vec_y = tx - ox, ty - oy
        distance = np.linalg.norm([vec_x, vec_y])

        # Check max FOV distance
        if distance > observer.fov_range:
            return False

        # Angle between facing direction and target vector
        dot = dir_x * vec_x + dir_y * vec_y
        norm = np.linalg.norm([dir_x, dir_y]) * np.linalg.norm([vec_x, vec_y])
        if norm == 0:
            return False

        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

        return angle <= observer.fov_angle / 2


    def _get_observation(self, agent_id: str):
        """
        Build the observation vector for a given agent.

        Structure:
        [self_x, self_y, self_has_ball] +
        [ball_x, ball_y, ball_vx, ball_vy] +
        [goal_x, goal_y] +
        For each other player:
            [player_x, player_y, action_code, visible_flag, team_flag, has_ball_flag]

        Notes:
        - visible_flag = 1 if agent is in the FOV, else 0
        - team_flag = 1 if teammate, 0 if opponent
        - has_ball_flag = 1 if that player is the current ball owner
        - if not visible now but seen before -> last known position
        - if never seen -> padding with zeros
        """

        player = self.players[agent_id]
        self_x, self_y = player.get_position()

        # Self has ball?
        self_has_ball = 1.0 if self.ball.get_owner() == agent_id else 0.0

        # Ball position and velocity
        ball_x, ball_y = self.ball.get_position()
        ball_vx, ball_vy = self.ball.get_velocity()

        # Goal position (attacker → opponent goal, defender/GK → own goal)
        if player.team == "A":  # attacking team
            goal_x, goal_y = self.pitch.width, self.pitch.center_y
        else:  # defending team
            goal_x, goal_y = 0.0, self.pitch.center_y

        # Start observation vector
        obs = [self_x, self_y, self_has_ball,
            ball_x, ball_y, ball_vx, ball_vy,
            goal_x, goal_y]

        # Ensure we have memory for last known positions
        if not hasattr(self, "last_known_positions"):
            self.last_known_positions = {aid: None for aid in self.agents}

        for other_id, other in self.players.items():
            if other_id == agent_id:
                continue  # skip self

            # Team flag (1 = teammate, 0 = opponent)
            team_flag = 1.0 if other.team == player.team else 0.0

            # Has ball flag
            has_ball_flag = 1.0 if self.ball.get_owner() == other_id else 0.0

            # Check visibility using FOV
            visible = self._is_in_fov(player, other)

            if visible:
                ox, oy = other.get_position()
                action_code = other.get_current_action_code()  # must be defined in Player
                self.last_known_positions[other_id] = (ox, oy, action_code)
                obs.extend([ox, oy, action_code, 1.0, team_flag, has_ball_flag])  # visible
            else:
                if self.last_known_positions[other_id] is not None:
                    ox, oy, action_code = self.last_known_positions[other_id]
                    obs.extend([ox, oy, action_code, 0.0, team_flag, has_ball_flag])  # not visible but known
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0, team_flag, has_ball_flag])  # never seen

        return np.array(obs, dtype=np.float32)
