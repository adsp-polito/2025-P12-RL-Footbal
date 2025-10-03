import numpy as np
from ray.rllib.env import MultiAgentEnv
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
class FootballMultiEnv(MultiAgentEnv):
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

        # Initialize parent class
        super().__init__()

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

        # Define agents dynamically from config
        n_attackers = self.config.get("n_attackers", 3)
        n_defenders = self.config.get("n_defenders", 2)
        include_goalkeeper = self.config.get("include_goalkeeper", True)

        # Attacking team
        self.attacker_ids = [f"att_{i}" for i in range(1, n_attackers + 1)]

        # Defending team
        self.defender_ids = [f"def_{i}" for i in range(1, n_defenders + 1)]

        # Goalkeeper
        self.gk_ids = ["gk_1"] if include_goalkeeper else []

        # Full agent list
        self.agents = self.attacker_ids + self.defender_ids + self.gk_ids
        self.possible_agents = self.agents[:]

        # Complete agent list (PettingZoo requirement)
        self.agents = self.attacker_ids + self.defender_ids + self.gk_ids
        self.possible_agents = self.agents[:]

        # Instantiate players
        players = {}

        # Attacker roles assignment
        if len(self.attacker_ids) == 3:
            attacker_roles = ["CF", "LW", "RW"]         # att_1=CF, att_2=LW, att_3=RW
        elif len(self.attacker_ids) == 2:
            attacker_roles = ["LCF", "RCF"]             # att_1=LCF, att_2=RCF
        elif len(self.attacker_ids) == 1:
            attacker_roles = ["CF"]                     # att_1=CF (fallback generic center forward)    
        else:
            raise ValueError("Unsupported number of attackers")

        players.update({
            aid: PlayerAttacker(agent_id=aid, team="A", role=role)
            for aid, role in zip(self.attacker_ids, attacker_roles)
        })

        # Defender roles assignment
        if len(self.defender_ids) == 2:
            defender_roles = ["RCB", "LCB"]
        elif len(self.defender_ids) == 1:
            defender_roles = ["CB"]
        elif len(self.defender_ids) == 0:
            defender_roles = []
        else:
            raise ValueError("Unsupported number of defenders")

        players.update({
            did: PlayerDefender(agent_id=did, team="B", role=role)
            for did, role in zip(self.defender_ids, defender_roles)
        })


        # Goalkeeper
        if include_goalkeeper:
            players.update({
                "gk_1": PlayerGoalkeeper(agent_id="gk_1", team="B", role="GK")
            })

        self.players = players

        # ======================================================================
        # ACTION SPACES
        #
        # Each player now uses a flat Box(7,) action space:
        # [dx, dy, flag1, flag2, power, dir_x, dir_y]
        #
        # - dx, dy ∈ [-1, 1] → movement direction
        # - flag1, flag2 ∈ [0, 1] → treated as binary with threshold (e.g. > 0.5)
        # - power ∈ [0, 1] → kick strength
        # - dir_x, dir_y ∈ [-1, 1] → direction vector
        #
        # Notes:
        # - Attackers interpret (flag1=pass, flag2=shoot)
        # - Defenders interpret (flag1=tackle, flag2=shoot)
        # - Goalkeeper interprets (flag1=dive, flag2=shoot)
        # ======================================================================

        self.action_spaces = {}
        for agent_id in self.agents:
            self.action_spaces[agent_id] = spaces.Box(
                low=np.array([-1, -1, 0, 0, 0, -1, -1], dtype=np.float32),
                high=np.array([ 1,  1, 1, 1, 1,  1,  1], dtype=np.float32),
                shape=(7,),
                dtype=np.float32,
            )


        # ======================================================================
        # OBSERVATION SPACE
        #
        # Each agent observes a flat vector:
        #
        # 1. Self (3):
        #    [self_x, self_y, self_has_ball]
        #
        # 2. Ball (4):
        #    [ball_x, ball_y, ball_vx, ball_vy]
        #
        # 3. Goal (2):
        #    [goal_x, goal_y]
        #
        # 4. Own parameters (depends on role):
        #    - Attacker: [shooting, passing, dribbling, speed, precision, fov_angle, fov_range] → 7
        #    - Defender: [shooting, tackling, speed, precision, fov_angle, fov_range] → 6
        #    - Goalkeeper: [shooting, reflexing, punch_power, reach, catching, speed, precision, fov_angle, fov_range] → 9
        #
        # 5. Other players (6 per player):
        #    [rel_x, rel_y, action_code, visible_flag, team_flag, has_ball_flag]
        #
        # Total dim = 3 + 4 + 2 + params_dim + (N-1)*6 ===> min: (3+4+2+9+(0*6))=18, max: (3+4+2+9+(5*6))=48
        # ======================================================================

        def _compute_obs_dim(player):
            base_dim = 3 + 4 + 2 + (len(self.agents) - 1) * 6
            role = player.get_role()
            if role in {"CF", "LW", "RW", "LCF", "RCF", "SS", "ATT"}:
                return base_dim + 7
            elif role in {"LCB", "RCB", "CB", "DEF"}:
                return base_dim + 6
            elif role == "GK":
                return base_dim + 9
            else:
                return base_dim

        self.observation_spaces = {
            agent_id: spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(_compute_obs_dim(player),), 
                dtype=np.float32
            )
            for agent_id, player in self.players.items()
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
            team = player.team  # "A" or "B"

            if role in {"LW", "RW", "CF", "LCF", "RCF", "SS", "ATT"}:
                self.reward_grids[agent_id] = build_attacker_grid(self.pitch, role=role, team=team)
            elif role in {"LCB", "RCB", "CB", "DEF"}:
                self.reward_grids[agent_id] = build_defender_grid(self.pitch, role=role, team=team)
            elif role == "GK":
                self.reward_grids[agent_id] = build_goalkeeper_grid(self.pitch, team=team)
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
            observations (dict): Initial observations for all active agents.
            infos (dict): Auxiliary info for each agent (empty at reset).
        """

        # Reset environment state
        self.agents = self.possible_agents[:]     # restore full agent list
        self.episode_step = 0                     # reset step counter
        self.ball.reset()                         # reset ball physics
        self.ball.set_owner("att_1")              # default: att_1 starts with ball

        # Define start positions dynamically depending on the number of agents
        start_positions = {}

        # Attacker positions (Team A)
        if len(self.attacker_ids) == 3:
            coords = [(60, 40), (60, 30), (60, 50)]   # CF central, LW, RW
        elif len(self.attacker_ids) == 2:
            coords = [(60, 35), (60, 45)]             # two forwards, slightly apart
        elif len(self.attacker_ids) == 1:
            coords = [(60, 40)]                       # single central striker
        else:
            coords = []

        for aid, pos in zip(self.attacker_ids, coords):
            start_positions[aid] = pos

        # Defender positions (Team B)
        if len(self.defender_ids) == 2:
            coords = [(100, 30), (100, 50)]           # two CBs (RCB and LCB)
        elif len(self.defender_ids) == 1:
            coords = [(100, 40)]                      # single CB central
        else:
            coords = []

        for did, pos in zip(self.defender_ids, coords):
            start_positions[did] = pos

        # Goalkeeper (Team B)
        if self.gk_ids:                               # only if GK enabled
            start_positions["gk_1"] = (120, 40)

        # Apply initial positions to players
        for aid, player in self.players.items():
            if aid in start_positions:
                x, y = start_positions[aid]
                player.reset_position(normalize(x, y))

        # Build initial observations for each agent
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        # Reset contexts (shot, pass)
        self._reset_shot_context()
        self._reset_pass_context()

        # Initialize info dict (empty per agent)
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, infos


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

        # Assign ball to a nearby player if unowned
        new_owner = self._assign_ball_if_nearby()

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
                if self.players[agent_id].get_role() in {"DEF", "LCB", "RCB", "CB"}:
                    context["interception_success"] = True

            if context.get("deflected", False):
                self._reset_shot_context()
                self._reset_pass_context()

        # Step 3: Ball movement
        collision = update_ball_state(
            ball=self.ball,
            players=self.players,
            pitch=self.pitch,
            actions=actions,
            time_step=self.time_step,
            shot_context=self.shot_context,
            pass_context=self.pass_context
        )

      # Step 3b: Check for possession loss by attackers
        if new_owner:
            new_owner_team = self.players[new_owner].team

            if self.pass_owner:
                target_id = self.pass_context.get("pass_to")

                if new_owner_team != self.players[self.pass_owner].team:
                    # Pass intercepted by opponent
                    temp_context[new_owner]["interception_success"] = True
                    temp_context[self.pass_owner]["pass_intercepted"] = True

                else:
                    # Ball still within same team
                    if target_id and new_owner == target_id:
                        # Intended receiver got the ball
                        temp_context[new_owner]["pass_completed"] = True
                        temp_context[new_owner]["pass_from"] = self.pass_owner
                        temp_context[self.pass_owner]["pass_completed"] = True
                        temp_context[self.pass_owner]["pass_to"] = new_owner
                    else:
                        # Wrong teammate received the ball
                        temp_context[new_owner]["pass_completed"] = False
                        temp_context[new_owner]["pass_from"] = self.pass_owner
                        temp_context[self.pass_owner]["pass_completed"] = False
                        temp_context[self.pass_owner]["pass_to"] = new_owner
                        temp_context[self.pass_owner]["pass_missed_target"] = True

                # Reset pass context after resolution
                self.pass_owner = None
                self.pass_context["pass_to"] = None
                self.pass_just_started = False   


        # Step 4: Check goal or out
        ball_x, ball_y = denormalize(*self.ball.get_position())
        goal_owner = None
        goal_team = None

        # Determine responsible team:
        # - Prefer shooter (shot_owner)
        # - Otherwise, fall back to ball owner
        # - If neither exists → no scorer team
        if self.shot_owner is not None:
            scorer_team = self.players[self.shot_owner].team

            # Cleanup in case of goal
            self.shot_owner = None

        elif self.ball.get_owner() is not None:
            scorer_team = self.players[self.ball.get_owner()].team
            
        else:
            scorer_team = None

        # Always check goal conditions if we have a valid scorer_team
        if scorer_team is not None:

            # Case 1: Normal goal (ball enters opponent's net)
            if self._is_goal(ball_x, ball_y, scorer_team):
                goal_team = scorer_team
                if self.shot_owner is not None:
                    goal_owner = self.shot_owner
                else:
                    # No shooter → fallback to ball owner if available
                    if self.ball.get_owner() is not None:
                        goal_owner = self.ball.get_owner()
                
                # Cleanup owners
                self.shot_owner = None
                self.pass_owner = None

            # Case 2: Own goal (ball enters own net)
            elif self._is_own_goal(ball_x, ball_y, scorer_team):
                goal_team = "A" if scorer_team == "B" else "B"
                if self.shot_owner is not None:
                    goal_owner = self.shot_owner
                else:
                    # No shooter → fallback to ball owner if available
                    if self.ball.get_owner() is not None:
                        goal_owner = self.ball.get_owner()

                # Cleanup owners
                self.shot_owner = None
                self.pass_owner = None


        ball_out_by = None
        if self._is_ball_completely_out(ball_x, ball_y):
            if self.shot_owner is not None:
                ball_out_by = self.shot_owner
            elif self.pass_owner is not None:
                ball_out_by = self.pass_owner
            elif self.ball.get_owner() is not None:
                ball_out_by = self.ball.get_owner()
            
            # Cleanup owners
            self.shot_owner = None
            self.pass_owner = None

        # Step 5: Build agent info and contextual updates
        for agent_id in self.agents:
            context = temp_context[agent_id]
            role = self.players[agent_id].get_role()

            # COMMON FIELDS 
            context.update({
                "goal_scored": goal_owner == agent_id,
                "goal_team": goal_team,
                "ball_out_by": ball_out_by,
                "start_shot_bonus": agent_id == self.shot_owner and self.shot_just_started
            })

            # ATTACKER-SPECIFIC
            if role in {"CF","LW","RW","LCF","RCF","SS","ATT"}:
                context.update({
                    "start_pass_bonus": agent_id == self.pass_owner and self.pass_just_started,
                    "possession_lost": self._check_possession_loss(agent_id),
                })

            # GOALKEEPER-SPECIFIC
            elif role in {"GK"}:
                # Deflection power
                deflection_power = np.linalg.norm(self.ball.get_velocity()) if context.get("deflected") else 0.0
                context.update({
                    "deflection_power": deflection_power
                })

            infos[agent_id] = context

        # Step 6: Reward calculation
        for agent_id in self.agents:
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
        terminated_event = (goal_team is not None) or (ball_out_by is not None)
        timeout_event = self.episode_step >= self.max_steps

        if terminated_event:
            # Episode closed by goal or out of bounds
            for agent_id in self.agents:
                terminations[agent_id] = True
                truncations[agent_id] = False
        elif timeout_event:
            # Episode closed by timeout
            for agent_id in self.agents:
                terminations[agent_id] = False
                truncations[agent_id] = True
        else:
            # Episode still ongoing
            for agent_id in self.agents:
                terminations[agent_id] = False
                truncations[agent_id] = False

        # Global termination / truncation (for RLib compliance)
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"]  = any(truncations.values())

        # Step 9: Cleanup for next step
        self.shot_just_started = False
        self.pass_just_started = False

        self._reset_shot_context()
        self._reset_pass_context()

        return observations, rewards, terminations, truncations, infos
    
    def _reset_shot_context(self):
        """
        Reset the shot context after a shot attempt.
        """
        self.shot_context = {"shot_by": None, "direction": None, "power": 0.0}
        self.shot_just_started = False

    def _reset_pass_context(self):
        """
        Reset the pass context after a pass attempt.
        """
        self.pass_context = {"pass_from": None, "direction": None, "power": 0.0, "pass_to": None}
        self.pass_just_started = False

    def _process_shot_attempt(self, agent_id: str, context: dict):
        """
        Validate and register a shot attempt.
        - Only stores info into self.shot_context.
        - The actual detachment of the ball is handled in update_ball_state.
        """

        direction = context.get("shot_direction")
        power = context.get("shot_power", 0.0)

        # Validate before setting context
        if direction is None or np.linalg.norm(direction) < 1e-6 or power <= 0:
            context["invalid_shot_attempt"] = True
            return

        if self.ball.get_owner() == agent_id and context.get("fov_visible", False):
            self.shot_context.update({
                "shot_by": agent_id,
                "direction": direction,
                "power": power
            })
            self.shot_owner = agent_id
            self.shot_just_started = True
            context["shot_attempted"] = True
        else:
            context["invalid_shot_attempt"] = True
            context["shot_attempted"] = False


    def _process_pass_attempt(self, agent_id: str, context: dict):
        """
        Validate and register a pass attempt.
        - Only stores info into self.pass_context.
        - The actual detachment of the ball is handled in update_ball_state.
        """

        power = context.get("pass_power", 0.0)

        if self.ball.get_owner() != agent_id or power <= 0:
            context["invalid_pass_attempt"] = True
            context["pass_attempted"] = False
            return

        passer = self.players[agent_id]
        passer_team = passer.team
        passer_pos = np.array(passer.get_position())

        teammates = [(pid, p) for pid, p in self.players.items() if p.team == passer_team and pid != agent_id]
        if not teammates:
            context["invalid_pass_attempt"] = True
            context["pass_attempted"] = False
            return

        # Closest teammate
        target_id, target = min(teammates, key=lambda kv: np.linalg.norm(np.array(kv[1].get_position()) - passer_pos))
        target_pos = np.array(target.get_position())

        direction = target_pos - passer_pos
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1.0, 0.0])  # fallback

        # Save pass context
        self.pass_context.update({
            "pass_from": agent_id,
            "pass_to": target_id,
            "direction": direction,
            "power": power
        })

        self.pass_owner = agent_id
        self.pass_just_started = True

        context["pass_attempted"] = True
        context["pass_to"] = target_id
        context["pass_power"] = power
        context["pass_direction"] = direction.tolist()
        context["fov_pass"] = self._is_in_fov(passer, target)


    def _assign_ball_if_nearby(self, threshold: float = 0.017):  # in normalized units ~ 2 meter
        if self.ball.get_owner() is not None:
            return None  # Already owned

        ball_pos = np.array(self.ball.get_position())
        for agent_id, player in self.players.items():
            player_pos = np.array(player.get_position())
            distance = np.linalg.norm(player_pos - ball_pos)

            # Prevent shooter from instantly re-taking the ball
            if distance < threshold:
                if agent_id == self.shot_owner or agent_id == self.pass_owner:
                    continue  # skip to next player

                self.ball.set_owner(agent_id)
                # self.ball.set_position(player_pos)

                # Reset contexts after real possession change
                self._reset_shot_context()
                self._reset_pass_context()
                return agent_id
        return None



    def _check_possession_loss(self, agent_id: str) -> bool:
        """
        Check if an attacker has lost possession specifically
        to a defender or goalkeeper.
        """
        # Only attackers can lose possession in this way
        if not agent_id.startswith("att"):
            return False

        new_owner = self.ball.get_owner()

        # If ball is unowned or still with the same attacker, no loss
        if new_owner is None or new_owner == agent_id:
            return False

        # Check if the new owner is a defender or goalkeeper
        if new_owner in self.players:
            role = self.players[new_owner].get_role()
            return role in {"DEF", "LCB", "RCB", "CB", "GK"}

        return False



    def _is_ball_completely_out(self, ball_x_m, ball_y_m):
        """
        Check if ball is completely out of play.
        - If ball is inside the goal area (between posts, behind goal line),
        do nothing until it touches the net (back or sides).
        - Otherwise, normal out-of-bounds detection.
        """

        margin_m = 1.0   # margin outside pitch
        eps = 0.01        # tolerance for hitting side net

        GOAL_MIN_Y = self.pitch.center_y - self.pitch.goal_width / 2
        GOAL_MAX_Y = self.pitch.center_y + self.pitch.goal_width / 2
        GOAL_DEPTH = self.pitch.goal_depth

        # Ball in right goal area (not yet out)
        if self.pitch.width < ball_x_m <= self.pitch.width + GOAL_DEPTH and GOAL_MIN_Y <= ball_y_m <= GOAL_MAX_Y:
            return False

        # Ball in left goal area (not yet out)
        if -GOAL_DEPTH <= ball_x_m < 0 and GOAL_MIN_Y <= ball_y_m <= GOAL_MAX_Y:
            return False

        # Ball hits back net
        if ball_x_m > self.pitch.width + GOAL_DEPTH and GOAL_MIN_Y <= ball_y_m <= GOAL_MAX_Y:
            return True
        if ball_x_m < -GOAL_DEPTH and GOAL_MIN_Y <= ball_y_m <= GOAL_MAX_Y:
            return True

        # Ball hits side net (only if aligned with posts)
        if ball_x_m > self.pitch.width and abs(ball_y_m - GOAL_MIN_Y) < eps:
            return True
        if ball_x_m > self.pitch.width and abs(ball_y_m - GOAL_MAX_Y) < eps:
            return True
        if ball_x_m < 0 and abs(ball_y_m - GOAL_MIN_Y) < eps:
            return True
        if ball_x_m < 0 and abs(ball_y_m - GOAL_MAX_Y) < eps:
            return True

        # Normal out-of-bounds
        if (ball_x_m < -margin_m or
            ball_x_m > self.pitch.width + margin_m or
            ball_y_m < -margin_m or
            ball_y_m > self.pitch.height + margin_m):
            return True

        return False
    
    def _is_goal(self, x, y, scoring_team: str):
        """
        Check if the ball is a valid goal for the given scoring_team.
        scoring_team: "A" or "B"
        """
        GOAL_MIN_Y = self.pitch.center_y - self.pitch.goal_width / 2
        GOAL_MAX_Y = self.pitch.center_y + self.pitch.goal_width / 2
        r_ball = self.ball.radius

        # Team A attacks right goal
        if scoring_team == "A":
            return x - r_ball > self.pitch.width and GOAL_MIN_Y <= y <= GOAL_MAX_Y

        # Team B attacks left goal
        elif scoring_team == "B":
            return x + r_ball < 0 and GOAL_MIN_Y <= y <= GOAL_MAX_Y

        return False


    def _is_own_goal(self, x, y, scoring_team: str):
        """
        Check if the ball is an own goal for the given scoring_team.
        scoring_team: "A" or "B"
        """
        GOAL_MIN_Y = self.pitch.center_y - self.pitch.goal_width / 2
        GOAL_MAX_Y = self.pitch.center_y + self.pitch.goal_width / 2
        r_ball = self.ball.radius

        # Team A defends left goal → own goal if it goes there
        if scoring_team == "A":
            return x + r_ball < 0 and GOAL_MIN_Y <= y <= GOAL_MAX_Y

        # Team B defends right goal → own goal if it goes there
        elif scoring_team == "B":
            return x - r_ball > self.pitch.width and GOAL_MIN_Y <= y <= GOAL_MAX_Y

        return False


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
        [own_parameters...] +
        For each other player:
            [rel_x, rel_y, action_code, visible_flag, team_flag, has_ball_flag]
        """

        player = self.players[agent_id]
        self_x, self_y = player.get_position()

        # Self has ball?
        self_has_ball = 1.0 if self.ball.get_owner() == agent_id else 0.0

        # Ball info (normalized)
        ball_x, ball_y = self.ball.get_position()
        ball_vx, ball_vy = self.ball.get_velocity()

        # Goal position (depends on team)
        if player.team == "A":  # attacking team → right goal
            goal_x, goal_y = self.pitch.width, self.pitch.center_y
        else:  # defending team → left goal
            goal_x, goal_y = 0.0, self.pitch.center_y

        # Normalize goal coordinates
        goal_x, goal_y = normalize(goal_x, goal_y)

        # Start observation vector with base features
        obs = [
            self_x, self_y, self_has_ball,
            ball_x, ball_y, ball_vx, ball_vy,
            goal_x, goal_y
        ]

        # Add own parameters (role-dependent)
        # These values are already normalized in [0, 1].
        # They allow the shared policy (per role) to differentiate agents
        role = player.get_role()
        if role in {"CF", "LW", "RW", "LCF", "RCF", "SS", "ATT"}:
            params = [
                player.shooting,
                player.passing,
                player.dribbling,
                player.speed,
                player.precision,
                player.fov_angle,
                player.fov_range,
            ]
        elif role in {"LCB", "RCB", "CB", "DEF"}:
            params = [
                player.shooting,
                player.tackling,
                player.speed,
                player.precision,
                player.fov_angle,
                player.fov_range,
            ]
        elif role == "GK":
            params = [
                player.shooting,
                player.reflexes,
                player.punch_power,
                player.reach,
                player.catching,
                player.speed,
                player.precision,
                player.fov_angle,
                player.fov_range,
            ]
        else:
            params = []

        obs.extend(params)

        # Memory for last known positions of other players
        # This allows the agent to "remember" where unseen players were
        if not hasattr(self, "last_known_positions"):
            self.last_known_positions = {aid: None for aid in self.agents}

        # Add information about other players
        # Each other player contributes 6 values:
        # [rel_x, rel_y, action_code, visible_flag, team_flag, has_ball_flag]
        for other_id, other in self.players.items():
            if other_id == agent_id:
                continue

            team_flag = 1.0 if other.team == player.team else 0.0
            has_ball_flag = 1.0 if self.ball.get_owner() == other_id else 0.0
            visible = self._is_in_fov(player, other)

            if visible:
                ox, oy = other.get_position()
                rel_x = ox - self_x
                rel_y = oy - self_y
                action_code = other.get_current_action_code()

                # Store last known info
                self.last_known_positions[other_id] = (rel_x, rel_y, action_code)
                obs.extend([rel_x, rel_y, action_code, 1.0, team_flag, has_ball_flag])
            else:
                # Use last known position if available, otherwise -1
                if self.last_known_positions[other_id] is not None:
                    rel_x, rel_y, action_code = self.last_known_positions[other_id]
                    obs.extend([rel_x, rel_y, action_code, 0.0, team_flag, has_ball_flag])
                else:
                    obs.extend([-1.0, -1.0, -1.0, 0.0, team_flag, has_ball_flag])

        # Check that the final observation vector matches the expected dimension
        expected_dim = self.observation_spaces[agent_id].shape[0]
        assert len(obs) == expected_dim, (
            f"Observation length mismatch for {agent_id}: "
            f"got {len(obs)}, expected {expected_dim}"
        )

        return np.array(obs, dtype=np.float32)
