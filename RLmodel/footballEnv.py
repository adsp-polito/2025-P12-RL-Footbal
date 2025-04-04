import gymnasium as gym
import numpy as np

from RLmodel.player import Player
from RLmodel.team import Team
from RLmodel.ball import Ball

DISTANCE_TO_BALL = 0.005  # ~0.60 meters if pitch is 120m wide

class FootballEnv(gym.Env):
    """
    FootballEnv models a football simulation environment tailored for reinforcement learning.
    It simulates an 11 vs 11 football scenario on a 2D pitch, with two teams and a ball.
    Each player can be controlled either by a learned policy (agent) or rule-based logic.

    This environment follows the Gymnasium API, supporting standard methods such as
    reset(), step(), render(), and close().

    Attributes:
        - field_width (int): Width of the football pitch in meters.
        - field_height (int): Height of the football pitch in meters.
        - players (list): List of all 22 Player instances (11 per team).
        - teams (list): List containing the two Team instances.
        - ball (Ball): The Ball object representing the football.
        - initial_possessor_id (int or None): ID of the player to receive initial possession (from 0 to 21),
                                              or None to assign possession automatically.
        - current_step (int): Internal counter tracking simulation time steps.
        - observation_space (gym.Space): Defines the structure and bounds of observations.
        - action_space (gym.Space): Defines the set and shape of available agent actions.
        - render_mode (str or None): Rendering mode ("human" for GUI, or None for headless).
        - window (object or None): Window object used for graphical rendering (if enabled).
    """

    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(self, render_mode=None, teamAColor='red', teamBColor='blue',
                 teamAFormation='433', teamBFormation='433', 
                 initial_possessor_id=None):
        super().__init__()

        # Field dimensions
        self.field_width = 120
        self.field_height = 80

        # Frame per seconds
        self.fps = 24

        # Initialize lists for players and teams
        self.players = []
        self.teams = []

        # Create the ball instance
        self.ball = Ball()

        # Track the initial owner for reuse during resets
        self.initial_possessor_id = initial_possessor_id

        # Step counter to keep track of simulation time
        self.current_step = 0

        # Define placeholder observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(22, 10), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([10 for _ in range(11)])

        # Rendering options
        self.render_mode = render_mode
        self.window = None

        # Initialize the teams with custom colors
        self._init_teams(teamAColor, teamBColor, teamAFormation, teamBFormation)

        # Assign initial ball possession
        self._assign_initial_possession(self.initial_possessor_id)

    def _init_teams(self, teamAColor, teamBColor, teamAFormation, teamBFormation):
        """Initializes both teams with 11 players each and assigns colors and sides"""
        team_A = Team(team_id=0, color=teamAColor, side="left", formation=teamAFormation)
        team_B = Team(team_id=1, color=teamBColor, side="right", formation=teamBFormation)
        self.teams = [team_A, team_B]
        self.players = team_A.players + team_B.players

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state before starting a new simulation"""
        super().reset(seed=seed)
        self.current_step = 0

        # Reset ball state
        self.ball.reset()

        # Reset all players
        for player in self.players:
            player.reset()

        # Assign possession to the appropriate player
        self._assign_initial_possession(self.initial_possessor_id)

        # Return initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, actions):
        """
        Advances the environment by one step using the given actions.

        Args:
            actions (list): A list of actions, one per player. Each action can be a string (e.g., "move") 
                            or a tuple (e.g., ("pass", receiver_id))

        Returns:
            - observation: Updated observation after applying actions
            - reward: Numerical feedback signal (to be defined)
            - done: Boolean indicating if the episode has ended
            - truncated: Placeholder (False)
            - info: Dictionary with additional metadata
        """
        self.current_step += 1
        receiver_id = None
        passer_id = None

        # Apply each player's action
        for i, player in enumerate(self.players):
            action = actions[i]

            if isinstance(action, tuple) and action[0] == "pass":
                
                # Handle passing action
                # action[1] is the receiver_id
                # action[0] is the action type (e.g., "pass")
                receiver_id = action[1] 
                receiver = self.players[receiver_id]

                passer = player
                passer_id = passer.player_id

                passer.pass_to(receiver, self.ball)
                self.ball.owner_id = None  # Remove ball ownership temporarily
            else:
                player.step(action)

        # Update ball position
        self.ball.update_position(self.players)

        # If the ball is free, assign it to the first player close enough (excluding the passer)
        if self.ball.owner_id is None:

            for player in self.players:
                if passer_id == player.player_id:
                    continue  # Skip the passer to avoid instant re-possession

                distance = np.linalg.norm(player.position - self.ball.position)
                print(f"[DEBUG] Player {player.player_id} distance to ball: {distance:.4f}")
                if distance < DISTANCE_TO_BALL:
                    print("[INFO] Player {} is close enough to the ball".format(player.player_id))
                    self.ball.owner_id = player.player_id
                    player.has_ball = True

                    # Reset ball possession for all other players
                    for other_player in self.players:
                        if other_player.player_id != player.player_id:
                            other_player.has_ball = False

                    print(f"[INFO] Player {player.player_id} gained ball possession")
                    break
                else:
                    player.has_ball = False

        # Check for end of episode conditions
        done, reward, info = self._check_ball_out_of_bounds()

        # Return new observation and metadata
        observation = self._get_observation()
        return observation, reward, done, False, info

    def _get_observation(self):
        """
        Returns the current observation of the environment
        This will later include positions, velocities, and status of all players and the ball
        """
        obs = np.zeros((22, 10), dtype=np.float32)
        return obs

    def render(self):
        """
        Renders the current environment frame
        For now, this is a placeholder that simply prints the frame number
        """
        if self.render_mode == "human":
            print("Rendering frame", self.current_step)

    def close(self):
        """
        Properly closes any rendering windows or processes
        """
        if self.window:
            self.window.close()
            self.window = None

    def _assign_initial_possession(self, initial_possessor_id=None):
        """
        Assigns initial ball possession to a specified player (if given),
        or automatically assigns it to the Team A player closest to the ball's position.

        Args:
            initial_possessor_id (int, optional): ID of the player to assign ball possession to.
                                                If None, defaults to closest player in Team A.
        """
        if initial_possessor_id is not None:
            self.players[initial_possessor_id].has_ball = True
            self.ball.owner_id = initial_possessor_id
        else:
            ball_spawn_point = self.ball.get_position()
            team_A_players = self.players[:11]
            distances = [np.linalg.norm(p.get_position() - ball_spawn_point) for p in team_A_players]
            closest_index = np.argmin(distances)
            self.players[closest_index].has_ball = True
            self.ball.owner_id = closest_index

    def _check_ball_out_of_bounds(self):
        """
        Checks whether the ball has left the pitch area.

        Returns:
            - done (bool): True if the ball is out of bounds
            - reward (int): Reward for the current step (e.g., -1 for ball out)
            - info (dict): Additional metadata about the event
        """
        x, y = self.ball.get_position()
        if x < 0 or x > 1 or y < 0 or y > 1:
            return True, -1, {"ball_out": True}
        return False, 0, {}
