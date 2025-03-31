import gymnasium as gym
import numpy as np

from RLmodel.player import Player
from RLmodel.team import Team
from RLmodel.ball import Ball

class FootballEnv(gym.Env):
    """
    This class define a football simulation environment designed for reinforcement learning
    simulations. It models a 11 vs 11 football match scenario with a 2D pitch, 2 teams,
    and a ball. Each player can be controlled either by a learned agent or a rule-based system

    The environment connects to the Gymnasium API, supporting methods like reset(), 
    step(), render(), and close()

    Attributes:
        - field_width (int): Width of the football pitch in meters
        - field_height (int): Height of the football pitch in meters
        - players (list): List of all 22 player instances (11 per team)
        - teams (list): List of the two team instances
        - ball (Ball): Instance of the ball object
        - current_step (int): Step counter for the simulation
        - observation_space (gym.Space): Defines the format and bounds of environment observations
        - action_space (gym.Space): Defines the format and range of actions available to agents
        - render_mode (str): Mode for rendering ('human' or None)
        - window (object): Optional window for graphical rendering
    """

    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(self, render_mode=None, teamAColor='red', teamBColor='blue'):
        super().__init__()

        # Field dimensions
        # The measures are aligned with the StatsBomb dataset
        self.field_width = 120
        self.field_height = 80

        # Initialize lists for players and teams
        self.players = []
        self.teams = []

        # Create the ball instance
        self.ball = Ball()

        # Step counter to keep track of simulation time
        self.current_step = 0

        # (Placeholder) Observation space:
        # defines the shape and bounds of what each agent perceives
        # here, it's a placeholder representing 22 players each with 10 numerical features
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(22, 10), dtype=np.float32
        )

        # (Placeholder) Action space:
        # defines the set of possible actions the agent(s) can take
        # for the moment it defines 11 discrete actions (one per player), each with 8 possible choices
        self.action_space = gym.spaces.MultiDiscrete([8 for _ in range(11)])

        # Rendering options for visual output
        self.render_mode = render_mode
        self.window = None

        # Initialize the two teams with customizable colors
        self._init_teams(teamAColor, teamBColor)

    def _init_teams(self, teamAColor = 'red', teamBColor = 'blue'):
        """Initializes both teams with 11 players each and assigns colors and sides"""
        team_A = Team(team_id=0, color=teamAColor, side="left")
        team_B = Team(team_id=1, color=teamBColor, side="right")
        self.teams = [team_A, team_B]
        self.players = team_A.players + team_B.players

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state before starting a new simulation"""
        super().reset(seed=seed)
        self.current_step = 0

        # Reset ball position and status
        self.ball.reset()

        # Reset all players to initial conditions
        for player in self.players:
            player.reset()

        # Return the initial observation state of the environment
        observation = self._get_observation()
        return observation, {}

    def step(self, actions):
        """
        Advances the environment by one step using the given actions

        Args:
            actions (list): A list of actions, one per player

        Returns:
            - observation: Updated observation after applying actions
            - reward: Numerical feedback signal (to be defined)
            - done: Boolean indicating if the episode has ended
            - truncated: Placeholder (False)
            - info: Dictionary with additional metadata
        """
        # Increase simulation step counter
        self.current_step += 1

        # Apply each player's action to update their state
        for i, player in enumerate(self.players):
            player.step(actions[i])

        # Update ball position and dynamics based on physics and interactions
        self.ball.update()

        # Compute placeholder reward and end condition
        reward = 0
        done = False

        # Get new observation after the environment update
        observation = self._get_observation()
        info = {}

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
