import gymnasium as gym
import numpy as np

from RLEnvironment.player import Player
from RLEnvironment.team import Team
from RLEnvironment.ball import Ball

DISTANCE_TO_BALL = 0.005  # 0.60 meters if pitch is 120m wide

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

    # Constructor to initialize the environment
    def __init__(self, render_mode=None, teamAColor='red', teamBColor='blue',
                 teamAFormation='433', teamBFormation='433', 
                 initial_possessor_id=None, movement_mode="discrete"):
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

        # Define observation and action spaces
        # Each player provides (x, y, vx, vy, has_ball) → 22 × 5 = 110
        # Ball provides (x, y, vx, vy, is_free) → 5
        # Total = 115
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(115,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(12)  # 12 discrete actions for each player 

        # Rendering options
        self.render_mode = render_mode  
        self.window = None 

        # Initialize the teams with custom colors
        self._init_teams(teamAColor, teamBColor, teamAFormation, teamBFormation, movement_mode)

        # Assign initial ball possession
        self._assign_initial_possession(self.initial_possessor_id)

    def _init_teams(self, teamAColor, teamBColor, teamAFormation, teamBFormation, movement_mode):
        """Initializes both teams with 11 players each and assigns colors and sides"""
        
        # Create Team A with specified color, side, formation, and movement mode
        team_A = Team(team_id=0, color=teamAColor, side="left", formation=teamAFormation, movement_mode=movement_mode)
        
        # Create Team B with specified color, side, formation, and movement mode
        team_B = Team(team_id=1, color=teamBColor, side="right", formation=teamBFormation, movement_mode=movement_mode)
        
        # Store the teams in the environment
        self.teams = [team_A, team_B]  

        # Combine players from both teams into a single list for easier access
        self.players = team_A.players + team_B.players 


    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state before starting a new simulation"""
        
        # Reset the base class 
        super().reset(seed=seed)  
        self.current_step = 0

        # Reset ball state
        self.ball.reset()

        # Reset all players
        for player in self.players:
            player.reset()

        # Assign possession to the appropriate player
        self._assign_initial_possession(self.initial_possessor_id) 

        # Get the current observation of the environment
        observation = self._get_observation()  

        return observation, {}

    def step(self, actions):
        """
        Advances the simulation by one timestep using the given actions.
        
        Args:
            actions (list): List of actions, one per player.

        Returns:
            observation (np.ndarray): Current state representation.
            reward (float): Total reward from this step.
            done (bool): Whether the episode should terminate.
            truncated (bool): Always False (not used here).
            info (dict): Extra information about the step.
        """
        # Increment the step counter
        self.current_step += 1  

        # Apply the actions provided by the players
        self._apply_actions(actions)  

        # Update the ball's position based on player actions
        self.ball.update_position(self.players)  

        # Update which player possesses the ball
        self._update_ball_possession()  

        # Evaluate the current step for rewards and done conditions
        done, reward, info = self._evaluate_step()  

        # Get the current observation of the environment
        observation = self._get_observation()  

        return observation, reward, done, False, info  

    def _apply_actions(self, actions):
        """
        Applies the actions of each player. Each action is a tuple:
        (action_type, optional_data), e.g., ("move", direction) or ("pass", receiver_id)
        """

        # Loop through each player and their corresponding action
        for i, player in enumerate(self.players):
            
            # Get the action for the current player
            action = actions[i]  

            # Check if the action is a tuple (which it should be)
            if isinstance(action, tuple):

                # Extract the action type from the tuple
                action_type = action[0]  

                # Handle different action types

                # Move the player in the specified direction
                if action_type == "move":
                    player.step(action[1])  

                # Handle passing the ball
                elif action_type == "pass":
                    if player.has_ball:  # Only allow passing if the player has the ball
                        receiver_id = action[1] 
                        player.pass_to_id(receiver_id, self.players, self.ball)  
                        self.ball.owner_id = None  # Reset ball possession after the pass

                # Handle shooting the ball
                elif action_type == "shoot":
                    if player.has_ball:  # Only allow shooting if the player has the ball
                        player.shoot(self.ball)  
                        self.ball.owner_id = None  # Reset ball possession after shooting

                # Handle tackling the ball
                elif action_type == "tackle":
                    target = next((p for p in self.players if p.has_ball), None)  # Find the player with the ball
                    if target is not None and not player.has_ball:  # Only allow tackling if the player does not have the ball
                        # Check if the player is from the opposing team
                        if player.team_id != target.team_id:  # Check if they are on opposite teams
                            player.tackle(self.ball, target)  # Perform tackle

                else:
                    raise ValueError(f"Unknown action type: {action_type}")  # Raise an error for unknown action types

            else:
                raise ValueError(f"Invalid action format for player {i}: {action}")  # Raise an error for invalid action format

    def _update_ball_possession(self):
        """Updates ball possession if the ball is free"""
        # Check if the ball is currently free (not held by any player)
        if self.ball.owner_id is None:

            # Loop through each player to find the nearest one to the ball
            for player in self.players:
                distance = np.linalg.norm(player.position - self.ball.position)  # Calculate distance to the ball
                if distance < DISTANCE_TO_BALL:  
                    self.ball.owner_id = player.player_id  # Assign the ball to the nearest player
                    player.has_ball = True  # Mark the player as having the ball

                    # Reset ball possession for all other players
                    for other_player in self.players:
                        if other_player.player_id != player.player_id:  # Exclude the player who has the ball
                            other_player.has_ball = False  # Reset possession for other players

                    break 

    def _evaluate_step(self):
        """
        Evaluates the current step for rewards and done conditions.
        Returns:
            - done (bool): Whether the episode should end
            - total_reward (float): Accumulated reward for the step
            - info (dict): Additional debug info
        """
        total_reward = 0
        info = {}

        # Track players that went out of bounds
        players_out = []
        for player in self.players:
            if player.is_out_of_bounds():
                total_reward -= 1
                players_out.append(player.player_id)

        info["players_out"] = players_out

        # Penalize if ball is out
        ball_done, ball_reward, ball_info = self._check_ball_out_of_bounds()
        total_reward += ball_reward
        info.update(ball_info)

        # Reward for successful pass
        if hasattr(self.ball, "just_passed_by") and self.ball.just_passed_by is not None:
            last_passer = self.ball.just_passed_by
            if self.ball.owner_id is not None and self.ball.owner_id != last_passer:
                total_reward += 0.5
                info["pass_success"] = True
            else:
                total_reward -= 0.2
                info["pass_failed"] = True
            self.ball.just_passed_by = None

        # Reward or penalty for change of possession (tackle or lost control)
        if hasattr(self, "last_owner_id"):
            if self.ball.owner_id != self.last_owner_id:
                if self.ball.owner_id is not None:
                    total_reward += 0.3  # tackle succeeded
                    info["tackle_success"] = True
                else:
                    total_reward -= 0.3  # possession lost
                    info["lost_possession"] = True

        # Check for goal (right side goal area)
        x_ball, y_ball = self.ball.position
        if x_ball >= 1.0 and 0.3 <= y_ball <= 0.7:
            total_reward += 5.0
            info["goal_scored"] = True
            return True, total_reward, info

        # Save last owner for next step comparison
        self.last_owner_id = self.ball.owner_id

        return ball_done, total_reward, info 

    def _get_observation(self):
        """
        Builds a flattened observation vector for the entire environment.

        Each player contributes:
            - x, y: normalized position
            - vx, vy: current velocity (delta position over 1 frame)
            - has_ball: 1 if the player possesses the ball, else 0

        The ball contributes:
            - x, y: position
            - vx, vy: velocity
            - is_free: 1 if no player possesses the ball

        Returns:
            - obs (np.ndarray): Flattened observation of shape (115,)
        """

        player_features = []

        # Extract features from each player
        for player in self.players:
            x, y = player.position
            vx, vy = player.velocity
            has_ball = 1.0 if player.has_ball else 0.0
            player_features.extend([x, y, vx, vy, has_ball])

        # Extract ball features
        x_ball, y_ball = self.ball.position
        vx_ball, vy_ball = self.ball.velocity
        is_free = 1.0 if self.ball.owner_id is None else 0.0
        ball_features = [x_ball, y_ball, vx_ball, vy_ball, is_free]

        # Concatenate all features and return as flat numpy array 
        # The observation shape is (22 players * 5 features + 1 ball * 5 features = 115)
        obs = np.array(player_features + ball_features, dtype=np.float32)

        return obs
    
    def render(self):
        """
        Renders the current environment frame.

        Currently a placeholder that prints the current frame number.
        Override this method to enable graphical simulation or visual debugging.
        """
        if self.render_mode == "human":  
            print("Rendering frame", self.current_step)  

    def close(self):
        """
        Properly closes any rendering-related resources or windows if used.
        """
        if self.window:  # Check if there's an open rendering window
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
        # Check if an initial player ID is provided, if so, assign the ball to that player
        if initial_possessor_id is not None:  
            self.players[initial_possessor_id].has_ball = True  
            self.ball.owner_id = initial_possessor_id 

        # If no initial player ID is provided, find the closest player in Team A to the ball
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
        # Check if the ball is out of bounds and return appropriate values
        if self.ball.is_out_of_bounds():  
            return True, -1, {"ball_out": True}
        return False, 0, {}
