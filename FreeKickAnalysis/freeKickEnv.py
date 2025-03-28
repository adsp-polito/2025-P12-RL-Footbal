import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from drawPitch import draw_pitch
import random

PITCH_LENGTH = 120
PITCH_WIDTH = 80

class FreeKickEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating a football free kick scenario.
    Attacker attempts to score, defenders block, ball can be passed, dribbled or shot.
    """

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # 3 attackers + 3 defenders + 1 ball = 7 entities × 2D positions
        obs_dim = (3 + 3 + 1) * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Actions: 0 = pass to A1, 1 = pass to A2, 2 = shoot, 3 = dribble left, 4 = dribble right, 5 = dribble up, 6 = dribble down
        self.action_space = spaces.Discrete(7)

        # Max steps per episode
        self.max_steps = 10
        self.current_step = 0

        self.reset()

    def _init_positions(self):
        """
        Initialize player and ball positions for a realistic free kick.
        """
        # ATTACKERS (blue)
        self.attackers = np.array([
            [0.74, 0.50],  # A0 - kicker
            [0.90, 0.65],  # A1 - forward left
            [0.90, 0.35]   # A2 - forward right
        ])

        # DEFENDERS (red)
        self.defenders = np.array([
            [0.826, 0.49],  # D0 - wall left
            [0.826, 0.51],  # D1 - wall right
            [0.98, 0.50]    # D2 - goalkeeper
        ])

        # BALL
        self.ball = np.array([0.75, 0.50])  # in front of A0
        self.ball_owner = 0  # A0 starts with the ball

    def _get_obs(self):
        return np.concatenate([
            self.attackers.flatten(),
            self.defenders.flatten(),
            self.ball.flatten()
        ]).astype(np.float32)

    def _check_goal(self):
        """
        Return True if the ball is inside the goal area (right side of pitch).
        """
        x, y = self.ball[0] * PITCH_LENGTH, self.ball[1] * PITCH_WIDTH
        return x >= 120 and (40 - 3.66) <= y <= (40 + 3.66)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._init_positions()
        return self._get_obs(), {}

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random

PITCH_LENGTH = 120
PITCH_WIDTH = 80

class FreeKickEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating a football free kick scenario.
    """
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # 3 attackers + 3 defenders + 1 ball = 7 entities × 2D positions
        obs_dim = (3 + 3 + 1) * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Actions: 0 = pass to A1, 1 = pass to A2, 2 = shoot, 3 = dribble left, 4 = dribble right, 5 = dribble up, 6 = dribble down
        self.action_space = spaces.Discrete(7)

        # Max steps per episode
        self.max_steps = 10
        self.current_step = 0

        self.reset()

    def _init_positions(self):
        """
        Initialize player and ball positions for a realistic free kick.
        """
        # ATTACKERS (blue)
        self.attackers = np.array([
            [0.74, 0.50],  # A0 - kicker
            [0.90, 0.65],  # A1 - forward left
            [0.90, 0.35]   # A2 - forward right
        ])

        # DEFENDERS (red)
        self.defenders = np.array([
            [0.826, 0.49],  # D0 - wall left
            [0.826, 0.51],  # D1 - wall right
            [0.98, 0.50]    # D2 - goalkeeper
        ])

        # BALL
        self.ball = np.array([0.75, 0.50])  # in front of A0
        self.ball_owner = 0

    def _get_obs(self):
        return np.concatenate([
            self.attackers.flatten(),
            self.defenders.flatten(),
            self.ball.flatten()
        ]).astype(np.float32)

    def _check_goal(self):
        """
        Return True if the ball is inside the goal area (right side of pitch).
        """
        x, y = self.ball[0] * PITCH_LENGTH, self.ball[1] * PITCH_WIDTH
        return x >= 120 and (40 - 3.66) <= y <= (40 + 3.66)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._init_positions()
        return self._get_obs(), {}

    def step(self, actions):
        """
        Applies an action to the environment and returns the new state, reward, and termination flag.
        """
        self.current_step += 1
        done = False
        rewards = np.zeros(6)  # Reward for each player (A0, A1, A2, D0, D1, D2)
        total_team_reward = 0  # Total team reward (sum of individual rewards)

        # If it's the first step (the free kick)
        if self.current_step == 1:
            for player_id in range(3):  # 3 attackers (A0, A1, A2)
                action = actions[player_id]  # Action for each player

                if player_id == 0:  # A0, the free kick taker
                    if self.ball_owner == 0:
                        if action == 0:  # Pass to A1
                            self.ball_owner = 1
                            self.ball = self.attackers[1].copy()
                            rewards[player_id] += 0.1
                        elif action == 1:  # Pass to A2
                            self.ball_owner = 2
                            self.ball = self.attackers[2].copy()
                            rewards[player_id] += 0.1
                        elif action == 2:  # Shoot
                            shot_power = random.uniform(0.01, 1.0)  # Shot power
                            shot_angle = random.uniform(-0.1, 0.1)  # Shot angle
                            self.ball[0] += shot_power * 25
                            self.ball[1] += shot_angle * shot_power

                            if self._check_goal():  # Goal scored
                                rewards[player_id] += 5.0
                            else:  # Missed goal
                                rewards[player_id] -= 0.2
                            done = True
                    else:
                        rewards[player_id] = -10  # Penalization if A0 does an INVALID ACTION
                        done = True
                else:  # Actions for A1 and A2 (the other attackers)
                    if self.ball_owner == player_id:  # If the player has the ball
                        if action == 3:  # Dribble left
                            self.attackers[player_id][0] -= 0.02
                        elif action == 4:  # Dribble right
                            self.attackers[player_id][0] += 0.02
                        elif action == 5:  # Dribble up
                            self.attackers[player_id][1] += 0.02
                        elif action == 6:  # Dribble down
                            self.attackers[player_id][1] -= 0.02
                        elif action == 0:  # Pass to A1
                            if player_id != 1:  # Can't pass to self
                                self.ball_owner = 1
                                self.ball = self.attackers[1].copy()
                                rewards[player_id] += 0.1
                        elif action == 1:  # Pass to A2
                            if player_id != 2:  # Can't pass to self
                                self.ball_owner = 2
                                self.ball = self.attackers[2].copy()
                                rewards[player_id] += 0.1
                    else:
                        # Penalization if a player without the ball tries to pass or dribble
                        if action in [0, 1, 2, 3, 4, 5, 6]:  # Invalid actions without the ball
                            rewards[player_id] = -10
                            done = True

        else:  # After the first step, all players can do anything
            for player_id in range(6):  # 6 players in total
                action = actions[player_id]

                # If player has the ball, they can pass, shoot, or dribble
                if self.ball_owner == player_id:
                    if action == 0:  # Pass to A1
                        if player_id != 1:  # Prevent passing to self
                            self.ball_owner = 1
                            self.ball = self.attackers[1].copy()
                            rewards[player_id] += 0.1
                    elif action == 1:  # Pass to A2
                        if player_id != 2:  # Prevent passing to self
                            self.ball_owner = 2
                            self.ball = self.attackers[2].copy()
                            rewards[player_id] += 0.1
                    elif action == 2:  # Shoot
                        shot_power = random.uniform(0.01, 1.0)
                        shot_angle = random.uniform(-0.1, 0.1)

                        self.ball[0] += shot_power * 25
                        self.ball[1] += shot_angle * shot_power

                        if self._check_goal():  # Goal
                            rewards[player_id] += 5.0
                        else:  # Missed goal
                            rewards[player_id] -= 0.2
                        done = True
                    elif action == 3:  # Dribble left
                        self.attackers[player_id][0] -= 0.02
                    elif action == 4:  # Dribble right
                        self.attackers[player_id][0] += 0.02
                    elif action == 5:  # Dribble up
                        self.attackers[player_id][1] += 0.02
                    elif action == 6:  # Dribble down
                        self.attackers[player_id][1] -= 0.02
                else:
                    # Allow players without the ball to move (change position)
                    if action == 3:  # Move left
                        self.attackers[player_id][0] -= 0.02
                    elif action == 4:  # Move right
                        self.attackers[player_id][0] += 0.02
                    elif action == 5:  # Move up
                        self.attackers[player_id][1] += 0.02
                    elif action == 6:  # Move down
                        self.attackers[player_id][1] -= 0.02

                    # Reward players for moving without the ball
                    rewards[player_id] += 0.1


            # Actions for defenders (D0, D1, D2), they can only move (change position)
            for player_id in range(3, 6):  # 3 defenders (D0, D1, D2)
                action = actions[player_id]

                # Defenders can only move (dribble or change position)
                if action == 3:  # Move left
                    self.defenders[player_id - 3][0] -= 0.02
                elif action == 4:  # Move right
                    self.defenders[player_id - 3][0] += 0.02
                elif action == 5:  # Move up
                    self.defenders[player_id - 3][1] += 0.02
                elif action == 6:  # Move down
                    self.defenders[player_id - 3][1] -= 0.02

                # Reward players for moving without the ball
                rewards[player_id] += 0.1

            # Prevent teleportation of the ball
            self.attackers[self.ball_owner][0] = np.clip(self.attackers[self.ball_owner][0], 0.0, 1.0)
            self.attackers[self.ball_owner][1] = np.clip(self.attackers[self.ball_owner][1], 0.0, 1.0)

            # Ball always follows the player unless shoot or pass
            if action not in [0, 1, 2]:  # Skip when pass or shoot
                self.ball = self.attackers[self.ball_owner].copy()

            # End episode after too many steps
            if self.current_step >= self.max_steps:
                done = True

        # If goal is scored, reward all attackers, penalty all defenders
        if self._check_goal():
            # Reward all attackers
            for player_id in range(3):  # A0, A1, A2
                rewards[player_id] += 2.5  # Reward for attackers when a goal is scored
        else:
            # Penalize defenders if a goal is conceded
            for player_id in range(3, 6):  # D0, D1, D2
                rewards[player_id] -= 5.0  # Penalty for defenders if goal is conceded

        # Calculate total reward for the team
        total_team_reward = np.sum(rewards)  # Sum of individual rewards for the team

        # Return observation, total reward, done flag, and individual rewards
        return self._get_obs(), total_team_reward, done, False, {"individual_rewards": rewards}

    def render(self):
        """
        Draw the current game state on the pitch.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax=ax)

        def to_field(pos):
            return pos[0] * PITCH_LENGTH, pos[1] * PITCH_WIDTH

        # Plot attackers
        for i, player in enumerate(self.attackers):
            x, y = to_field(player)
            ax.plot(x, y, 'bo', markersize=10)
            ax.text(x, y - 2, f"A{i}", color='white', ha='center')

        # Plot defenders
        for i, player in enumerate(self.defenders):
            x, y = to_field(player)
            ax.plot(x, y, 'ro', markersize=10)
            ax.text(x, y - 2, f"D{i}", color='white', ha='center')

        # Plot ball
        x, y = to_field(self.ball)
        ax.plot(x, y, 'yo', markersize=8)

        # Ball owner highlight
        owner_x, owner_y = to_field(self.attackers[self.ball_owner])
        ax.plot(owner_x, owner_y, 'o', markersize=16, markerfacecolor='none', markeredgecolor='yellow', linewidth=1.5)

        plt.title("Free Kick Simulation")
        plt.show()