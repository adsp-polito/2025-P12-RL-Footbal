class PlayerAttacker:
    def __init__(self, shooting=0.5, passing=0.5, dribbling=0.5, speed=0.5, max_speed=10.0):
        """
        Initialize an attacking player with technical and physical attributes.

        Args:
            shooting (float): Shooting skill [0, 1]
            passing (float): Passing skill [0, 1]
            dribbling (float): Dribbling skill [0, 1]
            speed (float): Current speed factor [0, 1]
            max_speed (float): Maximum speed in m/s (real-world units)
        """
        # Technical attributes
        self.shooting = shooting
        self.passing = passing
        self.dribbling = dribbling

        # Speed is a factor of max_speed, where 1.0 means full speed
        # This allows for dynamic speed changes during the game based
        # on stamina, fatigue, or tactical decisions.
        self.speed = speed
        self.max_speed = max_speed

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

    def reset_position(self, start_x=0.0, start_y=0.0):
        """
        Reset player position to specific starting point.
        Coordinates are normalized in [0, 1].
        """
        self.position = [start_x, start_y]

    def move(self, delta_position):
        """
        Move the player by a delta in normalized coordinates.
        Used for simple manual updates (resetting, repositioning, debugging).
        It does NOT compute any speed or physics.
        """
        self.position[0] += delta_position[0]
        self.position[1] += delta_position[1]
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))

    def move_with_action(self, action, time_per_step, x_range, y_range):
        """
        Move the player based on an action and physical constraints.
        This is the method used by the environment during simulation.

        Args:
            action (np.array): Action in [-1, 1] for x and y directions.
            time_per_step (float): Time duration of one step (seconds).
            x_range (float): Physical field width (meters).
            y_range (float): Physical field height (meters).
        """
        dx = action[0] * self.speed * self.max_speed * time_per_step / x_range
        dy = action[1] * self.speed * self.max_speed * time_per_step / y_range
        self.move([dx, dy])

    def get_position(self):
        """Return current player position as a tuple (x, y), normalized in [0, 1]."""
        return tuple(self.position)

    def get_parameters(self):
        """Return player's technical attributes as dictionary (for logs, debug, stats)."""
        return {
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "ATT"

    def copy(self):
        """
        Create a deep copy of the player instance, used for rendering state snapshots.
        """
        new_player = PlayerAttacker(
            shooting=self.shooting,
            passing=self.passing,
            dribbling=self.dribbling,
            speed=self.speed,
            max_speed=self.max_speed
        )
        new_player.position = self.position.copy()
        return new_player
