class PlayerDefender:
    def __init__(self, tackling=0.5, marking=0.5, anticipation=0.5, speed=0.5, max_speed=10.0):
        """
        Initialize a defending player with technical and physical attributes.

        Args:
            tackling (float): Tackling skill [0, 1]
            marking (float): Marking skill [0, 1]
            anticipation (float): Anticipation skill [0, 1]
            speed (float): Current speed factor [0, 1]
            max_speed (float): Maximum speed in m/s (real-world units)
        """
        self.tackling = tackling
        self.marking = marking
        self.anticipation = anticipation
        self.speed = speed
        self.max_speed = max_speed

        # Player position in normalized coordinates [0, 1]
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

    def move_with_action(self, direction, time_per_step, x_range, y_range):
        """
        Move the player towards a direction with physical constraints.
        This is the method used by the environment during simulation.

        Args:
            direction (np.array): Normalized direction towards a target (unit vector).
            time_per_step (float): Time duration of one step (seconds).
            x_range (float): Physical field width (meters).
            y_range (float): Physical field height (meters).
        """
        dx = direction[0] * self.speed * self.max_speed * time_per_step / x_range
        dy = direction[1] * self.speed * self.max_speed * time_per_step / y_range
        self.move([dx, dy])

    def get_position(self):
        """Return current player position as (x, y), normalized in [0, 1]."""
        return tuple(self.position)

    def get_parameters(self):
        """Return player's technical attributes as dictionary (for logs, debug, stats)."""
        return {
            "tackling": self.tackling,
            "marking": self.marking,
            "anticipation": self.anticipation
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "DEF"

    def copy(self):
        """
        Create a deep copy of the player instance, used for rendering state snapshots.
        """
        new_player = PlayerDefender(
            tackling=self.tackling,
            marking=self.marking,
            anticipation=self.anticipation,
            speed=self.speed,
            max_speed=self.max_speed
        )
        new_player.position = self.position.copy()
        return new_player
