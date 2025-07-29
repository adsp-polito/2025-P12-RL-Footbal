class PlayerGoalkeeper:
    def __init__(self, reflexes=0.5, diving=0.5, positioning=0.5, speed=0.5, max_speed=10.0):
        """
        Initialize a goalkeeper with technical and physical attributes.

        Args:
            reflexes (float): Reflexes skill [0, 1]
            diving (float): Diving skill [0, 1]
            positioning (float): Positioning skill [0, 1]
            speed (float): Current speed factor [0, 1]
            max_speed (float): Maximum speed in m/s (real-world units)
        """
        # Technical attributes
        self.reflexes = reflexes
        self.diving = diving
        self.positioning = positioning

        # Speed is a factor of max_speed, where 1.0 means full speed
        # This allows for dynamic speed changes during the game based
        # on stamina, fatigue, or tactical decisions.
        self.speed = speed
        self.max_speed = max_speed

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

    def reset_position(self, position):
        """
        Reset player position using a list of two normalized coordinates.

        Args:
            position (list): [x, y] in normalized coordinates.
        """
        if isinstance(position, list) and len(position) == 2:
            self.position = position
        else:
            raise ValueError("Position must be a list of two elements [x, y].")

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
        Move the goalkeeper based on an action and physical constraints.
        This method is used during simulation.

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
        """Return current goalkeeper position as (x, y), normalized in [0, 1]."""
        return tuple(self.position)

    def get_parameters(self):
        """Return goalkeeper's technical attributes as dictionary (for logs, debug, stats)."""
        return {
            "reflexes": self.reflexes,
            "diving": self.diving,
            "positioning": self.positioning
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "GK"

    def copy(self):
        """
        Create a deep copy of the goalkeeper instance.
        This is used to store the current state during rendering.
        """
        new_player = PlayerGoalkeeper(
            reflexes=self.reflexes,
            diving=self.diving,
            positioning=self.positioning,
            speed=self.speed,
            max_speed=self.max_speed
        )
        new_player.position = self.position.copy()
        return new_player
