import numpy as np

class PlayerAttacker:
    def __init__(self, shooting=0.5, passing=0.5, dribbling=0.5, speed=0.5, max_speed=10.0, precision=0.7, max_power=15.0):
        """
        Initialize an attacking player with technical and physical attributes.

        Args:
            shooting (float): Shooting skill [0, 1], overall shot quality and effectiveness.
            passing (float): Passing skill [0, 1].
            dribbling (float): Dribbling skill [0, 1].
            speed (float): Current speed factor [0, 1], multiplier for max_speed.
            max_speed (float): Maximum running speed in meters per second.
            precision (float): Shot accuracy [0, 1], how accurate the shot direction is.
            max_power (float): Maximum shot power in meters per second (ball velocity).
        """
        # Technical attributes
        self.shooting = shooting      # Ability to successfully score shots
        self.passing = passing
        self.dribbling = dribbling
        self.precision = precision    # Accuracy of shots, affects shot direction noise

        # Speed attributes
        self.speed = speed            # Current speed factor (dynamic, e.g., fatigue)
        self.max_speed = max_speed    # Physical max running speed (m/s)

        # Shooting power
        self.max_power = max_power    # Max velocity player can impart to the ball

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

    def reset_position(self, start_x=0.0, start_y=0.0):
        """
        Reset player position to a specific starting point.
        Coordinates are normalized between 0 and 1.
        """
        self.position = [start_x, start_y]

    def move(self, delta_position):
        """
        Move the player by a delta in normalized coordinates.
        Used for manual updates such as repositioning or debugging.
        No physics or speed constraints applied here.
        """
        self.position[0] += delta_position[0]
        self.position[1] += delta_position[1]
        # Clamp position inside pitch boundaries
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))

    def move_with_action(self, action, time_per_step, x_range, y_range):
        """
        Move player based on continuous action input and physical constraints.

        Args:
            action (np.array): Array with x and y movement values in [-1, 1].
            time_per_step (float): Duration of simulation step (seconds).
            x_range (float): Width of pitch in meters.
            y_range (float): Height of pitch in meters.
        """
        dx = action[0] * self.speed * self.max_speed * time_per_step / x_range
        dy = action[1] * self.speed * self.max_speed * time_per_step / y_range
        self.move([dx, dy])

    def shoot(self, desired_direction, desired_power):
        """
        Calculate shot parameters based on agent's desired input,
        adjusted by player's precision and shooting skill to simulate uncertainty.

        Args:
            desired_direction (np.array): 2D vector (not necessarily normalized) indicating desired shot direction.
            desired_power (float): Desired shot power in [0, 1].

        Returns:
            shot_quality (float): Estimated probability of successful shot (0 to 1).
            shot_direction (np.array): Actual 2D unit vector shot direction after applying noise.
            shot_power (float): Actual shot power scaled by max_power and skill.
        """
        # Normalize desired direction to unit vector
        norm = np.linalg.norm(desired_direction)
        if norm > 0:
            desired_dir_norm = desired_direction / norm
        else:
            # Default to straight towards goal center if zero vector
            desired_dir_norm = np.array([1.0, 0.0])

        # Add noise to direction proportional to (1 - precision)
        max_noise_radians = 0.1  # max ~5.7 degrees noise
        noise_strength = (1 - self.precision) * max_noise_radians
        angle_noise = np.random.uniform(-noise_strength, noise_strength)
        cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)

        # Rotate desired direction by noise angle
        shot_direction = np.array([
            cos_a * desired_dir_norm[0] - sin_a * desired_dir_norm[1],
            sin_a * desired_dir_norm[0] + cos_a * desired_dir_norm[1]
        ])

        # Clip desired_power to [0,1]
        desired_power = np.clip(desired_power, 0.0, 1.0)

        # Scale shot power by max_power and modulate by shooting skill
        # We add a minimum base power of 0.5 * max_power for realism
        shot_power = self.max_power * (0.5 + 0.5 * self.shooting * desired_power)

        # Calculate shot quality based on player's shooting skill and position on pitch
        x_pos, y_pos = self.position

        x_factor = 0.2 + 0.8 * x_pos
        y_dist = abs(y_pos - 0.5)
        y_factor = max(0.5, 1 - 2 * y_dist)

        # Shot quality is affected by position and skill (does not depend on power here)
        shot_quality = x_factor * y_factor * self.shooting

        return shot_quality, shot_direction, shot_power



    def get_position(self):
        """Return current position as normalized (x, y) tuple."""
        return tuple(self.position)

    def get_parameters(self):
        """Return technical attributes for logging or debugging."""
        return {
            "shooting": self.shooting,
            "precision": self.precision,
            "passing": self.passing,
            "dribbling": self.dribbling
        }

    def get_role(self):
        """Return the player role name as string."""
        return "ATT"

    def copy(self):
        """
        Create a deep copy of the player instance.
        Useful for rendering snapshots without side effects.
        """
        new_player = PlayerAttacker(
            shooting=self.shooting,
            passing=self.passing,
            dribbling=self.dribbling,
            speed=self.speed,
            max_speed=self.max_speed,
            precision=self.precision,
            max_power=self.max_power
        )
        new_player.position = self.position.copy()
        return new_player