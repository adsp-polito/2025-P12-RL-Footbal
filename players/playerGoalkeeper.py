class PlayerGoalkeeper:
    def __init__(self, reflexes=0.5, diving=0.5, positioning=0.5, speed=0.5):
        # Technical attributes, normalized between 0 and 1
        self.reflexes = reflexes
        self.diving = diving
        self.positioning = positioning
        self.speed = speed

        # Player position on the field, normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

    def reset_position(self, start_x=0.0, start_y=0.0):
        # Reset player position to the starting point
        self.position = [start_x, start_y]

    def move(self, direction, speed):
        # Combine player's intrinsic speed with input speed
        effective_speed = speed * self.speed

        self.position[0] += direction[0] * effective_speed
        self.position[1] += direction[1] * effective_speed

        # Clamp position within the field limits [0, 1]
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))

    def get_position(self):
        # Return the current player position as a tuple (x, y)
        return tuple(self.position)

    def get_parameters(self):
        # Return the technical parameters as a dictionary
        return {
            "reflexes": self.reflexes,
            "diving": self.diving,
            "positioning": self.positioning
        }

    def get_role(self):
        # Return the player's role as a string
        return "GK"
