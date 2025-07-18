class PlayerAttacker:
    def __init__(self, shooting=0.5, passing=0.5, dribbling=0.5, speed=0.5):
        # Technical attributes, normalized between 0 and 1
        self.shooting = shooting
        self.passing = passing
        self.dribbling = dribbling
        self.speed = speed

        # Player position on the field, normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

    def reset_position(self, start_x=0.0, start_y=0.0):
        # Reset player position to the starting point
        self.position = [start_x, start_y]


    def move(self, delta_position):
        # Move the player by a delta position, ensuring it stays within bounds
        self.position[0] += delta_position[0]
        self.position[1] += delta_position[1]
        
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))



    def get_position(self):
        # Return the current player position as a tuple (x, y)
        return tuple(self.position)

    def get_parameters(self):
        # Return the technical parameters as a dictionary
        return {
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling
        }

    def get_role(self):
        # Return the player's role as a string
        return "ATT"
