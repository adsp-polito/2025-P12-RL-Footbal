import numpy as np
from RLmodel.player import Player
from lineups.lineupsDict import formations

class Team:
    """
    The Team class manages a group of 11 football players, including their initialization
    from predefined tactical formations stored in the formations dictionary (lineupsDict.py)

    Attributes:
        - team_id (int): Identifier for the team (0 = home, 1 = away)
        - color (str): Color for rendering purposes
        - side (str): Field side ('left' or 'right') to mirror positions
        - formation (str): Chosen tactical formation (e.g., '433')
        - players (list): List of Player instances belonging to this team
    """

    def __init__(self, team_id, color, side, formation="433"):
        self.team_id = team_id
        self.color = color
        self.side = side
        self.formation = formation
        self.players = []

        # Load formation data from dictionary
        self.roles_data = self._load_formation_data()

        # Build players from role-position data
        for i, role_data in enumerate(self.roles_data):
            role = role_data["role"]
            x = role_data["x"] / 120  # normalize x from meters (field width = 120)
            y = role_data["y"] / 80   # normalize y from meters (field height = 80)
            
            

            position = np.array([1.0 - x, y]) if self.side == "right" else np.array([x, y])

            self.players.append(Player(
                player_id=i,
                team_id=team_id,
                role=role,
                init_position=position
            ))

    def _load_formation_data(self):
        """
        Retrieves the role and position data for the current formation from the dictionary
        Raises an error if the formation is not defined
        Returns:
            list[dict]: List of role dictionaries with role name, abbreviation, and position
        """
        if self.formation not in formations:
            raise ValueError(f"Formation {self.formation} not found in formations dictionary.")
        return formations[self.formation]

# -----------------------------
# Future GUI integration notes:
# -----------------------------
# In future implementations, the application will allow the coach to select
# a formation (e.g., 433, 442) directly from the interface. Once selected,
# the system will retrieve the correct sequence of roles and absolute positions
# from lineupsDict.py. These values will be normalized and optionally overridden
# via drag-and-drop interaction to place players freely while maintaining tactical coherence.
# ---------------------------------------------------------------------------
