# utils.py

PITCH_LENGTH = 120
PITCH_WIDTH = 80

ACTION_NAMES = [
    "Pass to A1",
    "Pass to A2",
    "Shoot",
    "Dribble Left",
    "Dribble Right",
    "Dribble Up",
    "Dribble Down"
]

def to_field(pos):
    """
    Convert normalized position (x, y) to field coordinates.
    
    Args:
    - pos: (x, y) tuple with normalized positions (0 to 1).
    
    Returns:
    - Tuple with actual field coordinates (in meters).
    """
    return pos[0] * PITCH_LENGTH, pos[1] * PITCH_WIDTH