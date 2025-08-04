"""
Single source-of-truth for global dimensions.
Edit these numbers and all other files will be updated accordingly.
"""

# Field geometry (metres)
FIELD_WIDTH   = 120
FIELD_HEIGHT  = 80
MARGIN        = 5       # extra border around the rectangle

# Pitch settings
X_MIN = -MARGIN
X_MAX = FIELD_WIDTH + MARGIN
Y_MIN = -MARGIN
Y_MAX = FIELD_HEIGHT + MARGIN

# Grid
CELL_SIZE     = 1
STRIPE_WIDTH  = 5       # for rendering stripes

# Areas
PENALTY_DEPTH        = 18
PENALTY_HEIGHT       = 44
GOAL_AREA_DEPTH      = 6
GOAL_AREA_HEIGHT     = 20
CENTER_CIRCLE_RADIUS = 10
PENALTY_SPOT_X_LEFT  = 11
PENALTY_SPOT_X_RIGHT = 109
GOAL_WIDTH          = 7.32
GOAL_DEPTH           = 2.44
