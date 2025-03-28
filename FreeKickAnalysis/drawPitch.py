import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from utils import PITCH_LENGTH, PITCH_WIDTH 

def draw_pitch(ax=None, field_color='green'):
    """
    Draw a 120 x 80 football pitch including realistic goalposts.
    (0,0) is top-left, (120,80) is bottom-right.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set pitch background
    if field_color == 'green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke'  # line color
        border_color = 'white'
    else:
        lc = 'black'
        border_color = 'white'
    
    # Outer boundaries
    ax.plot([0,0],[0,80], color=border_color, linewidth=3)
    ax.plot([120,120],[0,80], color=border_color, linewidth=3)
    ax.plot([0,120],[0,0], color=border_color, linewidth=3)
    ax.plot([0,120],[80,80], color=border_color, linewidth=3)
    
    # Center line
    ax.plot([60,60],[0,80], color=lc, linewidth=2)
    
    # Center circle
    center_circle = plt.Circle((60, 40), 10, color=lc, fill=False, linewidth=2)
    ax.add_patch(center_circle)
    
    # Penalty areas and arcs (both sides)
    for x in [0, 120]:
        penalty_x = 18 if x == 0 else 102
        goal_x = 6 if x == 0 else 114
        penalty_spot = 11 if x == 0 else 109
        arc_center = (12, 40) if x == 0 else (108.2, 40)
        arc_angle = 0 if x == 0 else 180
        
        # Penalty area
        ax.plot([x, penalty_x], [18, 18], color=lc, linewidth=2)
        ax.plot([penalty_x, penalty_x], [18, 62], color=lc, linewidth=2)
        ax.plot([x, penalty_x], [62, 62], color=lc, linewidth=2)
        
        # Goal area
        ax.plot([x, goal_x], [30, 30], color=lc, linewidth=2)
        ax.plot([goal_x, goal_x], [30, 50], color=lc, linewidth=2)
        ax.plot([x, goal_x], [50, 50], color=lc, linewidth=2)
        
        # Penalty spot
        ax.plot([penalty_spot], [40], marker='o', markersize=5, color=lc)

        # Penalty arc
        penalty_arc = Arc(
            arc_center, 20, 20, angle=arc_angle, 
            theta1=308, theta2=52, color=lc, linewidth=2
        )
        ax.add_patch(penalty_arc)
        
    # --- Draw goals as rectangles ---
    goal_width = 7.32  # in meters (FIFA standard)
    goal_height = 2.44  # in meters (FIFA standard)
    post_size = 0.6  # visual radius of post markers (optional)

    # Goal position parameters
    goal_y1 = 40 - 3.66
    goal_y2 = 40 + 3.66

    # Left goal (draw as simple rectangle)
    left_goal = plt.Rectangle(
        (0, goal_y1), 
        -goal_height,
        goal_width, 
        linewidth=2, 
        edgecolor=lc, 
        facecolor='none', 
        zorder=3
    )
    ax.add_patch(left_goal)

    # Right goal (draw as simple rectangle)
    right_goal = plt.Rectangle(
        (120, goal_y1), 
        goal_height, 
        goal_width,
        linewidth=2, 
        edgecolor=lc, 
        facecolor='none', 
        zorder=3
    )
    ax.add_patch(right_goal)

    # Corners
    corner_radius = 2

    # Define the positions for the arcs at the four corners and the angles
    corner_positions = [(0, 0), (120, 0), (0, 80), (120, 80)]
    corner_angles = [0, 90, 270, 180]  # Angles for each corner

    # Draw arcs instead of full circles
    for (x, y), angle in zip(corner_positions, corner_angles):
        corner_arc = Arc(
            (x, y),  # center of the arc (corner)
            corner_radius * 2,  # width of the arc
            corner_radius * 2,  # height of the arc
            angle=0,  # no rotation of the arc itself
            theta1=0,  # start angle (0 degrees)
            theta2=90,  # end angle (90 degrees)
            color=lc,  # arc color (same as line color)
            linewidth=2  # line width for the arc
        )
        # Rotate the arc according to the specific corner's angle
        corner_arc.set_angle(angle)
        ax.add_patch(corner_arc)  

    # Axis config
    ax.set_xlim(-5, 125)  # allow space for goal depth
    ax.set_ylim(85, -5)    # flip y-axis
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return ax