import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import numpy as np
from helpers.drawPitch import draw_pitch

# Create a yellow highlight ring to indicate the current ball possessor
def highlight_possessor(ax):
    """
    Creates and returns a visual ring to highlight the player currently in possession of the ball

    Args:
        - ax (matplotlib.axes): Axis where the pitch and elements are drawn

    Returns:
        - Circle: A yellow circle that will be repositioned around the ball possessor each frame
    """
    circle = Circle((-10, -10), radius=1.0, edgecolor='yellow', facecolor='none', linewidth=6, zorder=4)
    ax.add_patch(circle)
    return circle

# Set up the initial pitch and visual elements for simulation (players, ball, labels)
def setup_pitch(env):
    """
    Initializes the pitch, players, ball, labels and title for the visual environment

    Args:
        - env (FootballEnv): The football environment containing players, teams, and the ball

    Returns:
        Tuple containing:
            - fig, ax: Matplotlib figure and axis
            - player_dots: List of graphical player markers
            - role_labels: List of text labels for player roles/abbreviations
            - ball_circle: Circle patch representing the ball
            - highlight_circle: Circle to highlight the ball possessor
            - possessor_id_labels: List of text labels placed on player dots
            - title: Title object to display time/frame info
    """
    # Create a new figure and axis for the pitch
    fig, ax = plt.subplots(figsize=(12, 8))  
    draw_pitch(ax) 

    # Set team colors based on the teams
    team_colors = [env.teams[0].color] * 11 + [env.teams[1].color] * 11  
    player_dots, role_labels, possessor_id_labels = [], [], []  # Initialize lists for players and labels

    # Create player markers and labels
    for i, player in enumerate(env.players):
        pos = player.get_position() * np.array([120, 80])  
        dot, = ax.plot(pos[0], pos[1], 'o', markersize=10, color=team_colors[i], markeredgecolor="black", zorder=5)  
        
        # Create role label
        label = ax.text(pos[0], pos[1] - 2, getattr(player, 'abbr', player.role[:2]),  
                        ha='center', va='center', fontsize=8, color='black', fontweight='bold', zorder=6)
        
        # Create player ID label
        pid_label = ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=6, color='white', fontweight='bold', zorder=7) 
        
        # Add player dot and labels to lists
        player_dots.append(dot) 
        role_labels.append(label)  
        possessor_id_labels.append(pid_label)

    # Create ball circle and set its position
    ball_pos = env.ball.get_position() * np.array([120, 80])  
    ball_circle = Circle(ball_pos, radius=0.65, facecolor='white', edgecolor='black', linewidth=1.5, zorder=5) 
    ax.add_patch(ball_circle)  

    # Create highlight circle for ball possessor
    highlight_circle = highlight_possessor(ax) 

    # Create title for the plot
    title = ax.set_title("", fontsize=18, fontweight="bold", color='black', pad=15)  

    return fig, ax, player_dots, role_labels, ball_circle, highlight_circle, possessor_id_labels, title

# Update the entire visual scene at a specific simulation frame
def update_frame_visuals(env, frame_data, player_dots, role_labels, 
                         ball_circle, highlight_circle, possessor_id_labels, 
                         title, num_frames):
    """
    Updates all visual elements on the pitch for a specific frame

    Args:
        - env (FootballEnv): The football environment.
        - frame_data (dict): Contains player positions, ball position, possessor ID, and frame number.
        - player_dots (list): List of matplotlib plot markers representing players.
        - role_labels (list): List of role/abbr text labels.
        - ball_circle (Circle): The ball graphical object.
        - highlight_circle (Circle): The yellow ring around the ball possessor.
        - possessor_id_labels (list): List of player ID text labels positioned on top of player dots.
        - title (matplotlib.text.Text): Title object for displaying frame info.
        - num_frames (int): Total number of frames in the simulation.
    """
    # Extract player positions, ball position, and owner ID from frame data
    players = frame_data["players"]  
    ball = frame_data["ball"]  
    owner_id = frame_data["owner"]  

    # Update player positions and labels
    for i, pos in enumerate(players):
        pos_scaled = pos * np.array([120, 80])  # Scale player position
        player_dots[i].set_data(pos_scaled[0], pos_scaled[1])  
        role_labels[i].set_position((pos_scaled[0], pos_scaled[1] - 2))  
        role_labels[i].set_text(getattr(env.players[i], 'abbr', env.players[i].role[:2]))  
        possessor_id_labels[i].set_position((pos_scaled[0], pos_scaled[1]))

    # Update ball position
    ball_scaled = ball * np.array([120, 80])  # Scale ball position
    ball_circle.set_center(ball_scaled) 

    # Update highlight circle position based on ball owner
    if owner_id is not None:
        owner_pos = players[owner_id] * np.array([120, 80])  
        highlight_circle.set_center(owner_pos) 
    else:
        highlight_circle.set_center((-10, -10))  # Move highlight circle off-screen if no owner

    # Update title with frame information
    title.set_text(f"Frame {frame_data['frame']} / {num_frames} — Time {frame_data['frame'] / 24:.2f} s")  # Update title with frame info

# Animate the simulation (continuous playback)
def animate_simulation(env, num_frames=240, interval_ms=1000/24, action_selector=None):
    
    # Set up the pitch
    fig, _, player_dots, role_labels, ball_circle, highlight_circle, possessor_id_labels, title = setup_pitch(env)  

    # Function to update the plot for each frame
    def update(frame):
        # Support strategies that either accept (env) or (env, frame)
        try:
            actions = action_selector(env, frame) if action_selector else [0] * 22  # Get actions for current frame
        except TypeError:
            actions = action_selector(env) if action_selector else [0] * 22
        
        env.step(actions)  # Step the environment with selected actions

        # Get current player positions, ball position, and owner ID
        frame_data = {
            "players": [p.get_position().copy() for p in env.players],  
            "ball": env.ball.get_position().copy(),  
            "owner": env.ball.owner_id, 
            "frame": frame + 1  
        }

        # Update player positions, ball position, and title
        update_frame_visuals(env, frame_data, player_dots, role_labels, ball_circle, highlight_circle, possessor_id_labels, title, num_frames)
        return player_dots + [ball_circle, highlight_circle, title] + role_labels + possessor_id_labels 

    # Set up the animation
    _ = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False) 
    plt.show()  

# Visualize the simulation using a slider to move frame-by-frame (interactive mode)
def simulate_with_slider(env, num_frames=240, action_selector=None):
    """
    Visualizes the simulation using a slider to move frame-by-frame
    Args:
        - env (FootballEnv): The football environment.
        - num_frames (int): Total number of frames in the simulation.
        - action_selector (function): Function to select actions for each frame.
    Returns:
        - None
    """ 
    # Initialize the pitch and visual elements
    saved_states = []  

    # Save initial state
    saved_states.append({
        "players": [p.get_position().copy() for p in env.players],  
        "ball": env.ball.get_position().copy(), 
        "owner": env.ball.owner_id,  
        "frame": 0
    })

    # Simulate the environment for the specified number of frames
    for frame in range(1, num_frames + 1):

        # Get actions for the current frame
        # Allow compatibility with strategies that accept one or two arguments
        try:
            actions = action_selector(env, frame) if action_selector else [0] * 22  
        except TypeError:
            actions = action_selector(env) if action_selector else [0] * 22
            
        env.step(actions)  

        # Save state for this frame
        saved_states.append({
            "players": [p.get_position().copy() for p in env.players],  
            "ball": env.ball.get_position().copy(), 
            "owner": env.ball.owner_id,  
            "frame": frame  
        })

    # Set up the pitch and visual elements
    _, _, player_dots, role_labels, ball_circle, highlight_circle, possessor_id_labels, title = setup_pitch(env)  
    plt.subplots_adjust(bottom=0.22)  #

    # Create a slider for frame selection
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03]) 
    slider = Slider(ax_slider, 'Frame', 0, num_frames, valinit=0, valstep=1)  

    # Create a button to reset the slider
    ax_reset = plt.axes([0.85, 0.02, 0.1, 0.05])  
    reset_button = Button(ax_reset, 'Reset')

    # Update the plot based on the slider value
    def update_slider(val):
        # Get the current frame from the slider
        frame = int(val)
        update_frame_visuals(env, saved_states[frame], player_dots, role_labels, ball_circle, highlight_circle, possessor_id_labels, title, num_frames)  # Update visuals for selected frame

    # Reset the slider to frame 0
    def on_reset(event):
        slider.set_val(0)  

    # Connect the slider and button to their respective functions
    slider.on_changed(update_slider)  
    reset_button.on_clicked(on_reset)

    # Set the initial frame to 0
    update_slider(0)  

    # Display the plot
    plt.show()  
