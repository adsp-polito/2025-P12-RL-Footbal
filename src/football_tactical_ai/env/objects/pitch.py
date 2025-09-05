from football_tactical_ai.configs import pitchSettings as PS

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle
import numpy as np


# GRID INFORMATION (USED FOR RL)
# The pitch is overlaid with a 1x1m (or 5x5m) cell grid for RL training, visual debugging, and reward shaping.

# ==> Full pitch grid:
#       - Area: 130m (x) × 90m (y) = full pitch with 5m margin on all sides
#       - If using 5x5m cells:
#           - N_COLS_FULL = 130 / 5 = 26 columns
#           - N_ROWS_FULL = 90 / 5 = 18 rows
#           - Total cells: 26 × 18 = 468

# ==> Half pitch grid:
#       - Area: 70m (x) × 90m (y) = half pitch with 5m margin on all sides
#       - If using 5x5m cells:
#           - N_COLS_HALF = 70 / 5 = 14 columns
#           - N_ROWS_HALF = 90 / 5 = 18 rows
#           - Total cells: 14 × 18 = 252

# Z-ORDER LAYERS EXPLANATION
# Z-order defines the drawing priority (higher values appear on top).
# The following convention is used to control rendering layers:

# zorder = 0 ==> Field background stripes (base layer)

# zorder = 1 ==> Debug grid lines (optional, for RL debugging)

# zorder = 2 ==> Static pitch elements:
#                   - Field boundaries
#                   - Penalty areas and goal areas
#                   - Center line and circle
#                   - Penalty arcs and corner arcs
#                   - Text labels for cell indices (debugging)
#                These are usually drawn without explicitly setting zorder, as matplotlib defaults place them

# zorder = 3 ==> Dynamic elements (always drawn on top):
#              - Attacking player
#              - Defenders
#              - Ball

# Most matplotlib drawing functions (e.g., ax.plot, ax.add_patch) default to zorder=2.


class Pitch:
    """
    Represents a football pitch with defined dimensions, margins, and grid discretization.
    Provides methods to draw the full pitch, half pitch, and various pitch elements.
    Also includes methods for rendering heatmaps, rewards, and grid overlays.
    """

    def __init__(self, *, margin: float | None = None, cell_size: float | None = None):
        # Base dimensions (immutable defaults)
        self.width  = PS.FIELD_WIDTH
        self.height = PS.FIELD_HEIGHT

        # Allow optional overrides for tests / experiments
        self.margin    = margin if margin is not None else PS.MARGIN
        self.cell_size = cell_size if cell_size is not None else PS.CELL_SIZE

        # Derived coordinates
        self.x_min, self.y_min = PS.X_MIN, PS.Y_MIN
        self.x_max, self.y_max = PS.X_MAX, PS.Y_MAX
        self.half_field_x      = self.width // 2
        self.center_y          = self.height // 2

        # Grid resolution
        self.num_cells_x = int((self.x_max - self.x_min) / self.cell_size)
        self.num_cells_y = int((self.y_max - self.y_min) / self.cell_size)

        # Visual stripes
        self.stripe_width = PS.STRIPE_WIDTH

        # Area dimensions (constants from settings)
        self.penalty_depth        = PS.PENALTY_DEPTH
        self.penalty_height       = PS.PENALTY_HEIGHT
        self.goal_area_depth      = PS.GOAL_AREA_DEPTH
        self.goal_area_height     = PS.GOAL_AREA_HEIGHT
        self.center_circle_radius = PS.CENTER_CIRCLE_RADIUS
        self.penalty_spot_x_left  = PS.PENALTY_SPOT_X_LEFT
        self.penalty_spot_x_right = PS.PENALTY_SPOT_X_RIGHT
        self.goal_width           = PS.GOAL_WIDTH
        self.goal_depth           = PS.GOAL_DEPTH

    def draw_pitch(self, ax=None, field_color='green', stripes=False,
               show_grid=False, show_heatmap=False, show_rewards=False,
               reward_grid=None):
        """
        Draw the full football pitch with optional stripes, grid, heatmap, and rewards.

        Parameters:
            ax (matplotlib.axes.Axes, optional): Axis to draw the pitch on.
            field_color (str): Background color ('green' for default).
            stripes (bool): Whether to draw alternating mowing stripes.
            show_grid (bool): Whether to show reward shaping grid lines.
            show_heatmap (bool): Whether to fill cells with colors based on reward.
            show_rewards (bool): Whether to write reward numbers in the center of each cell.
            reward_grid (np.ndarray, optional): 2D array with reward values for heatmap.

        Returns:
            tuple: (figure, axis) with the completed drawing.
        """

        # Create a new figure and axis if not provided
        fig, ax = (None, ax)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Draw the background and static elements
        self._draw_background(ax, field_color, stripes)
        self._draw_static_elements(ax)

        # Draw heatmap, rewards or grid overlay if requested
        if show_heatmap and reward_grid is not None:
            self._draw_heatmap(ax, reward_grid)

        if show_rewards and reward_grid is not None:
            self._draw_rewards(ax, reward_grid)

        if show_grid:
            self._draw_grid(ax)

        
        # Set axis limits and aspect ratio
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_max, self.y_min)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax

    def draw_half_pitch(self, ax=None, field_color='green', stripes=False,
                    show_grid=False, show_heatmap=False, show_rewards=False,
                    reward_grid=None):
        """
        Draw only the offensive half of the football pitch with optional stripes, grid, heatmap, and reward annotations.

        Parameters:
            ax (matplotlib.axes.Axes, optional): Axis to draw the pitch on.
                If None, a new figure and axis will be created.
            field_color (str): Background color of the pitch ('green' by default).
            stripes (bool): Whether to add alternating mowing stripes for visual realism.
            show_grid (bool): Whether to overlay the RL reward grid on the half-pitch.
            show_heatmap (bool): Whether to fill cells with colors based on reward.
            show_rewards (bool): Whether to annotate reward numbers inside the cells.
            reward_grid (np.ndarray, optional): 2D array with reward values for heatmap.

        Returns:
            tuple: (figure, axis) containing the rendered matplotlib figure and axis.
        """

        # Create a new figure and axis if none provided by the user
        fig, ax = (None, ax)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))

        # Draw pitch background and static elements (lines, goals, etc.)
        self._draw_background(ax, field_color, stripes)
        self._draw_half_pitch_elements(ax)

        # Draw heatmap, rewards or grid overlay if requested
        if show_heatmap and reward_grid is not None:
            self._draw_heatmap(ax, reward_grid)

        if show_rewards and reward_grid is not None:
            self._draw_rewards(ax, reward_grid)

        if show_grid:
            self._draw_grid(ax)

        # Set axis limits to focus on the offensive half
        ax.set_xlim(self.half_field_x, self.x_max)
        ax.set_ylim(self.y_max, self.y_min)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax



    def _draw_background(self, ax, field_color, stripes):
        """
        Draw the pitch background color and optional mowing stripes.

        This method sets the background color of the pitch and optionally 
        adds alternating mowing stripes for enhanced visual realism.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
            field_color (str): Background color of the pitch ('green' for standard).
            stripes (bool): Whether to add alternating mowing stripes to the pitch.
        """

        # Set the background color
        ax.set_facecolor('#4CAF50' if field_color == 'green' else field_color)

        # Draw stripes if enabled
        if stripes and field_color == 'green':

            # Draw alternating stripes if enabled and background is green
            for i in range(0, self.width, self.stripe_width):
                color = '#43A047' if (i // self.stripe_width) % 2 == 0 else '#4CAF50'
                ax.add_patch(Rectangle((i, 0), self.stripe_width, self.height, color=color, zorder=0))

    def _draw_static_elements(self, ax):
        """
        Draw all static pitch elements: boundaries, center line and circle, 
        penalty areas, goals, and corner arcs.

        This method is responsible for rendering all the fixed components of the 
        football field, following standard pitch markings.
        
        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
        """

        # Draw the field boundaries (outer rectangle)
        lc = 'whitesmoke'
        ax.plot([0, 0], [0, self.height], color='white', linewidth=3)
        ax.plot([self.width, self.width], [0, self.height], color='white', linewidth=3)
        ax.plot([0, self.width], [0, 0], color='white', linewidth=3)
        ax.plot([0, self.width], [self.height, self.height], color='white', linewidth=3)

        # Draw the center line
        ax.plot([self.width // 2, self.width // 2], [0, self.height], color=lc, linewidth=2, zorder=2)

        # Draw the center circle
        ax.add_patch(Circle((self.width // 2, self.center_y), self.center_circle_radius, color=lc, fill=False, linewidth=2, zorder=2))

        # Draw the center spot
        ax.plot([self.width // 2], [self.center_y], marker='o', markersize=5, color=lc, zorder=2)

        # Draw penalty areas and goals on both sides
        for side in ['left', 'right']:
            self._draw_penalty_area(ax, side, lc)
            self._draw_goal(ax, side, lc)

        # Draw corner arcs in all four corners
        corners = [(0, 0), (self.width, 0), (0, self.height), (self.width, self.height)]
        angles = [0, 90, 270, 180]
        for (x, y), angle in zip(corners, angles):
            arc = Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2, zorder=2)
            ax.add_patch(arc)

    def _draw_half_pitch_elements(self, ax):
        """
        Draw the static field elements for the offensive half only.

        This method draws all the standard pitch markings (boundaries, center line, goals, etc.)
        but limits the axis to the offensive half of the field for focused visualization.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
        """

        # Draw the full field static elements (center line, penalty areas, goals, etc.)
        self._draw_static_elements(ax)

        # Limit the view to the offensive half of the field
        ax.set_xlim(self.width / 2, self.x_max)


    def _draw_penalty_area(self, ax, side, lc):
        """
        Draw the penalty area, goal area, penalty spot, and penalty arc for the given side.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
            side (str): 'left' or 'right' side of the field.
            lc (str): Line color for the drawing elements.
        """
        # Determine direction and x-coordinates based on side
        direction = 1 if side == 'right' else -1
        x_base = self.width if side == 'right' else 0
        penalty_x = x_base - direction * self.penalty_depth
        goal_x = x_base - direction * self.goal_area_depth

        # y-coordinates for penalty and goal areas (centered vertically)
        y1_pen = (self.height - self.penalty_height) / 2
        y2_pen = (self.height + self.penalty_height) / 2
        y1_goal = (self.height - self.goal_area_height) / 2
        y2_goal = (self.height + self.goal_area_height) / 2

        # Draw penalty area rectangle
        ax.plot([x_base, penalty_x], [y1_pen, y1_pen], color=lc, linewidth=2)
        ax.plot([penalty_x, penalty_x], [y1_pen, y2_pen], color=lc, linewidth=2)
        ax.plot([x_base, penalty_x], [y2_pen, y2_pen], color=lc, linewidth=2)

        # Draw goal area rectangle
        ax.plot([x_base, goal_x], [y1_goal, y1_goal], color=lc, linewidth=2)
        ax.plot([goal_x, goal_x], [y1_goal, y2_goal], color=lc, linewidth=2)
        ax.plot([x_base, goal_x], [y2_goal, y2_goal], color=lc, linewidth=2)

        # Draw penalty spot
        penalty_spot_x = self.penalty_spot_x_right if side == 'right' else self.penalty_spot_x_left
        ax.plot([penalty_spot_x], [self.center_y], marker='o', markersize=2.5, color=lc)

        # Draw penalty arc
        arc_center_x = penalty_spot_x - 1.8 if side == 'right' else penalty_spot_x + 1.8
        arc_angle = 180 if side == 'right' else 0
        arc = Arc((arc_center_x, self.center_y), 20, 20, angle=arc_angle, theta1=301, theta2=59, color=lc, linewidth=2, zorder=2)
        ax.add_patch(arc)


    def _draw_goal(self, ax, side, lc):
        """
        Draw the goal on the specified side.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
            side (str): 'left' or 'right' side of the field.
            lc (str): Line color for the goal.
        """
        # Position and direction of the goal rectangle
        x_goal = self.width if side == 'right' else 0
        direction = 1 if side == 'right' else -1
        goal_y1 = self.center_y - self.goal_width / 2

        # Draw the goal rectangle
        ax.add_patch(Rectangle(
            (x_goal, goal_y1),
            direction * self.goal_depth,
            self.goal_width,
            linewidth=2,
            edgecolor=lc,
            facecolor='none',
            zorder=2
        ))

    def _draw_grid(self, ax):
        """
        Draw the reward shaping grid over the entire pitch area.

        Each cell corresponds to a discrete region used for RL reward shaping.
        """
        for i in range(self.num_cells_x):
            for j in range(self.num_cells_y):
                x0 = self.x_min + i * self.cell_size
                y0 = self.y_min + j * self.cell_size

                # Draw a rectangle for each cell
                rect = Rectangle(
                    (x0, y0),
                    self.cell_size,
                    self.cell_size,
                    edgecolor='black',
                    facecolor='none',
                    linewidth=0.4,
                    zorder=1
                )
                ax.add_patch(rect)

    def _draw_heatmap(self, ax, reward_grid, show_grid=False):
        """
        Draw a heatmap of rewards over the pitch based on a reward grid.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
            reward_grid (2D np.array): Grid of reward values.
            show_grid (bool): Whether to overlay grid lines on top of the heatmap.
        """
        # Get global min and max values for the reward grid
        global_min = reward_grid.min()
        global_max = reward_grid.max()

        # Mask out cells that have the global min or max values to isolate internal cell values
        mask_internal = (reward_grid != global_min) & (reward_grid != global_max)
        internal_values = reward_grid[mask_internal]

        # Compute min and max excluding margin and goal cells, fallback to global min/max if none
        if internal_values.size > 0:
            internal_min = internal_values.min()
            internal_max = internal_values.max()
        else:
            internal_min = global_min
            internal_max = global_max

        # Iterate over each cell to draw the colored rectangle
        for i in range(self.num_cells_x):
            for j in range(self.num_cells_y):
                # Compute bottom-left corner coordinates of the cell in pitch meters
                x0 = self.x_min + i * self.cell_size
                y0 = self.y_min + j * self.cell_size

                reward = reward_grid[i, j]

                # Normalize reward to [0, 1] for colormap use
                if reward == global_min:
                    norm = 0.0  # Fixed color for margin cells
                elif reward == global_max:
                    norm = 1.0  # Fixed color for goal cells
                else:
                    if internal_max == internal_min:
                        norm = 0.5  # Avoid division by zero when all internal values equal
                    else:
                        norm = (reward - internal_min) / (internal_max - internal_min)
                    norm = np.clip(norm, 0, 1)

                # Map normalized reward to a color using a colormap
                color = plt.cm.coolwarm(norm)

                # Create a rectangle patch for the heatmap cell
                rect = Rectangle(
                    (x0, y0),
                    self.cell_size,
                    self.cell_size,
                    facecolor=color,
                    edgecolor='black' if show_grid else 'none',  # Draw grid lines if requested
                    linewidth=0.4,
                    alpha=0.6, 
                    zorder=1
                )
                ax.add_patch(rect)


    def _draw_rewards(self, ax, reward_grid):
        """
        Draw numeric reward values inside each cell of the reward grid.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
            reward_grid (2D np.array): Grid of reward values.
        """
        # Iterate over each cell and add text annotation with reward value
        for i in range(self.num_cells_x):
            for j in range(self.num_cells_y):
                # Compute bottom-left corner coordinates of the cell in pitch meters
                x0 = self.x_min + i * self.cell_size
                y0 = self.y_min + j * self.cell_size

                reward = reward_grid[i, j]

                # Add centered text with the reward value formatted with 2 decimals
                ax.text(
                    x0 + self.cell_size / 2,
                    y0 + self.cell_size / 2,
                    f"{reward:.2f}",
                    ha='center',
                    va='center',
                    fontsize=self.cell_size * 0.85,  # Scale font size with cell size
                    color='black',
                    alpha=0.7,
                    zorder=2  
                )
