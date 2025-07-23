import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle
import numpy as np


# GRID INFORMATION (USED FOR RL)
# The pitch is overlaid with a 5x5m (or 1x1m) cell grid for RL training, visual debugging, and reward shaping.

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
# Z-order defines the drawing priority (higher values appear on top)
# We use the following convention to control rendering layers:

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
# We make it explicit where needed to ensure consistent rendering layers.

class Pitch:
    """
    Pitch class for representing a football field and providing utilities for 
    grid-based reinforcement learning environments and tactical analysis.

    This class encapsulates:
    - Field dimensions and proportions (including margins, penalty areas, goals, etc.)
    - Grid discretization for reward shaping and visualization
    - Coordinate normalization and denormalization between meters and [0, 1]
    - Rendering functions for full and half-pitch visualization using matplotlib

    Attributes:
        FIELD_WIDTH (float): Length of the field in meters (x-axis, typically 120m)
        FIELD_HEIGHT (float): Height of the field in meters (y-axis, typically 80m)
        X_MIN, Y_MIN (float): Margins included, lower bounds of the field in meters
        X_MAX, Y_MAX (float): Margins included, upper bounds of the field in meters
        CELL_SIZE (float): Grid resolution in meters (e.g., 1m or 5m cells)
        num_cells_x, num_cells_y (int): Grid dimensions in number of cells
        Other attributes define standard football field elements.

    Typical usage:
        pitch = Pitch()
        pitch.assert_coordinates_match_helpers()
        fig, ax = pitch.draw_pitch(show_grid=True)
    """

    def __init__(self, field_width=120, field_height=80, margin=5, cell_size=1):
        """
        Initialize the football pitch with dimensions, margins, and grid discretization.

        Parameters:
            field_width (float): Length of the field in meters (default 120).
            field_height (float): Height of the field in meters (default 80).
            margin (float): Extra margin around the pitch in meters (default 5).
            cell_size (float): Size of each reward grid cell in meters (default 1).
        """

        # Field dimensions
        self.FIELD_WIDTH = field_width
        self.FIELD_HEIGHT = field_height
        self.HALF_FIELD_X = field_width // 2
        self.CENTER_Y = field_height // 2
        self.STRIPE_WIDTH = 5  # Visual rendering stripes width in meters

        # Field margins (expands the pitch for RL grids, visualization, etc.)
        self.X_MIN, self.Y_MIN = -margin, -margin
        self.X_MAX, self.Y_MAX = field_width + margin, field_height + margin

        # Standard football field proportions (measured in meters)
        self.PENALTY_AREA_DEPTH = 18
        self.PENALTY_AREA_HEIGHT = 44
        self.GOAL_AREA_DEPTH = 6
        self.GOAL_AREA_HEIGHT = 20
        self.PENALTY_SPOT_X_RIGHT = 109
        self.PENALTY_SPOT_X_LEFT = 11
        self.CENTER_CIRCLE_RADIUS = 10
        self.GOAL_HEIGHT = 7.32
        self.GOAL_DEPTH = 2.44

        # Grid cell size for reward shaping (in meters)
        # Smaller cells (e.g., 1m) offer finer resolution but increase computational cost.
        # Larger cells (e.g., 5m) reduce detail but improve efficiency.
        self.CELL_SIZE = cell_size
        self.num_cells_x = int((self.X_MAX - self.X_MIN) / self.CELL_SIZE)
        self.num_cells_y = int((self.Y_MAX - self.Y_MIN) / self.CELL_SIZE)


    def assert_coordinates_match_helpers(self):
        """
        Validate that the pitch margins (X_MIN, X_MAX, Y_MIN, Y_MAX) are consistent 
        with the constants defined in helpers.py.

        This method should be called during initialization or debugging to prevent 
        inconsistencies between the pitch object and the normalization functions 
        that rely on fixed field boundaries.

        Raises:
            AssertionError: If any of the field boundaries differ from those in helpers.py.
        """


        # Import constants from helpers.py for comparison
        from helpers.helperFunctions import X_MIN as H_X_MIN, X_MAX as H_X_MAX, Y_MIN as H_Y_MIN, Y_MAX as H_Y_MAX

        # Check consistency between this Pitch instance and the normalization constants
        assert self.X_MIN == H_X_MIN, f"\nX_MIN mismatch: {self.X_MIN} != {H_X_MIN}"
        assert self.X_MAX == H_X_MAX, f"\nX_MAX mismatch: {self.X_MAX} != {H_X_MAX}"
        assert self.Y_MIN == H_Y_MIN, f"\nY_MIN mismatch: {self.Y_MIN} != {H_Y_MIN}"
        assert self.Y_MAX == H_Y_MAX, f"\nY_MAX mismatch: {self.Y_MAX} != {H_Y_MAX}"


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
        ax.set_xlim(self.X_MIN, self.X_MAX)
        ax.set_ylim(self.Y_MAX, self.Y_MIN)
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
        ax.set_xlim(self.FIELD_WIDTH / 2, self.X_MAX)
        ax.set_ylim(self.Y_MAX, self.Y_MIN)
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
            for i in range(0, self.FIELD_WIDTH, self.STRIPE_WIDTH):
                color = '#43A047' if (i // self.STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
                ax.add_patch(Rectangle((i, 0), self.STRIPE_WIDTH, self.FIELD_HEIGHT, color=color, zorder=0))

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
        ax.plot([0, 0], [0, self.FIELD_HEIGHT], color='white', linewidth=3)
        ax.plot([self.FIELD_WIDTH, self.FIELD_WIDTH], [0, self.FIELD_HEIGHT], color='white', linewidth=3)
        ax.plot([0, self.FIELD_WIDTH], [0, 0], color='white', linewidth=3)
        ax.plot([0, self.FIELD_WIDTH], [self.FIELD_HEIGHT, self.FIELD_HEIGHT], color='white', linewidth=3)

        # Draw the center line
        ax.plot([self.FIELD_WIDTH // 2, self.FIELD_WIDTH // 2], [0, self.FIELD_HEIGHT], color=lc, linewidth=2, zorder=2)

        # Draw the center circle
        ax.add_patch(Circle((self.FIELD_WIDTH // 2, self.CENTER_Y), self.CENTER_CIRCLE_RADIUS, color=lc, fill=False, linewidth=2, zorder=2))
        
        # Draw penalty areas and goals on both sides
        for side in ['left', 'right']:
            self._draw_penalty_area(ax, side, lc)
            self._draw_goal(ax, side, lc)

        # Draw corner arcs in all four corners
        corners = [(0, 0), (self.FIELD_WIDTH, 0), (0, self.FIELD_HEIGHT), (self.FIELD_WIDTH, self.FIELD_HEIGHT)]
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
        ax.set_xlim(self.FIELD_WIDTH / 2, self.X_MAX)


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
        x_base = self.FIELD_WIDTH if side == 'right' else 0
        penalty_x = x_base - direction * self.PENALTY_AREA_DEPTH
        goal_x = x_base - direction * self.GOAL_AREA_DEPTH

        # y-coordinates for penalty and goal areas (centered vertically)
        y1_pen = (self.FIELD_HEIGHT - self.PENALTY_AREA_HEIGHT) / 2
        y2_pen = (self.FIELD_HEIGHT + self.PENALTY_AREA_HEIGHT) / 2
        y1_goal = (self.FIELD_HEIGHT - self.GOAL_AREA_HEIGHT) / 2
        y2_goal = (self.FIELD_HEIGHT + self.GOAL_AREA_HEIGHT) / 2

        # Draw penalty area rectangle
        ax.plot([x_base, penalty_x], [y1_pen, y1_pen], color=lc, linewidth=2)
        ax.plot([penalty_x, penalty_x], [y1_pen, y2_pen], color=lc, linewidth=2)
        ax.plot([x_base, penalty_x], [y2_pen, y2_pen], color=lc, linewidth=2)

        # Draw goal area rectangle
        ax.plot([x_base, goal_x], [y1_goal, y1_goal], color=lc, linewidth=2)
        ax.plot([goal_x, goal_x], [y1_goal, y2_goal], color=lc, linewidth=2)
        ax.plot([x_base, goal_x], [y2_goal, y2_goal], color=lc, linewidth=2)

        # Draw penalty spot
        penalty_spot_x = self.PENALTY_SPOT_X_RIGHT if side == 'right' else self.PENALTY_SPOT_X_LEFT
        ax.plot([penalty_spot_x], [self.CENTER_Y], marker='o', markersize=2.5, color=lc)

        # Draw penalty arc
        arc_center_x = penalty_spot_x - 1.8 if side == 'right' else penalty_spot_x + 1.8
        arc_angle = 180 if side == 'right' else 0
        arc = Arc((arc_center_x, self.CENTER_Y), 20, 20, angle=arc_angle, theta1=301, theta2=59, color=lc, linewidth=2, zorder=2)
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
        x_goal = self.FIELD_WIDTH if side == 'right' else 0
        direction = 1 if side == 'right' else -1
        goal_y1 = self.CENTER_Y - self.GOAL_HEIGHT / 2

        # Draw the goal rectangle
        ax.add_patch(Rectangle(
            (x_goal, goal_y1),
            direction * self.GOAL_DEPTH,
            self.GOAL_HEIGHT,
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
                x0 = self.X_MIN + i * self.CELL_SIZE
                y0 = self.Y_MIN + j * self.CELL_SIZE

                # Draw a rectangle for each cell
                rect = Rectangle(
                    (x0, y0),
                    self.CELL_SIZE,
                    self.CELL_SIZE,
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
                x0 = self.X_MIN + i * self.CELL_SIZE
                y0 = self.Y_MIN + j * self.CELL_SIZE

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
                    self.CELL_SIZE,
                    self.CELL_SIZE,
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
                x0 = self.X_MIN + i * self.CELL_SIZE
                y0 = self.Y_MIN + j * self.CELL_SIZE

                reward = reward_grid[i, j]

                # Add centered text with the reward value formatted with 2 decimals
                ax.text(
                    x0 + self.CELL_SIZE / 2,
                    y0 + self.CELL_SIZE / 2,
                    f"{reward:.2f}",
                    ha='center',
                    va='center',
                    fontsize=self.CELL_SIZE * 0.85,  # Scale font size with cell size
                    color='black',
                    alpha=0.7,
                    zorder=2  
                )
