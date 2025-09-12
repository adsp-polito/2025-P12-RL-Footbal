import matplotlib.pyplot as plt
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.env.scenarios.multiAgent.rewardGrids import (
    build_attacker_grid, build_defender_grid, build_goalkeeper_grid
)

def plot_grids(selected_roles):
    """
    Plot reward grids for up to 2 roles, each shown for Team A and Team B.
    Args:
        selected_roles (list[str]): roles to visualize, e.g. ["LW"] or ["LCB", "GK"]
    """
    pitch = Pitch()
    teams = ["A", "B"]

    # Create subplots: rows = number of roles, cols = 2 teams
    fig, axes = plt.subplots(len(selected_roles), len(teams), figsize=(8, 4 * len(selected_roles)))

    # Handle single role case (axes would not be 2D)
    if len(selected_roles) == 1:
        axes = [axes]  # make it iterable

    for i, role in enumerate(selected_roles):
        for j, team in enumerate(teams):
            # Build the grid depending on role type
            if role in ["LW", "RW", "CF", "LCF", "RCF", "SS", "ATT"]:
                grid = build_attacker_grid(pitch, role=role, team=team)
            elif role in ["LCB", "RCB", "CB", "DEF"]:
                grid = build_defender_grid(pitch, role=role, team=team)
            elif role == "GK":
                grid = build_goalkeeper_grid(pitch, team=team)
            else:
                raise ValueError(f"Unknown role: {role}")

            ax = axes[i][j] if len(selected_roles) > 1 else axes[j]
            pitch.draw_pitch(
                ax=ax,
                show_grid=False,
                show_heatmap=True,
                show_rewards=False,
                reward_grid=grid
            )
            ax.set_title(f"{role} - Team {team}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # choose up to 2 roles here
    roles_to_plot = ["LCF", "RCB"]   # example: defender + goalkeeper
    plot_grids(roles_to_plot)