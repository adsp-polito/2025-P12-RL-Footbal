import matplotlib.pyplot as plt
from RLEnvironment.pitch import draw_pitch, draw_half_pitch, FIELD_WIDTH, FIELD_HEIGHT

def test_full_pitch_render():
    """
    Test the full pitch rendering with grid and cell IDs.
    Shows a static field with optional visual debugging aids.
    """
    ax = draw_pitch(
        field_color='green',
        show_grid=True,
        show_cell_ids=True,
        stripes=True
    )

    ax.set_title("Full Football Pitch (with Grid & Cell Indices)", fontsize=14)
    plt.show()

if __name__ == "__main__":
    test_full_pitch_render()