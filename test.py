import matplotlib.pyplot as plt
from RLEnvironment.pitch import draw_pitch, draw_half_pitch

fig, ax = plt.subplots(figsize=(12, 8))
draw_half_pitch(ax=ax, show_grid=True, show_cell_ids=True)
plt.title("Full Pitch with Cell IDs")
plt.show()