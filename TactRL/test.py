import matplotlib.pyplot as plt
from field import draw_half_pitch, draw_pitch

fig, ax = plt.subplots()
#plt.get_current_fig_manager().full_screen_toggle()
# draw_pitch(ax, field_color='green', show_grid=True)
draw_half_pitch(ax, field_color='green', show_grid=True)
plt.show()