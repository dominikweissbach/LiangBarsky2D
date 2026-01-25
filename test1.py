from liang_barsky import liang_barsky_clip
from jax import numpy as jnp
import matplotlib.pyplot as plt


x_min, y_min, x_max, y_max = 0.0, 0.0, 10.0, 10.0
p_start = jnp.array([-5.0, 5.0])
p_end = jnp.array([15.0, 5.0])

start, end, valid = liang_barsky_clip(p_start, p_end, x_min, y_min, x_max, y_max)

print("Valid:", valid)

plt.figure()
plt.xlim(-10, 20)
plt.ylim(-10, 20)
plt.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', label='Original Line')
# Draw clipping rectangle
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], 'b-', label='Clipping Rectangle')

if valid:
    plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, label='Clipped Line')

plt.show()