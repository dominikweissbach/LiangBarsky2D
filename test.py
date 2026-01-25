from liang_barsky import liang_barsky_clip
from jax import numpy as jnp
import matplotlib.pyplot as plt

test = "parallel_missed"  # Options: "diagonal", "off_diagonal", "missed", "straight"


if test == "diagonal":
    p_start = jnp.array([-1.0, -1.0])
    p_end = jnp.array([2.0, 2.0])
elif test == "off_diagonal":
    p_start = jnp.array([-0.2, 0.3])
    p_end = jnp.array([0.5, 1.2])
elif test == "missed":
    p_start = jnp.array([-1.0, 0.5])
    p_end = jnp.array([0.5, 1.5])
elif test == "straight":
    p_start = jnp.array([-1.0, 0.5])
    p_end = jnp.array([2.0, 0.5])
elif test == "parallel_missed":
    p_start = jnp.array([-1.0, 1.5])
    p_end = jnp.array([2.0, 1.5])

x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
start, end, valid = liang_barsky_clip(p_start, p_end, x_min, y_min, x_max, y_max)

print("Valid:", valid)

plt.figure()
plt.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', label='Original Line')
# Draw clipping rectangle
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], 'b-', label='Clipping Rectangle')

if valid:
    plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, label='Clipped Line')

plt.show()