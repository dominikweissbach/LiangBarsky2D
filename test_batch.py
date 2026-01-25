from liang_barsky import liang_barsky_clip, liang_barsky_clip_batch
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

number_of_lines = 100

start_points = np.random.rand(number_of_lines, 2) * 3 - 1
end_points = np.random.rand(number_of_lines, 2) * 3 - 1

x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
start, end, valid = liang_barsky_clip_batch(start_points, end_points, x_min, y_min, x_max, y_max)

print("Valid:", valid)

plt.figure()
for i in range(len(start_points)):
    plt.plot([start_points[i, 0], end_points[i, 0]], [start_points[i, 1], end_points[i, 1]], 'r--', label='Original Line' if i == 0 else "")
    if valid[i]:
        plt.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], 'g-', linewidth=2, label='Clipped Line' if i == 0 else "")
# Draw clipping rectangle
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], 'b-', label='Clipping Rectangle')

plt.show()