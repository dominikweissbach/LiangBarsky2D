# Vectorized version of the Liang-Barsky line clipping in 2D using jax
# Liang, Y. D., and Barsky, B., "A New Concept and Method for Line Clipping", ACM Transactions on Graphics, 3(1):1–22, January 1984.
import jax
import jax.numpy as jnp
from jax import vmap

def liang_barsky_clip(p_start, p_end, x_min, y_min, x_max, y_max):
    # Liang-Barsky line clipping algorithm
    # p_start, p_end: shape (2,) arrays representing the start and end points of the line
    # x_min, y_min, x_max, y_max: scalars defining the clipping rectangle
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    p = jnp.array([-dx, dx, -dy, dy])
    q = jnp.array([p_start[0] - x_min,
                   x_max - p_start[0],
                   p_start[1] - y_min,
                   y_max - p_start[1]])

    t = q/p

    t_enter = jnp.where(p<0, t, -jnp.inf)
    t_enter_within_box = jnp.maximum(0.0, jnp.max(t_enter))

    t_exit = jnp.where(p>0, t, jnp.inf)
    t_exit_within_box = jnp.minimum(1.0, jnp.min(t_exit))

    valid = t_enter_within_box < t_exit_within_box

    start_clipped = p_start + t_enter_within_box * jnp.array([dx, dy])
    end_clipped = p_start + t_exit_within_box * jnp.array([dx, dy])
    return start_clipped, end_clipped, valid



