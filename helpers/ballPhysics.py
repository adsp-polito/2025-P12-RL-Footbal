import numpy as np

# ------------------------------------------------------------------------------
# Friction model rationale:
#
# In this simulation, friction is applied once per frame to the ball's velocity.
# Supposing that the simulation runs at 24 frames per second, even moderate per-frame
# friction values (e.g., 0.01 or 1%) would result in a rapid loss of speed over
# just a few seconds. For example:
#
#   - A friction_factor of 0.01 → 99% speed retained per frame
#   - After 1 second (24 frames): (0.99)^24 ≈ 0.79 → 21% velocity lost
#   - After 2 seconds: (0.99)^48 ≈ 0.63 → 37% velocity lost
#
# While this may be acceptable for certain physics simulations, in football
# the ball typically travels 10–30 meters in a pass and needs to maintain
# momentum long enough to realistically simulate this behavior.
#
# For this reason, we use a very small friction factor (e.g., 0.001):
#   - (0.999)^24 ≈ 0.976 → only 2.4% lost after 1 second
#   - This provides a much smoother and more realistic deceleration
#     over multiple seconds of simulation.
# ------------------------------------------------------------------------------
def apply_friction(velocity, friction_factor=0.0015):
    """
    Applies friction to reduce ball velocity over time.

    Args:
        - velocity (np.ndarray): Current velocity vector [vx, vy]
        - friction_factor (float): Multiplicative decay per frame (between 0 and 1)

    Returns:
        np.ndarray: Updated velocity vector after applying friction
    """
    return velocity * (1 - friction_factor)

# ------------------------------------------------------------------------------
# Future Extensions:
#  - Ball spin:
#       If needed for simulating free kicks, we could incorporate curvature caused
#       by spin — though likely unnecessary for most tactical AI use cases.
#
#   - Bounce dynamics:
#       Add logic for how the ball interacts with boundaries (e.g., walls, goalposts)
#       using a coefficient of restitution to simulate elastic collisions.
#
#   - Deflection mechanics:
#       Enable the ball to change trajectory upon player contact based on angle,
#       speed, and point of impact, useful for modeling blocks or glancing touches.
#
#   - Air drag:
#       Introduce velocity-proportional resistance to simulate ball slowing over
#       longer distances or under windy conditions (if modeled).
#
#   - Rolling resistance:
#       For extremely fine-grained realism, distinguish between sliding and rolling
#       phases on different pitch types.
#
#   - Magnus effect (spin curve):
#       If needed for simulating free kicks, we could incorporate curvature caused
#       by spin — though likely unnecessary for most tactical AI use cases.
# ------------------------------------------------------------------------------