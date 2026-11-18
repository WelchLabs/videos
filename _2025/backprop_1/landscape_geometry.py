"""Coordinate and finite difference helpers for the Backprop 1 loss landscapes."""


def get_pivot_and_scale(axis_min, axis_max, axis_end):
    return axis_min, axis_end / (axis_max - axis_min)


def get_numerical_gradient(surface_fn, u, v, epsilon=0.01):
    height = surface_fn(u, v)[2]
    return ((surface_fn(u + epsilon, v)[2] - height) / epsilon,
            (surface_fn(u, v + epsilon)[2] - height) / epsilon)
