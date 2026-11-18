"""Shared Legendre fits and training data for the generalization diagrams."""

import manimgx as gx
import numpy as np
import scipy.special
from welch_style import GREEN


def f(x): return 0.5*(x**2)


def fit_legendre_pinv(x_train, y_train, degree, x_min=-2, x_max=2):
    """Fit Legendre polynomial using pseudoinverse"""
    # Rescale x to [-1, 1]
    x_scaled = 2 * (x_train - x_min) / (x_max - x_min) - 1

    feature_degrees = np.arange(degree + 1)[:, None]
    X_train_poly = scipy.special.eval_legendre(feature_degrees, x_scaled).T
    beta_hat = np.linalg.pinv(X_train_poly) @ y_train
    return beta_hat


def eval_legendre_poly(beta, x, degree, x_min=-2, x_max=2):
    """Evaluate Legendre polynomial with given coefficients"""
    # Rescale x to [-1, 1]
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

    feature_degrees = np.arange(degree + 1)[:, None]
    X_poly = scipy.special.eval_legendre(feature_degrees, x_scaled).T
    return X_poly @ beta


def get_noisy_data(n_points=10, noise_level=0.2, random_seed=428):
    np.random.seed(random_seed)
    x=np.random.uniform(-2, 2, n_points)
    y=f(x)+noise_level*np.random.randn(n_points)
    return x,y


def get_fit_line(axes, x_train, y_train, x_test, y_test, all_x, degree=1, color=GREEN):
    beta_hat = fit_legendre_pinv(x_train, y_train, degree)
    all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
    y_train_pred = eval_legendre_poly(beta_hat, x_train, degree)
    y_test_pred = eval_legendre_poly(beta_hat, x_test, degree)
    all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
    train_error = np.mean((y_train - y_train_pred)**2)
    test_error = np.mean((y_test - y_test_pred)**2)

    fit_points = [axes.c2p(all_x[i], all_y_fit[i]) for i in range(len(all_x))]
    fit_line = gx.VMobject(stroke_width=3)
    fit_line.set_points_smoothly(fit_points)
    fit_line.set_color(color)
    return fit_line, test_error, train_error, y_train_pred, y_test_pred
