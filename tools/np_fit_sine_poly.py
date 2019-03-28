import numpy as np
from itertools import chain
from np_least_squares import wls_gen


def fit_sine_poly(t, x, polyord=1, freqs=None, dt=None, dx=None, tran=None, center=False, t0=0.):
    if tran is not None:
        inside = (t >= tran[1]) & (t <= tran[2])
        t = t[inside]
        x = x[inside]
        dt = dt[inside]
        dx = dx[inside]

    t_c = 0.
    if center:
        t_c = np.mean(t)

    poly_function = [(lambda z: (lambda y: np.power((y - t_c), z)))(m) for m in np.arange(0, polyord+1)]
    poly_derivative = [(lambda z: (lambda y: z * np.power((y - t_c), z-1)))(m) for m in np.arange(0, polyord+1)]
    sine_function = []
    sine_derivative = []
    if freqs is not None:
        a_function = [(lambda z: (lambda y: np.cos(2 * np.pi * z * (y - t0))))(freq) for freq in freqs]
        b_function = [(lambda z: (lambda y: np.sin(2 * np.pi * z * (y - t0))))(freq) for freq in freqs]

        sine_function = [None] * 2 * len(freqs)
        sine_function[::2] = a_function
        sine_function[1::2] = b_function

        a_derivative = [(lambda z: (lambda y: -2 * np.pi * z * np.sin(2 * np.pi * z * (y - t0))))(freq) for freq in freqs]
        b_derivative = [(lambda z: (lambda y: 2 * np.pi * z * np.cos(2 * np.pi * z * (y - t0))))(freq) for freq in freqs]

        sine_derivative = [None] * 2 * len(freqs)
        sine_derivative[::2] = a_derivative
        sine_derivative[1::2] = b_derivative

    return wls_gen(t, x, dX=dt, dY=dx, functions=poly_function + sine_function,
                   d_functions=poly_derivative + sine_derivative)
