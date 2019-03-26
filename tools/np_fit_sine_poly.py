import numpy as np
from itertools import chain
from np_least_squares import wls_gen


def fit_sine_poly(t, x, polyord, freqs, dt=None, dx=None, tran=None, center=True, t0=0.):
    if tran is not None:
        inside = (t >= tran[1]) & (t <= tran[2])
        t = t[inside]
        x = x[inside]
        dt = dt[inside]
        dx = dx[inside]

    t_c = 0.
    if center:
        t_c = np.mean(t)

    poly_function = [lambda y: np.power(y - t_c, m) for m in range(polyord)]
    poly_derivative = [lambda y: m * np.power(y - t_c, m - 1) for m in range(polyord)]
    sine_function = list(chain.from_iterable(
        (lambda y: np.cos(2 * np.pi * freq * (y - t0)), lambda y: np.sin(2 * np.pi * freq * y)) for freq in freqs))
    sine_derivative = list(chain.from_iterable((lambda y: -2 * np.pi * freq * np.sin(2 * np.pi * freq * (y - t0)),
                                                lambda y: 2 * np.pi * freq * np.cos(2 * np.pi * freq * (y - t0))) for
                                               freq in freqs))
    function = poly_function + sine_function
    derivative = poly_derivative + sine_derivative

    return wls_gen(t, x, dX=dt, dY=dx, functions=function, d_functions=derivative)
