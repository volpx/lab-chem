import numpy as np
from numpy.linalg import inv as np_inv
from scipy.misc import derivative


def wls(F, Y, w=None, dY=None):
    w_given = True
    if dY is not None:
        w = np.reciprocal(np.square(dY))
    elif w is None:
        # assume errors are NOT specified, and thus we assign unity error to each point
        w = np.ones(Y.shape)
        dY = np.zeros(Y.shape)
        w_given = False
    else:
        dY = np.sqrt(np.reciprocal(w))

    V = np.sum(F.T * w * Y, axis=1)
    C = np_inv(F.T * w @ F)  # covariance matrix
    A = C @ V

    Yfit = F @ A  # model
    dYfit = Y - Yfit  # residual
    DOF = Y.size - A.size  # number of DOF
    dYmean = np.sum(dYfit) * np.reciprocal(DOF)  # Average fit residual
    chi2 = np.asarray([np.sum(np.square(dYfit) * w) * np.reciprocal(DOF)])
    sigma = np.sqrt(np.sum(np.square(dYfit)) * np.reciprocal(DOF))  # RMS residual
    if not w_given:
        # here we assume a good fit, and scale the covariance matrix such that chi^2 would be = 1
        # this allows us to get an idea of the fit errors
        C = C * chi2
        chi2 = np.ones(chi2.shape)

    args = {'model': [Yfit], 'residual': [dYfit], 'dResidual': [dY], 'average_residual': [dYmean],
            'RMS_residual': [sigma], 'chi2': [chi2], 'DOF': [DOF], 'covariance': [C]}

    dA = np.sqrt(np.diag(C))

    return np.squeeze(A), np.squeeze(dA), args


def error_propagation(x0, functions=None, d_functions=None, coef=None, dx0=None, Dx0=None, dx=1e-5):
    # x0: nominal values
    # dx0: std values of x0
    # Dx0: variances of x0
    # linear combination of functions f tot = sum ( coef * functions)
    # d_functions = d(functions) / dx0
    # dx is infinitesimal increment for derivative
    # x0, dx0, Dx0, coef: np.array
    # functions, d_functions: list
    # return error propagation from x0 to y0

    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0).reshape((-1, 1))
    else:
        x0 = x0.reshape((-1, 1))

    if Dx0 is None:
        Dx0 = np.square(dx0)

    Dx0 = Dx0.reshape((-1, 1))

    if d_functions is not None:
        der = np.array([d_f(x0) for d_f in d_functions])
        if coef is None:
            coef = np.ones(len(d_functions)).reshape((-1, 1))

    elif functions is not None:
        der = np.array([derivative(f, x0, dx=dx) for f in functions])
        if coef is None:
            coef = np.ones(d_functions.shape).reshape((-1, 1))

    else:
        raise KeyError('functions and d_functions is None')

    coef = coef.reshape((-1,))
    Df = np.square(np.sum(np.squeeze(der).T * coef, axis=1)).reshape((-1, 1)) * Dx0

    if dx0 is not None:
        Df = np.sqrt(Df)

    return np.squeeze(Df)


def wls_gen(X, Y, dX=None, dY=None, functions=None, d_functions=None):
    # d_functions is analytic form of derivative of functions. If unknown it will be calculate numerically.
    # default functions is linear proportional
    if functions is None:
        functions = [lambda x: x, lambda x: np.ones(X.shape)]
        d_functions = [lambda x: np.ones(X.Shape), lambda x: np.zeros(X.Shape)]

    F = np.array([f(X) for f in functions]).T
    A, dA, args = wls(F, Y, dY=dY)

    if dX is not None:
        if d_functions is None:
            w = np.reciprocal(
                np.square(dY) + error_propagation(x0=X, functions=functions, coef=A, Dx0=np.square(dX)))
        else:
            w = np.reciprocal(
                np.square(dY) + error_propagation(x0=X, d_functions=d_functions, coef=A, Dx0=np.square(dX)))

        A, dA, args = wls(F, Y, w=w)

    return A, dA, args
