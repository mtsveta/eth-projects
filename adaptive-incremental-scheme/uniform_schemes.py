import math
import numpy as np

from plotting import *

def rk_4th_order(t_0, t_final, h, y_0, y, f_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    err_array = np.append(np.array([]), np.array([]))

    def norm(val):
        return math.fabs(val)

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

        print('% -------------------------------------------------------------------------------------------- %')
        print('t = %4.4e\n' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        k_1 = h * f_n_val

        f_nk1_val = f_n(t_n + h / 2, y_n + k_1 / 2)
        k_2 = h * f_nk1_val

        f_nk2_val = f_n(t_n + h / 2, y_n + k_2 / 2)
        k_3 = h * f_nk2_val

        f_nk3_val = f_n(t_n + h, y_n + k_3)
        k_4 = h * f_nk3_val

        f_evals += 4

        y_n1 = y_n + k_1 / 6 + k_2 / 3 + k_3 / 3 + k_4 / 6

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))

        t_array = np.append(t_array, np.array([t_n + h]))
        err_array = np.append(err_array, np.array([err]))
        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_uniform_results(t_array, yn_array, y_array, err_array, result_path)
    return err, n, f_evals

def tdrk_4th_order(t_0, t_final, h, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    err_array = np.append(np.array([]), np.array([]))

    def norm(val):
        return math.fabs(val)

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

        print('% -------------------------------------------------------------------------------------------- %')
        print('t = %4.4e\n' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # reconstruct approximation y_1/2 of the half step
        y_half = y_n + h / 2 * f_n_val + h ** 2 / 8 * fprime_n_val

        # evaluate f_* and (f')_*
        f_half_val = f_n(t_n + h / 2, y_half)
        fprime_half_val = fprime_n(t_n + h / 2, y_half)
        f_evals += 2

        y_n1 = y_n + h * f_n_val \
               + h**2 / 6 * fprime_n_val \
               + h**2 / 3 * fprime_half_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n + h * f_n_val + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_2nd)
        print('h = %4.4e\terr = %4.4e\tlte = %4.4e' % (h, err, lte))

        t_array = np.append(t_array, np.array([t_n + h]))
        err_array = np.append(err_array, np.array([err]))
        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_uniform_results(t_array, yn_array, y_array, err_array, result_path)
    return lte, err, n, f_evals