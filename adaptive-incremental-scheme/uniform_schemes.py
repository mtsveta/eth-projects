import math
import numpy as np

from plotting import *


def norm(val):
    return math.fabs(val)

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

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

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

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

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

        t_array = np.append(t_array, np.array([t_n + h]))
        err_array = np.append(err_array, np.array([err]))
        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_uniform_results(t_array, yn_array, y_array, err_array, result_path)
    return lte, err, n, f_evals


def tdrk_5th_order(t_0, t_final, h, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    err_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        # stage 1
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # stage 2
        a_21 = 2 / 5
        a_bar_21 = 2 / 25

        y_2 = y_n + a_21 * h * f_n_val + a_bar_21 * h ** 2 * fprime_n_val
        c_2 = 2 / 5

        f_2_val = f_n(t_n + c_2 * h, y_2)
        fprime_2_val = fprime_n(t_n + c_2 * h, y_2)
        f_evals += 2

        # stage 3
        a_31 = 139 / 64
        a_32 = -75 / 64
        a_bar_31 = 17 / 64
        a_bar_32 = 45 / 64

        y_3 = y_n + h * (a_31 * f_n_val + a_32 * f_2_val) \
                  + h**2 * (a_bar_31 * fprime_n_val + a_bar_32 * fprime_2_val)
        c_3 = 1

        f_3_val = f_n(t_n + c_3 * h, y_3)
        fprime_3_val = fprime_n(t_n + c_3 * h, y_3)
        f_evals += 2

        # reconstruct approximation y_{n+1} of the 5th order
        b_1 = 9/16
        b_2 = 125/432
        b_3 = 4/27
        b_bar_1 = 1/16
        b_bar_2 = 25/144
        b_bar_3 = 0

        y_n1 = y_n \
               + h * (b_1 * f_n_val + b_2 * f_2_val + b_3 * f_3_val) \
               + h**2 * (b_bar_1 * fprime_n_val + b_bar_2 * fprime_2_val + b_bar_3 * fprime_3_val)
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n \
                   + h * f_n_val \
                   + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_2nd)

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals