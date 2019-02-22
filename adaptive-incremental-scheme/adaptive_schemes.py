import math
import numpy as np

from plotting import *

# auxiliary functions estimating an intermediate step h_star
def h_star_estimate(y_n, fprime_n, eps_rel, eps_abs):
    return math.sqrt(2 * (eps_rel * math.fabs(y_n) + eps_abs) / math.fabs(fprime_n))

#
def h_estimate(h_star, y_n, f_n, fprime_n, f_star, fprime_star, eps_rel, eps_abs):
    C = math.fabs(  1 / (2 * h_star ** 3) * (f_n - f_star)
                  + 1 / (4 * h_star ** 2) * (fprime_n + fprime_star))
    return math.pow((eps_rel * math.fabs(y_n) + eps_abs) / C, 0.25)

def norm(val):
    return math.fabs(val)

def adaptive_4th_order_2nd_middle_step_matched_with_tdrk(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path):
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
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val, eps_rel, eps_abs)
        else:
            # h = h_new
            h = h_pred

        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n
        rho = 2
        h_star = h / rho

        # reconstruct approximation y_star with the 2nd order tdrk
        y_star = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
        # reconstruct approximation for the 1st order
        y_1st = y_n + h_star * f_n_val

        # analysis of the errors of the h_star step
        err_star = norm(y_star - y(t_n + h_star))
        lte_star = norm(y_star - y_1st)

        # predicted time-step
        const = 1.1
        p = 2
        h_star_new = const * math.pow(math.fabs(eps_abs / lte_star), 1 / (p + 1)) * h_star

        # evaluate f_* and (f')_*
        f_star_val = f_n(t_n + h_star, y_star)
        fprime_star_val = fprime_n(t_n + h_star, y_star)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h step
        h_pred = h_estimate(h_star, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val, eps_rel, eps_abs)
        # if predicted h is beyond considered interval [t_0, t_final]
        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n
        # reconstruct approximation y_{n+1} of the 4th order
        rho = h / h_star
        #y_n1 = y(t + h)
        y_n1 = y_n \
               + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
               + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
               + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
               + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_3rd = y_n \
                   + h * (1 - rho ** 2) * f_n_val \
                   + h * rho ** 2 * f_star_val \
                   + h**2 / 2 * (1 - 4 * rho / 3) * fprime_n_val \
                   + h**2 / 2 * (- 2 * rho / 3) * fprime_star_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_3rd)

        print('h* = %4.4e             \t \terr = %4.4e\tlte = %4.4e' % (h_star, err_star, lte_star))
        print('h  = %4.4e rho = %3.2f \t (h_pred = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, rho, h_pred, err, lte))

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))
        err_star_array = np.append(err_star_array, np.array([err_star]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals

def adaptive_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path, test_params):
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
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        h_star = h_star_estimate(y_n, fprime_n_val, eps_rel, eps_abs)
        if t_n + h_star > t_final:
            # correct the step
            h_star = t_final - t_n

        if test_params['middle_step_order'] == 2:
            # reconstruct approximation y_star with the 2nd order tdrk
            y_star = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
        elif test_params['middle_step_order'] == 4:
            # reconstruct approximation y_star with the 4th order tdrk
            t_aux = h_star / 2
            y_aux = y_n + t_aux * f_n_val + t_aux ** 2 / 2 * fprime_n_val
            fprime_aux_val = fprime_n(t_aux, y_aux)
            y_star = y_n \
                     + h_star * f_n_val \
                     + h_star ** 2 / 6 * fprime_n_val \
                     + h_star ** 2 / 3 * fprime_aux_val
            f_evals += 1
        # reconstruct approximation for the 1st order
        y_1st = y_n + h_star * f_n_val

        # analysis of the errors of the h_star step
        err_star = norm(y_star - y(t_n + h_star))
        lte_star = norm(y_star - y_1st)

        # predicted time-step
        const = 1.1
        p = 2
        h_star_new = const * math.pow(math.fabs(eps_abs / lte_star), 1 / (p + 1)) * h_star

        # evaluate f_* and (f')_*
        f_star_val = f_n(t_n + h_star, y_star)
        fprime_star_val = fprime_n(t_n + h_star, y_star)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h step
        h = h_estimate(h_star, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val, eps_rel, eps_abs)
        # if predicted h is beyond considered interval [t_0, t_final]
        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n
        # reconstruct approximation y_{n+1} of the 4th order
        rho = h / h_star
        #y_n1 = y(t + h)
        y_n1 = y_n \
               + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
               + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
               + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
               + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_3rd = y_n \
                   + h * (1 - rho ** 2) * f_n_val \
                   + h * rho ** 2 * f_star_val \
                   + h**2 / 2 * (1 - 4 * rho / 3) * fprime_n_val \
                   + h**2 / 2 * (- 2 * rho / 3) * fprime_star_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_3rd)

        # predicted time-step
        const = 1.1
        p = 4
        h_new = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))
        err_star_array = np.append(err_star_array, np.array([err_star]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        print('h* = %4.4e             \t (h* pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e' \
              % (h_star, h_star_new, err_star, lte_star))
        print('h  = %4.4e rho = %3.2f \t (h  pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e\n' \
              % (h, rho, h_new, err, lte))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals

def adaptive_tdrk_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path, test_params):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # auxiliary functions estimating an intermediate step h_star
    def h_star_estimate(y_n, fprime_n):
        return math.sqrt(2 * (eps_rel * math.fabs(y_n) + eps_abs) / math.fabs(fprime_n))

    #
    def h_estimate(h_star, y_n, f_n, fprime_n, f_star, fprime_star):
        C = math.fabs(  1 / (2 * h_star ** 3) * (f_n - f_star)
                      + 1 / (4 * h_star ** 2) * (fprime_n + fprime_star))
        return math.pow((eps_rel * math.fabs(y_n) + eps_abs) / C, 0.25)

    def norm(val):
        return math.fabs(val)

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        #f_n_val = f_n(y_n)
        #fprime_n_val = fprime_n(y_n)
        #f_evals += 2
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val)
        else:
            #h = h_new
            h = h_pred

        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n

        # reconstruct approximation y_star with the 2nd order tdrk
        y_star = y_n + h / 2 * f_n_val + h ** 2 / 8 * fprime_n_val

        # evaluate f_* and (f')_*
        f_star_val = f_n(t_n + h / 2, y_star)
        fprime_star_val = fprime_n(t_n + h / 2, y_star)
        f_evals += 2

        # reconstruct approximation y_{n+1} of the 4th order
        y_n1 = y_n \
               + h * f_n_val \
               + h**2 / 6 * fprime_n_val \
               + h**2 / 3 * fprime_star_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n \
                   + h * f_n_val \
                   + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_2nd)

        # predicted time-step
        const = 0.9
        p = 4
        if test_params['adaptive_stepping'] == 'our_prediction':
            h_pred = h_estimate(h/2, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val)
        else:
            if lte != 0.0:
                h_pred = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h
            else:
                h_pred = const * h

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        print('h  = %4.4e \t (h_pred = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, h_pred, err, lte))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals


def adaptive_tdrk_5th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path, test_params):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # auxiliary functions estimating an intermediate step h_star
    def h_star_estimate(y_n, fprime_n):
        return math.sqrt(2 * (eps_rel * math.fabs(y_n) + eps_abs) / math.fabs(fprime_n))

    #
    def h_estimate(h_star, y_n, f_n, fprime_n, f_star, fprime_star):
        C = math.fabs(  1 / (2 * h_star ** 3) * (f_n - f_star)
                      + 1 / (4 * h_star ** 2) * (fprime_n + fprime_star))
        return math.pow((eps_rel * math.fabs(y_n) + eps_abs) / C, 0.25)

    def norm(val):
        return math.fabs(val)

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # stage 1
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val)
        else:
            #h = h_new
            h = h_pred

        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n

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

        # predicted time-step
        const = 0.9
        p = 4
        if test_params['adaptive_stepping'] == 'our_prediction':
            h_pred = h_estimate(c_2 * h, y_n, f_n_val, fprime_n_val, f_2_val, fprime_2_val)
        else:
            if lte != 0.0:
                h_pred = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h
            else:
                h_pred = const * h

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        print('h  = %4.4e \t (h_pred = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, h_pred, err, lte))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals