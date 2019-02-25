from adaptive_schemes import *
from uniform_schemes import *
from plotting import *

import numpy as np

def test_schemes(example_number, test_params):

    result_folder = 'example' + str(example_number)

    if example_number == 1:
        from example1 import *
    elif example_number == 2:
        from example2 import *
    elif example_number == 3:
        from example3 import *

    result_path = create_results_folder(result_folder)

    # order or the scheme
    order = 4

    # given tolerances
    factor  = 1e0
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]) #
    eps_rel = factor * eps_abs
    num_tol = len(eps_abs)
    # given time interval
    t_0   = 0.0
    t_fin = 3.0

    # given initial value
    y_0 = y(t_0)

    # error arrays

    # Test an adaptive scheme
    # -------------------------------------------------------------------------------------------------------------------- #
    lte_adapt = np.zeros(num_tol)
    err_adapt = np.zeros(num_tol)
    n_adapt   = np.zeros(num_tol)
    f_evals_adapt   = np.zeros(num_tol)

    for i in range(0, num_tol):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' integration for eps_abs = %4.4e' % (eps_abs[i]))
        print('% -------------------------------------------------------------------------------------------- %')
        lte_adapt[i], err_adapt[i], n_adapt[i], f_evals_adapt[i] \
            = adaptive_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path, test_params)
    #plot_convergence(err_adapt, n_adapt, f_evals_adapt, 'Our adaptive scheme', 'our-scheme-', result_path)

    # Test an adaptive scheme
    # -------------------------------------------------------------------------------------------------------------------- #
    lte_pred_rej = np.zeros((2, num_tol))
    err_pred_rej = np.zeros((2, num_tol))
    n_pred_rej   = np.zeros((2, num_tol))
    f_evals_pred_rej = np.zeros((2, num_tol))

    test_params['adaptive_stepping'] = 'our_prediction'
    for i in range(0, num_tol):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' integration for eps_abs = %4.4e' % (eps_abs[i]))
        print('% -------------------------------------------------------------------------------------------- %')
        lte_pred_rej[0, i], err_pred_rej[0, i], n_pred_rej[0, i], f_evals_pred_rej[0, i] \
            = adaptive_tdrk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path, test_params)

    for i in range(0, num_tol):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' integration for eps_abs = %4.4e' % (eps_abs[i]))
        print('% -------------------------------------------------------------------------------------------- %')
        lte_pred_rej[1, i], err_pred_rej[1, i], n_pred_rej[1, i], f_evals_pred_rej[1, i] \
            = adaptive_tdrk_5th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path, test_params)

    #plot_convergence_(err_pred_rej, n_pred_rej, f_evals_pred_rej, 'TDRK adaptive scheme', 'tdrk-scheme-our-h-pred-', result_path)

    plot_summary_adaptive_convergence(err_adapt, n_adapt, f_evals_adapt,
                                      err_pred_rej, n_pred_rej, f_evals_pred_rej, result_path,
                                      'Global error', 'global error', 'global-error-')

    plot_summary_adaptive_convergence(lte_adapt, n_adapt, f_evals_adapt,
                                      lte_pred_rej, n_pred_rej, f_evals_pred_rej, result_path,
                                      'Local truncation error', 'loc. trunc. error', 'lte-')

    # Test the two-derivatives Runge-Kutta scheme (4th order)
    # -------------------------------------------------------------------------------------------------------------------- #
    length = t_fin - t_0
    h = length * np.array([#math.pow(2, -2), math.pow(2, -3),
                           math.pow(2, -4), math.pow(2, -5),
                           math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -9)])
    lte_tdrk4 = np.zeros(len(h))
    err_tdrk4 = np.zeros(len(h))
    n_tdrk4   = np.zeros(len(h))
    f_evals_tdrk4   = np.zeros(len(h))

    for i in range(0, len(h)):
        lte_tdrk4[i], err_tdrk4[i], n_tdrk4[i], f_evals_tdrk4[i] \
            = tdrk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)

    # Test the two-derivatives Runge-Kutta scheme (5th order)
    # -------------------------------------------------------------------------------------------------------------------- #
    lte_rk4 = np.zeros(len(h))
    err_rk4 = np.zeros(len(h))
    n_rk4   = np.zeros(len(h))
    f_evals_rk4   = np.zeros(len(h))

    for i in range(0, len(h)):
        err_rk4[i], n_rk4[i], f_evals_rk4[i] \
            = rk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, result_path)

    lte_tdrk5 = np.zeros(len(h))
    err_tdrk5 = np.zeros(len(h))
    n_tdrk5   = np.zeros(len(h))
    f_evals_tdrk5   = np.zeros(len(h))

    for i in range(0, len(h)):
        lte_tdrk5[i], err_tdrk5[i], n_tdrk5[i], f_evals_tdrk5[i] \
            = tdrk_5th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)

    plot_summary_uniform_convergence(err_rk4, n_rk4, f_evals_rk4,
                                     err_tdrk4, n_tdrk4, f_evals_tdrk4,
                                     err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                     'Uniform schemes comparison', result_path)


    plot_summary_adaptive_uniform_convergence(err_adapt, n_adapt, f_evals_adapt,
                                      err_pred_rej, n_pred_rej, f_evals_pred_rej,
                                      err_rk4, n_rk4, f_evals_rk4,
                                      err_tdrk4, n_tdrk4, f_evals_tdrk4,
                                      err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                      'Adaptive & uniform schemes comparison', result_path)

def __main_():

    examples = [1, 2, 3]
    test_params = dict(adaptive_stepping='our_prediction',  # 'classic_prediction'
                       middle_step_order=2)
    for example in range(0, len(examples)):
        test_schemes(examples, test_params)
