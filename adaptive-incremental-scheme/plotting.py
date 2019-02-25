import matplotlib.pyplot as plt
import math
import numpy as np
import os, sys

def get_project_path():
    (project_path, src_folder) = os.path.split(os.path.abspath(os.path.dirname(__file__)))
    return project_path + '/' + src_folder

def create_results_folder(directory):
    # create the full directory of the result folder
    full_directory = get_project_path() + '/results/' + directory

    # if directory doesn't exist
    if not os.path.exists(full_directory):
        os.makedirs(full_directory)
    # writing to the log file
    # sys.stdout = open(full_directory + '/log-results.txt', "w+")
    return full_directory

def plot_results(time, approx, exact, h, err, result_path):

    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(3, 1, figsize=(6, 12))
    ax[0].plot(time, approx,
               color='orchid',
               linestyle='dashed',
               marker='',
               markerfacecolor='orchid',
               markersize=6,
               label=r'$y_n$')
    ax[0].plot(time, exact,
               color='coral',
               linestyle='',
               marker='o',
               markerfacecolor='coral',
               markersize=2,
               label=r'$y$')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$y_n, y$')
    #plt.title('Comparison of y_n to exact y(t)')
    ax[0].set_title(r'Comparison of $y_n$ to exact $y(t)$')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # time-steps wrt time
    ax[1].plot(time[1:], h,
               color='coral',
               linestyle='',
               marker='*',
               markerfacecolor='tan',
               markersize=2,
               label=f'$h$ by out approach')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$h$')
    ax[1].set_title('Predicted $h$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # time-steps wrt time
    ax[2].semilogy(time[1:],  err,
                   color='red',
                   linestyle=':',
                   marker='',
                   markerfacecolor='tan',
                   markersize=6,
                   label=r'$e$')
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('$e = \|y_{n+1} - y(t_{final})\|$')
    ax[2].set_title('Error')
    ax[2].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + ('/adaptive-scheme-approx-error-%d.eps' %(len(time))), dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1,
            frameon=None, metadata=None)

def plot_uniform_results(time, approx, exact, err, result_path):
    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].plot(time, approx, color='green', linestyle='dashed', marker='', markerfacecolor='green', markersize=6,
               label=f'y_n')
    ax[0].plot(time, exact, color='blue', linestyle='', marker='o', markerfacecolor='blue', markersize=6, label=f'y')
    # ax[0].plot(time, approx, '--', mew=1, ms=4, xmec='w', label=f'y_n')
    # ax[0].plot(time, exact, 'o-', mew=1, ms=4, mec='w', label=f'y')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.xlabel('time, t')
    plt.ylabel('y_n, y')
    # plt.title('Comparison of y_n to exact y(t)')
    ax[0].set_title('Comparison of y_n to exact y(t)')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # time-steps wrt time
    ax[1].semilogy(time[1:], err, color='red', linestyle=':', marker='', markerfacecolor='tan', markersize=6,
                   label=r'$log_10(err)$')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.xlabel('time, t')
    plt.ylabel('log_10(err)')
    ax[1].set_title('Error')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    plt.savefig(result_path + '/tdrk-approx-error-%d.eps' %(len(time)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1,
                frameon=None, metadata=None)

def plot_convergence(err, n, f_evals, title, tag, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n, err,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[0].loglog(n, np.power(n, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order conv.')
    ax[0].loglog(n, np.power(n, -3),
                 color='gray',
                 linestyle='dashed',
                 markerfacecolor='gray',
                 label=r'$3$th order conv.')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(r'$e$')
    ax[0].set_title('Errors vs. number of steps / func. evaluations')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals, err,
                 color='blue',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + tag + 'adaptive-convergence-%d.eps' % (len(err)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, pad_inches=0.1)

def plot_convergence_(err, n, f_evals, title, tag, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n[0, :], err[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'4th tdrk')
    ax[0].loglog(n[1, :], err[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'5th tdrk')
    ax[0].loglog(n[1, :], np.power(n[1, :], -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=r'$3$th order conv.')
    ax[0].loglog(n[1, :], np.power(n[1, :], -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order conv.')
    ax[0].loglog(n[1, :], np.power(n[1, :], -5),
                 color='lavender',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='lavender',
                 markersize=6,
                 label=r'$5$th order conv.')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'$e$')
    ax[0].set_title('Errors vs. number of steps / funcs. evals')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals[0, :], err[0, :],
                 color='green',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'4th tdrk')
    ax[1].loglog(f_evals[1, :], err[1, :],
                 color='blue',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'5th tdrk')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + tag + 'adaptive-convergences-%d.eps' % (len(err)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)

def plot_uniform_convergence(err, n, f_evals, title, tag, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].loglog(n, err,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e$')
    ax[0].loglog(n, np.power(n, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order scheme')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel('$e = \|y_{n+1} - y(t_{final})\|$')
    ax[0].set_title('Errors vs. number of steps')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals, err,
                 color='blue',
                 linestyle='',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'$e$')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    plt.xlabel('func. evaluations')
    plt.ylabel('$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].set_title('Errors vs. function evaluation')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + tag + 'uniform-convergence-%d.eps' % (len(err)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)

def plot_summary_adaptive_convergence(err_our, n_our, f_evals_our,
                                      err_tdrk, n_tdrk, f_evals_tdrk,
                                      result_path, title, y_axis_name, save_tag):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n_our, err_our,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=f'our method')
    ax[0].loglog(n_tdrk[0, :], err_tdrk[0, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=f'4th tdrk')

    ax[0].loglog(n_tdrk[1, :], err_tdrk[1, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=f'5th tdrk')
    ax[0].loglog(n_our, np.power(n_our, -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=f'3rd order conv.')
    ax[0].loglog(n_our, np.power(n_our, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order conv.')
    ax[0].loglog(n_our, np.power(n_our, -5),
                 color='lavender',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='lavender',
                 markersize=6,
                 label=f'5th order conv.')


    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(y_axis_name)
    ax[0].set_title(title + ' vs. number of steps / funct. evaluation')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals_our, err_our,
                 color='green',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'our method')
    ax[1].loglog(f_evals_tdrk[0, :], err_tdrk[0, :],
                 color='blue',
                 linestyle=':',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'4th tdrk')
    ax[1].loglog(f_evals_tdrk[1, :], err_tdrk[1, :],
                 color='darkred',
                 linestyle=':',
                 marker='x',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'5th tdrk')

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(y_axis_name)
    #ax[1].set_title('Errors vs. function evaluation')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + save_tag + 'adaptive-convergence-summary-our-%d-tdrk-%d.eps'
                % (len(err_our), len(err_tdrk)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)


def plot_summary_uniform_convergence(err_rk, n_rk, f_evals_rk,
                                     err_tdrk4, n_tdrk4, f_evals_tdrk4,
                                     err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                     title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n_rk, err_rk,
                 color='orchid',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=f'4th rk')
    ax[0].loglog(n_tdrk4, err_tdrk4,
                 color='coral',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='coral',
                 markersize=6,
                 label=f'4th tdrk')

    ax[0].loglog(n_tdrk5, err_tdrk5,
                 color='purple',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='purple',
                 markersize=6,
                 label=f'5th tdrk')

    ax[0].loglog(n_tdrk4, np.power(n_tdrk4, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order conv.')
    ax[0].loglog(n_tdrk4, np.power(n_tdrk4, -5),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=f'5th order conv.')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[0].set_title('Errors vs. number of steps')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals_rk, err_rk,
                 color='orchid',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=r'4th rk')
    ax[1].loglog(f_evals_tdrk4, err_tdrk4,
                 color='coral',
                 linestyle=':',
                 marker='^',
                 markerfacecolor='coral',
                 markersize=6,
                 label=r'4th tdrk')
    ax[1].loglog(f_evals_tdrk5, err_tdrk5,
                 color='purple',
                 linestyle=':',
                 marker='+',
                 markerfacecolor='purple',
                 markersize=6,
                 label=r'5th tdrk')

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/adaptive-convergence-summary-our-%d-tdrk-%d.eps' % (len(err_rk), len(err_tdrk4)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)



def plot_summary_adaptive_uniform_convergence(err_our, n_our, f_evals_our,
                                      err_pred_rej, n_pred_rej, f_evals_pred_rej,
                                      err_rk, n_rk, f_evals_rk,
                                      err_tdrk, n_tdrk, f_evals_tdrk,
                                      err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                      title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n_our, err_our,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=f'our method')
    ax[0].loglog(n_pred_rej[0, :], err_pred_rej[0, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=f'adaptive 4th tdrk')
    ax[0].loglog(n_pred_rej[1, :], err_pred_rej[1, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=f'adaptive 5th tdrk')

    ax[0].loglog(n_rk, err_rk,
                 color='orchid',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=f'4th rk')
    ax[0].loglog(n_tdrk, err_tdrk,
                 color='coral',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='coral',
                 markersize=6,
                 label=f'4th tdrk')
    ax[0].loglog(n_tdrk5, err_tdrk5,
                 color='purple',
                 linestyle=':',
                 marker='+',
                 markerfacecolor='purple',
                 markersize=6,
                 label=f'4th tdrk')

    ax[0].loglog(n_tdrk, np.power(n_tdrk, -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=f'3rd order conv.')

    ax[0].loglog(n_tdrk, np.power(n_tdrk, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order conv.')
    ax[0].loglog(n_tdrk, np.power(n_tdrk, -5),
                 color='lavender',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='lavender',
                 markersize=6,
                 label=f'5th order conv.')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(r'$e= \|y_{n+1} - y(t_{final})\|$')
    ax[0].set_title('Errors vs. number of steps / func. evaluation')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation

    ax[1].loglog(f_evals_our, err_our,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=f'our method')
    ax[1].loglog(f_evals_pred_rej[0, :], err_pred_rej[0, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=f'adaptive 4th tdrk')
    ax[1].loglog(f_evals_pred_rej[1, :], err_pred_rej[1, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=f'adaptive 5th tdrk')
    ax[1].loglog(f_evals_rk, err_rk,
                 color='orchid',
                 linestyle=':',
                 marker='d',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=r'4th rk')
    ax[1].loglog(f_evals_tdrk, err_tdrk,
                 color='coral',
                 linestyle=':',
                 marker='*',
                 markerfacecolor='coral',
                 markersize=6,
                 label=r'4th tdrk')
    ax[1].loglog(f_evals_tdrk5, err_tdrk5,
                 color='purple',
                 linestyle=':',
                 marker='+',
                 markerfacecolor='purple',
                 markersize=6,
                 label=r'5th tdrk')

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    #ax[1].set_title('Errors vs. function evaluation')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/convergence-summary-adaptive-%d-uniform-%d.eps' % (len(err_our), len(err_tdrk)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)

