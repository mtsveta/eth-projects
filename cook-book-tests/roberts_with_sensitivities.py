import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import time
from plotting_utilities import *

t0, tT = 0, 4 * 1e3             # considered interval
y0 = np.array([1.0, 0.0, 0.0])  # initial condition

neq = 3

time_clas = 0.0
time_sens = 0.0

with_jacobian = True

def Jf(y, t=0):
    """ Jacobian of rhs equations for the Roberts problem"""
    return np.array([[- 0.04,                    1e4 * y[2],   1e4 * y[1]],
                     [  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]],
                     [  0.00,                6 * 1e7 * y[1],         0.0]])
def f(y, t=0):
    """ Rhs equations for the Roberts problem"""
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                  3 * 1e7 * y[1] ** 2])
def F(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    s = y[neq : 2 * neq]
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                       0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                         3 * 1e7 * y[1] ** 2,
                     - 0.04 * s[0] +    1e4 * y[2]                   * s[1] + 1e4 * y[1] * s[2],
                       0.04 * s[0] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[1] - 1e4 * y[1] * s[2],
                       0.00 * s[0] +                 6 * 1e7 * y[1]  * s[1] +        0.0 * s[2]])

def JF(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    s = y[neq : 2 * neq]
    return np.array([[- 0.04,                    1e4 * y[2],   1e4 * y[1], 0, 0, 0],
                     [  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1], 0, 0, 0],
                     [  0.00,                6 * 1e7 * y[1],          0.0, 0, 0, 0],
                     [0, 0, 0, - 0.04,                    1e4 * y[2],   1e4 * y[1]],
                     [0, 0, 0,   0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]],
                     [0, 0, 0,   0.00,                6 * 1e7 * y[1],         0.0]])

num = 10000
t = np.linspace(t0, tT, num)

print("% ----------------------------------------------------------- %")
print("  Solving primal system ")
print("% ----------------------------------------------------------- %")

t_start = time.time()
if with_jacobian:   y, info = odeint(f, y0, t, full_output=True, Dfun=Jf)
else:               y, info = odeint(f, y0, t, full_output=True)
time_clas += time.time() - t_start
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))

for k in range(neq):
    n = k + 1
    plt.plot(t, y[:, k], label=r'$y^{%d}$' % n)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
plt.title('Solutions to the Roberts system')
plt.show()

print("% ----------------------------------------------------------- %")
print("  Solving system with just first n sensitivities ")
print("% ----------------------------------------------------------- %")

s0 = np.append(y0, np.ones(neq))
t_start = time.time()
if with_jacobian:   s, info = odeint(F, s0, t, full_output=True, Dfun=JF)
else:               s, info = odeint(F, s0, t, full_output=True)
time_sens += time.time() - t_start
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))

fig, ax = plt.subplots(2, 1, figsize=(6, 4))
for k in range(neq):
    ax[0].plot(t, s[:, k], label=r'$y^{%d}$' % (k + 1))
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

for k in range(neq, 2 * neq):
    n = k - neq + 1
    (j, i) = divmod(n, neq)
    if i == 0: i = neq; j -= 1
    ax[1].plot(t, s[:, k], label=r'$s^{%d, %d}$' % (i, j + 1) )
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
ax[0].set_title('Solutions and sensitivities of the Roberts system')
plt.show()
fig.savefig('approximations-and-sensitivities-small-system.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

# contructing the sensitivity matrix S \in M^{n x n} from the sensitivity vector s \in R^n
# such that S = [s^{i}, ..., s^n], where each s^i = s
S_ = np.tile(s[:, neq : 2 * neq], (1, neq))
print(S_.shape)
S_ = S_.reshape((int(num), neq, neq))
print(S_.shape)

deltas  = [0.5*1e-1, 1e-2, 0.5*1e-2, 1e-3]
num_deltas = len(deltas)
fig, ax = plt.subplots(num_deltas, 1, figsize=(6, 12))

for num_delta in range(num_deltas):

    # pertubation of the initial condition
    delta  = deltas[num_delta]
    #y0_pert = y0 + delta * np.ones(neq)
    y0_pert = y0 * (1 + delta)  # pertubation by delta percent
    dy0 = y0_pert - y0

    y_pred = y + np.dot(S_, dy0)

    for k in range(neq):
        ax[num_delta].plot(t, y[:, k],     label=r'$y^{%d}$ original' % (k + 1),
                 color=colors[2 * k],
                 linestyle=linestyles[0],
                 marker='',
                 markerfacecolor=colors[2 * k],
                 markersize=1)
        ax[num_delta].plot(t, y_pred[:, k], label=r'$\tilde{y}^{%d}$ predicted' % (k + 1),
                           color=colors[2 * k + 1],
                           linestyle=linestyles[1],
                           marker='',
                           markerfacecolor=colors[2 * k + 1],
                           markersize=1)
        ax[num_delta].set_ylabel(r'$\delta = %2.2e$' % delta)
        ax[num_delta].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

ax[0].set_title('Original and predicted approximations for different delta')
plt.show()
fig.savefig('predictions-small-system.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

print("% ----------------------------------------------------------- %")
print("  Solving the system with perturbed BC ")
print("% ----------------------------------------------------------- %")

fig, ax = plt.subplots(num_deltas, 1, figsize=(6, 12))
for num_delta in range(num_deltas):

    # pertubation of the initial condition
    delta  = deltas[num_delta]
    #y0_pert = y0 + delta * np.ones(neq)
    y0_pert = y0 * (1 + delta) # pertubation by delta percent
    dy0 = y0_pert - y0
    y_pred = y + np.dot(S_, dy0)

    t_start = time.time()
    if with_jacobian:   y_pert, info = odeint(f, y0_pert, t, full_output=True, Dfun=Jf)
    else:               y_pert, info = odeint(f, y0_pert, t, full_output=True)
    time_clas += time.time() - t_start
    print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))

    e_pred = np.zeros(neq)
    for i in range(neq): e_pred[i] = np.linalg.norm(y_pred[:, i] - y_pert[:, i])
    print('delta = %4.4e\te = norm(y - y_pred) = %4.4e' % (delta, np.linalg.norm(e_pred)))

    for k in range(neq):
        ax[num_delta].plot(t, y_pert[:, k],     label=r'$y^{%d}$ approximated' % (k + 1),
                 color=colors[2 * k],
                 linestyle=linestyles[0],
                 marker='',
                 markerfacecolor=colors[2 * k],
                 markersize=1)
        ax[num_delta].plot(t, y_pred[:, k], label=r'$\tilde{y}^{%d}$ predicted' % (k + 1),
                           color=colors[2 * k + 1],
                           linestyle=linestyles[1],
                           marker='',
                           markerfacecolor=colors[2 * k + 1],
                           markersize=1)
        ax[num_delta].set_ylabel(r'$\delta = %2.2e,\;e = %2.2e$' % (delta, np.linalg.norm(e_pred)))
        ax[num_delta].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

ax[0].set_title('Approximated and predicted solutions for different delta')
plt.show()
fig.savefig('comparison-of-approx-and-predictions-small-system.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

print('\ntime_clas = %4.4e\ttime_sens = %4.4e' % (time_clas, time_sens))
