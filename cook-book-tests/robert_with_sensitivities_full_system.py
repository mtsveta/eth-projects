import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

from plotting_utilities import *

t0, tT = 0, 4 * 1e3        # considered interval
y0 = np.array([1.0, 0.0, 0.0])   # initial condition

neq = 3
nsens = neq**2

def F(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    s = y[neq:]
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                       0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                         3 * 1e7 * y[1] ** 2,
                     - 0.04 * s[0] +    1e4 * y[2]                   * s[1] + 1e4 * y[1] * s[2],
                       0.04 * s[0] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[1] - 1e4 * y[1] * s[2],
                       0.00 * s[0] +                 6 * 1e7 * y[1]  * s[1] +        0.0 * s[2],
                     - 0.04 * s[3] +    1e4 * y[2]                   * s[4] + 1e4 * y[1] * s[5],
                       0.04 * s[3] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[4] - 1e4 * y[1] * s[5],
                       0.00 * s[3] +                 6 * 1e7 * y[1]  * s[4] +        0.0 * s[5],
                     - 0.04 * s[6] +    1e4 * y[2] *                   s[7] + 1e4 * y[1] * s[8],
                       0.04 * s[6] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[7] - 1e4 * y[1] * s[8],
                       0.00 * s[6] +                 6 * 1e7 * y[1]  * s[7] +        0.0 * s[8]])

num = 1e4
t = np.linspace(t0, tT, num)

print("% ----------------------------------------------------------- %")
print("  Solving the system without Jacobian ")
print("% ----------------------------------------------------------- %")

y, info = odeint(f, y0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for k in range(neq):
    n = k + 1
    plt.plot(t, y[:, k], label=r'$y^{%d}$' % n)
plt.legend()
plt.title('Solutions to the Roberts system')
plt.show()

print("% ----------------------------------------------------------- %")
print("  Solving the system with all sensitivities ")
print("% ----------------------------------------------------------- %")

s0 = np.append(y0, np.ones(nsens))
s, info = odeint(F, s0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for k in range(neq):
    n = k + 1
    plt.plot(t, s[:, k], label=r'$y^{%d}$' % n)
plt.legend()
plt.title('Solutions to the Roberts system')
plt.show()
fig.savefig('approximations.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)


fig, ax = plt.subplots(3, 1, figsize=(6, 12))
for k in  range(nsens):
    n = k + 1
    (j, i) = divmod(n, neq)
    if i == 0: i = neq; j -= 1
    ax[j].plot(t, s[:, k + neq], label=r'$s^{%d, %d}$' % (i, j + 1))
    ax[j].legend()
ax[0].set_title('Sensitivities to the Roberts system')
fig.show()
fig.savefig('sensitivities.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

S = s[:, neq:]
print(S.shape)
S = S.reshape((int(num), neq, neq))
print(S.shape)

deltas  = [0.5*1e-1, 1e-2, 0.5*1e-2, 1e-3]
num_deltas = len(deltas)
fig, ax = plt.subplots(num_deltas, 1, figsize=(6, 12))

for num_delta in range(num_deltas):

    # pertubation of the initial condition
    delta  = deltas[num_delta]
    y0_pert = y0 + delta * np.ones(neq)
    dy0 = y0_pert - y0

    y_pred = y + np.dot(S, dy0)
    #y_new_ = y + np.dot(S_, dy0)

    for k in range(neq):
        ax[num_delta].plot(t, y[:, k],     label=r'$y^{%d}$' % (k + 1),
                 color=colors[2 * k],
                 linestyle=linestyles[0],
                 marker='',
                 markerfacecolor=colors[2 * k],
                 markersize=1)
        ax[num_delta].plot(t, y_pred[:, k], label=r'$y^{%d}_{*}$' % (k + 1),
                           color=colors[2 * k + 1],
                           linestyle=linestyles[1],
                           marker='',
                           markerfacecolor=colors[2 * k + 1],
                           markersize=1)
        ax[num_delta].set_ylabel(r'$\delta = %4.2e$' % delta)
        ax[num_delta].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

ax[0].set_title('Predicted solutions to the Roberts system for different delta')
plt.show()
fig.savefig('predictions.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',
            transparent=True, bbox_inches='tight', pad_inches=0.1)
