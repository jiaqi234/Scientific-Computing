import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import root
from math import nan

def derivative(X, t, alpha, beta, delta):
    x, y = X
    dotx = x * (1 - x) - alpha*x*y/(delta+x)
    doty = beta * y * (1 - y/x)
    return np.array([dotx, doty])

alpha = 1
beta = 0.26
delta = 0.1
Nt = 100
tmax = 30
t = np.linspace(0.,tmax, Nt)
X0 = [0.4, 1]
res = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta))
x, y = res.T
plt.figure()
plt.grid()
plt.title("odeint")
plt.plot(t, x, 'xb')
plt.plot(t, y, '+r')
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()

Vval = np.linspace(0, 100, 101)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: derivative(X0, t, alpha, beta, delta)[0], 0)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan

Vval = np.linspace(-60, 20, 81)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: derivative(X0, t, alpha, beta, delta)[1], 20)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan

