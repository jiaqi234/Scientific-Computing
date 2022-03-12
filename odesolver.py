import numpy as np
import matplotlib.pyplot as plt

def f(u,t):
    dxdt = u[1]
    dydt = -u[0]
    return np.array([dxdt,dydt])

def euler_step(f,x0,t0,n):
    xn = x0 + n * f(x0,t0)
    tn = t0 + n
    return xn,tn

def rk4_step(f,x0, t0, n):
    k1 = f(x0,t0)
    k2 = f(x0 + 0.5 * n * k1, t0 + 0.5 * n)
    k3 = f(x0 + 0.5 * n * k2, t0 + 0.5 * n)
    k4 = f(x0 + n * k3, t0 + n)
    xn = x0 + n* (k1 + 2 * k2 + 2 * k3 + k4)/6
    tn = t0 + n
    return xn,tn

def solve_to(method,f,x0,t0,tn,deltat_max):
    while tn - t0 > deltat_max:
        x0,t0 = method(f,x0,t0,deltat_max)
    else:
        x0,t0 = method(f,x0,tn-t0,deltat_max)
    return x0

def solve_ode(method,f,x0,t,deltat_max):
    x = np.empty(shape = (len(t),len(x0)))
    x[0] = x0
    for i in range (len(t)-1):
        x[i+1] = solve_to(method,f,x[i],t[i],t[i+1],deltat_max)
    return x.transpose()

# t = np.linspace(0,20,100)
# euler = solve_ode(euler_step,f,[2,2],t,0.005)
# rk4 = solve_ode(rk4_step,f,[2,2],t,0.005)

# plt.plot(t, euler[0], label='Euler X')
# plt.plot(t, rk4[0], label='RK4 X')
# plt.plot(t, euler[1], label='Euler Y')
# plt.plot(t, rk4[1], label='RK4 Y')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('x and y')
# plt.show()