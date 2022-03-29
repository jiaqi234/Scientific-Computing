import numpy as np
import matplotlib.pyplot as plt

def f(u,t):
    # 2nd order ODE where dxdt = y , dydt = x
    # u: array, parameter x and y
    # t: array, time value
    # return: array, initial values for ODE
    dxdt = u[1]
    dydt = -u[0]
    return np.array([dxdt,dydt])

def euler_step(f,x0,t0,n,*args):
    # Single euler step to find numerical approximations
    # f: function to be passed
    # x0: array, start step value
    # t0: array, start step time
    # n : float, step size
    # args: array, additional argument to pass to the function
    # return: value of function after single step 
    xn = x0 + n * f(x0,t0,*args)
    tn = t0 + n
    return xn,tn

def rk4_step(f,x0, t0, n,*args):
    # Single 4th Runge Kutta step to find numerical approximations
    # f: function to be passed
    # x0: array, start step value
    # t0: array, start step time
    # n : float, step size
    # args: array, additional argument to pass to the function
    # return: value of function after single step 
    k1 = f(x0,t0,*args)
    k2 = f(x0 + 0.5 * n * k1, t0 + 0.5 * n,*args)
    k3 = f(x0 + 0.5 * n * k2, t0 + 0.5 * n,*args)
    k4 = f(x0 + n * k3, t0 + n,*args)
    xn = x0 + n * (k1 + 2 * k2 + 2 * k3 + k4)/6
    tn = t0 + n
    return xn,tn

def solve_to(method,f,x0,t0,tn,deltat_max,*args):
    # Solves between time interval which steps less than or equal to deltat_max
    # method: function, either euler_step or rk4_step
    # f: function to be passed
    # x0: array, start step value
    # t0: array, start step time
    # tn: array, next step time
    # deltat_max: float, step size
    # args: array, additional argument to pass to the function
    # return: array, x value of function after next step time
    while tn - t0 > deltat_max:
        x0,t0 = method(f,x0,t0,deltat_max, *args)
    else:
        x0,t0 = method(f,x0,t0,tn-t0,*args)
    return x0

def solve_ode(method,f,x0,t,deltat_max,*args):
    # Uses solve_to and methods to generate numerical solution estimates
    # method: function, either euler_step or rk4_step
    # f: function to be passed
    # x0: array, start step value
    # t: array, time value
    # deltat_max: float, step size
    # args: array, additional argument to pass to the function
    # return: array, x value of function in each time step
    x = np.empty(shape = (len(t),len(x0)))
    x[0] = x0
    for i in range (len(t)-1):
        x[i+1] = solve_to(method,f,x[i],t[i],t[i+1],deltat_max, *args)
    return x.transpose()

# t = np.linspace(0,10,100)
# euler = solve_ode(euler_step,f,[1,1],t,0.005)
# rk4 = solve_ode(rk4_step,f,[1,1],t,0.005)
# plt.plot(t, euler[0], label='Euler X')
# plt.plot(t, rk4[0], label='RK4 X')
# plt.plot(t, euler[1], label='Euler Y')
# plt.plot(t, rk4[1], label='RK4 Y')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('x and y')
# plt.show()