import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, newton
import odesolver

def predator_prey(x,t,args):
    # predator prey function
    # x: array, x1,x2
    # t: array, time value
    # return: array, equations for ODE
    alpha = 1
    delta = 0.1
    beta = args[0]
    dxdt = x[0] * (1 - x[0]) - alpha*x[0]*x[1]/(delta+x[0])
    dydt = beta * x[1] * (1 - x[1]/x[0])
    return np.array([dxdt, dydt])

def phase_condition(u0,args):
    # phase condition of the function
    # u0: array, x
    # return: function value at time 0
    return predator_prey(u0,0,args)[0]

def shoot(f):
    # define shooting root finding function
    # return: function

    def shooting(u0,pc,*args):
        # u0: array, initial value and time
        # pc: function, phase condition of the input function
        # args: array, additional argument to pass to the function
        # return: array, (difference between initial guess and solution, phase condition)
        t = np.linspace(0,u0[-1],1000)
        sol = odesolver.solve_ode(odesolver.rk4_step,f,u0[:-1],t,0.01,*args)
        return np.append(u0[:-1]- sol[:,-1],pc(u0,*args))
    return shooting

def limit_cycle(solver,f,pc,u0,*args):
    # find the periodic orbit given function
    # solver: function, solver to be used (fslove or newton)
    # f: function to be passed
    # pc: function, phase condition of the input function
    # args: array, additional argument to pass to the function
    return solver(shoot(f),u0,args = (pc,*args))


# t = np.linspace(0, 100, 1000)
# u0 = np.array([0.1, 0.2, 100])
# sol = odesolver.solve_ode(odesolver.rk4_step,predator_prey,u0[:-1],t,0.01,[0.4])
# orbit = limit_cycle(fsolve,predator_prey,phase_condition,u0,[0.4])
# plt.plot(sol[0], sol[1])
# plt.plot(orbit[0], orbit[1], 'go', label="Numerical shooting point")
# plt.legend()
# plt.title("b = 0.4")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

