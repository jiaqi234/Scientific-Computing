import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import odesolver

def predator_prey(x,t,args):
    alpha = 1
    delta = 0.1
    beta = args[0]
    dxdt = x[0] * (1 - x[0]) - alpha*x[0]*x[1]/(delta+x[0])
    dydt = beta * x[1] * (1 - x[1]/x[0])
    return np.array([dxdt, dydt])

def phase_condition(u0,args):
    return predator_prey(u0,0,args)[0]

def shoot(f):
    def shooting(u0,pc,*args):
        t = np.linspace(0,u0[-1],1000)
        sol = odesolver.solve_ode(odesolver.rk4_step,f,u0[:-1],t,0.01,*args)
        return np.append(u0[:-1]- sol[:,-1],pc(u0,*args))
    return shooting

def limit_cycle(f,pc,u0,*args):
    return fsolve(shoot(f),u0,args = (pc,*args))

# t = np.linspace(0, 1000, 1000)
# u0 = np.array([0.07, 0.16, 23])
# sol = odesolver.solve_ode(odesolver.rk4_step,predator_prey,u0[:-1],t,0.01,[0.2])
# orbit = limit_cycle(predator_prey,phase_condition,u0,[0.2])
# plt.plot(sol[0], sol[1])
# plt.plot(orbit[0], orbit[1], 'go', label="Numerical shooting point")
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()