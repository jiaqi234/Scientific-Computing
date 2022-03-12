import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import odesolver

def predator_prey(x,t):
    alpha = 1
    delta = 0.1
    beta = 0.2
    dxdt = x[0] * (1 - x[0]) - alpha*x[0]*x[1]/(delta+x[0])
    dydt = beta * x[1] * (1 - x[1]/x[0])
    return np.array([dxdt, dydt])

def phase_condition(u0):
    return predator_prey(u0,0)[0]

def shoot(f):
    def shooting(u0):
        t = np.linspace(0,u0[-1],50)
        sol = odesolver.solve_ode(odesolver.rk4_step,f,u0[:-1],t,0.01)
        return np.append(u0[:-1]- sol[:,-1],phase_condition(u0[:-1]))
    return shooting

def limit_cycle(f,u0):
    return fsolve(shoot(f),u0)

t = np.linspace(0, 50, 50)
u0 = np.array([0.1,0.15,20])
sol = odesolver.solve_ode(odesolver.rk4_step,predator_prey,u0[:-1],t,0.01)
orbit = limit_cycle(predator_prey,u0)
plt.plot(sol[0], sol[1])
plt.plot(orbit[0], orbit[1], 'go', label="Numerical shooting point")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()