import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import odesolver

def hopf(u0,t,args):
    du1dt = args[0]*u0[0]-u0[1]+ (u0[0]**2+u0[1]**2) * args[1]
    du2dt = u0[0] + args[0]*u0[1]+(u0[0]**2+u0[1]**2)* args[1]
    du3dt = -u0[2]
    return np.array([du1dt,du2dt,du3dt])


def hopf_sol(t,phase,args):
    u1 = np.sqrt(args[0]) * np.cos(t + phase)
    u2 = np.sqrt(args[0]) * np.sin(t + phase)
    u3 = np.exp(-(t + phase))
    return np.array([u1, u2, u3])

def shoot(f):
    def shooting(u0,pc,*args):
        t = np.linspace(0,u0[-1],1000)
        sol = odesolver.solve_ode(odesolver.rk4_step,f,u0[:-1],t,0.01,*args)
        return np.append(u0[:-1]- sol[:,-1],pc(u0,*args))
    return shooting

def limit_cycle(f,pc,u0,*args):
    return fsolve(shoot(f),u0,args = (pc,*args))

def phase_condition(u0,args):
    return hopf(u0,1,args)[0]

t = np.linspace(0, 20, 50)
u0 = np.array([1.2, 0.3, 1.2, 10])
sol = odesolver.solve_ode(odesolver.rk4_step,hopf,u0[:-1],t,0.01,[2,-1])
orbit = limit_cycle(hopf,phase_condition,u0,[2,-1])
ax = plt.axes(projection='3d')
ax.plot(sol[0],sol[1],sol[2])
ax.scatter(orbit[0], orbit[1],orbit[2],'go', label="Numerical shooting point")
plt.show()

if np.allclose(hopf_sol(0,orbit[-1],[2,-1]),orbit[:-1]):
    print("Test passed")
else:
    print("Test failed")

