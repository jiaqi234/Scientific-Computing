import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import fsolve,root
import shooting

def algebraic_cubic(x,args):
    return x ** 3 - x + args[0]

def hopf(u,t,args):
    du1dt = args[0] * u[0] - u[1] - u[0] * (u[0] ** 2 + u[1] ** 2)
    du2dt = u[0] + args[0] * u[1] - u[1] * (u[0] ** 2 + u[1] ** 2)
    return np.array([du1dt, du2dt])

def hopf_phase_condition(u0,args):
    return hopf(u0,1,args)[0]

def modifyhopf(u,t,args):
    du1dt = args[0] * u[0] - u[1] + u[0] * (u[0] ** 2 + u[1] ** 2) - u[0] * (u[0] ** 2 + u[1] ** 2) **2
    du2dt = u[0] + args[0] * u[1] + u[1] * (u[0] ** 2 + u[1] ** 2) - u[1] * (u[0] ** 2 + u[1] ** 2) **2
    return np.array([du1dt, du2dt])

def modifyhopf_phase_condiiton(u0,args):
    return modifyhopf(u0,1,args)[0]

def solve(x0,index,var,discretisation,f,u0,pc):
    x0[index] = var
    if pc is None:
        args = x0
    else:
        args = (pc, x0)
    return np.array(fsolve(discretisation(f),u0,args = args))

def natural_parameter_continuation(f,u0,x0,index,range,num,discretisation,pc = None):
    sol = []
    for var in (np.linspace(range[0],range[1],num)):
        u0 = solve(x0,index,var,discretisation,f,u0,pc)
        sol.append(u0)
        u0 = np.round(u0, 2)
    return np.linspace(range[0],range[1],num),np.array(sol)

def root_finding_arclength(x,f,discretisation,dx,dp,x0,index,pc = None):
    x0[index] = x[-1]
    xp = x[:-1] + dx
    pp = x[-1] + dp
    if pc is None:
        d = discretisation(f)(x[:-1],x0)
    else:
        d = discretisation(f)(x[:-1],pc,x0)
    return np.append(d,np.dot(x[:-1]-xp,dx) + np.dot(x[-1]-pp,dp))

def pseudo_arclength_continuation(f,u0,x0,index,range,num,discretisation,pc = None):
    sols = []
    para = []
    i0 = np.append(solve(x0,index,range[0],discretisation,f,u0,pc),range[0])
    # x0[index] = range[0] + np.sign(range[1]-range[0])*0.05
    i1 = np.append(solve(x0,index,range[0] + np.sign(range[1]-range[0])*0.05,discretisation,f,u0,pc),range[0] + np.sign(range[1]-range[0])*0.05)
    while True:
        x0[index] = np.append(i1[:-1] + i1[:-1] - i0[:-1], i1[-1]+i1[-1]-i0[-1])[-1]
        sol = root(root_finding_arclength,np.append(i1[:-1] + i1[:-1] - i0[:-1], i1[-1]+i1[-1]-i0[-1]),method = 'lm',args=(f,discretisation, i1[:-1] - i0[:-1], i1[-1]-i0[-1], x0, 0, pc))['x']
        if np.linalg.norm(sol[:-1]) - np.linalg.norm(i1[:-1]) > 0:
            break
        sols.append(sol[:-1])
        para.append(sol[-1])
        i0 = i1
        i1 = sol
    return para, np.array(sols) 

# npc, x_npc = natural_parameter_continuation(algebraic_cubic, np.array([1, 1, 1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
# pac, x_pac = pseudo_arclength_continuation(algebraic_cubic, np.array([1, 1, 1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('||x||')
# plt.legend()
# plt.show()

# npc, x_npc = natural_parameter_continuation(hopf, np.array([1.2,0.5,7]),[2],0,[2, -1], 30, shooting.shoot,hopf_phase_condition)
# pac, x_pac = pseudo_arclength_continuation(hopf, np.array([1.2,0,5.7]),[2],0,[2, -1], 30, shooting.shoot,hopf_phase_condition)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('x')
# plt.legend()
# plt.show()

# npc, x_npc = natural_parameter_continuation(modifyhopf, np.array([1.2,0.5,7]),[2],0,[2, -1], 30, shooting.shoot,modifyhopf_phase_condiiton)
# pac, x_pac = pseudo_arclength_continuation(modifyhopf, np.array([1.2,0,5.7]),[2],0,[2, -1], 30, shooting.shoot,modifyhopf_phase_condiiton)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('x')
# plt.legend()
# plt.show()