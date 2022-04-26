import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import fsolve,newton,root
import shooting
import pdesolver
import math
from progress.bar import Bar

def algebraic_cubic(x,args):
    # algebraic cubic function
    # x : float, independent variable
    # args: array, additional argument to pass to the function
    # return: float, dependent variable
    return x ** 3 - x + args[0]

def hopf(u,t,args):
    # Hopf bifurcation normal form
    # u: array, u1,u2
    # t: array, time value
    # return: array, equations for ODE
    du1dt = args[0] * u[0] - u[1] - u[0] * (u[0] ** 2 + u[1] ** 2)
    du2dt = u[0] + args[0] * u[1] - u[1] * (u[0] ** 2 + u[1] ** 2)
    return np.array([du1dt, du2dt])

def hopf_phase_condition(u0,args):
    # phase condition for hopf bifurcation normal form
    # u0: array, u
    # args: array, additional argument to pass to the function
    # return: hopf bifurcation normal form value at time 1
    return hopf(u0,1,args)[0]

def modifyhopf(u,t,args):
    # modified hopf bifurcation normal form
    # u0: array, u1, u2
    # t: array, time value
    # args: array, additional argument to pass to the function
    # return: array, equations for ODE
    du1dt = args[0] * u[0] - u[1] + u[0] * (u[0] ** 2 + u[1] ** 2) - u[0] * (u[0] ** 2 + u[1] ** 2) **2
    du2dt = u[0] + args[0] * u[1] + u[1] * (u[0] ** 2 + u[1] ** 2) - u[1] * (u[0] ** 2 + u[1] ** 2) **2
    return np.array([du1dt, du2dt])

def modifyhopf_phase_condiiton(u0,args):
    # phase condition for modified hopf bifurcation normal form
    # u0: array, u
    # args: array, additional argument to pass to the function
    # return: modified hopf bifurcation normal form value at time 1
    return modifyhopf(u0,1,args)[0]

def pde_solve(f,u,args):
    # convert function format to make it compatible with solver method
    return f(u,args)

def heater_equation_pde(u,args):
    # heater pde equations
    k = args[0]
    l = 5
    t = 2
    mx = 10
    mt = 100
    f_x, f_u = pdesolver.solve_pde('forward', k, l, t, mx, mt,'neumann', pdesolver.f,pdesolver.s, pdesolver.left_boundary, pdesolver.right_boundary, fargs=l)
    return f_u

def solve(solver, x0,index,var,discretisation,f,u0,pc):
    # solve the function with discretisation and parameter
    # solver: function, solver to be used (fslove or newton)
    # x0: array, function parameter
    # index: int, index of the parameter in function
    # var: parameter value
    # discretisation: function, discretisation
    # f: function, input function
    # u0: array, initial value of the function
    # pc: function, phase condition
    # return: array,solution of the function
    if not isinstance(index, (int, np.int_)):
        raise TypeError(f"index: {index} is not an integer")
    else:
        if index < 0:
            raise ValueError(f"index: {index} less than 0")
    x0[index] = var
    if pc is None:
        args = x0
    else:
        args = (pc, x0)
    return np.array(solver(discretisation(f),u0,args = args))

def natural_parameter_continuation(solver,f,u0,x0,index,range,num,discretisation,pc = None):
    # natural parameter continuation
    # solver: function, solver to be used (fslove or newton)
    # f: function, input function
    # u0: array, initial state of the function
    # x0: array, function parameters
    # index: int, index of the parameter in function
    # range: array, low bound and upper bound of the parameter
    # num: numer of split for range
    # discretisation: function, discretisation
    # pc: function, phase condition
    # return: array,(parameter value lists,solution of each parameter value)
    sol = []
    with Bar('Loading', fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for var in (np.linspace(range[0],range[1],num)):
            u0 = solve(solver,x0,index,var,discretisation,f,u0,pc)
            sol.append(u0)
            u0 = np.round(u0, 4)
            bar.next()
        bar.finish()
    return np.linspace(range[0],range[1],num),np.array(sol)

def root_finding_arclength(x,f,discretisation,dx,dp,x0,index,pc = None):
    # Find root of the function with pseudo-arclength equation
    # x: array, function solution
    # f: function, input function
    # discretisation: function, discretisation
    # dx: infinitesimal change in x
    # dp: infinitesimal change in parameter value
    # x0: array, function parameters
    # index: int, index of the parameter in function
    # pc: function, phase condition
    # return: array,value of the root finding function
    x0[index] = x[-1]
    xp = x[:-1] + dx
    pp = x[-1] + dp
    if pc is None:
        d = discretisation(f)(x[:-1],x0)
    else:
        d = discretisation(f)(x[:-1],pc,x0)
    return np.append(d,np.dot(x[:-1]-xp,dx) + np.dot(x[-1]-pp,dp))

def pseudo_arclength_continuation(solver,f,u0,x0,index,range,num,discretisation,pc = None):
    # pseudo arclength continuation
    # solver: function, solver to be used (fslove or newton)
    # f: function, input function
    # u0: array, initial state of the function
    # x0: array, function parameters
    # index: int, index of the parameter in function
    # range: array, low bound and upper bound of the parameter
    # num: numer of split for range
    # discretisation: function, discretisation
    # pc: function, phase condition
    # return: array,(parameter value lists,solution of each parameter value)
    sols = []
    para = []
    i0 = np.append(solve(solver,x0,index,range[0],discretisation,f,u0,pc),range[0])
    i1 = np.append(solve(solver,x0,index,range[0] + np.sign(range[1]-range[0])*0.05,discretisation,f,np.round(i0[:-1],2),pc),range[0] + np.sign(range[1]-range[0])*0.05)
    with Bar('Loading', fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
        while True:
            x0[index] = np.append(i1[:-1] + i1[:-1] - i0[:-1], i1[-1]+i1[-1]-i0[-1])[-1]
            sol = root(root_finding_arclength,np.append(i1[:-1] + i1[:-1] - i0[:-1], i1[-1]+i1[-1]-i0[-1]),method = 'lm',args=(f,discretisation, i1[:-1] - i0[:-1], i1[-1]-i0[-1], x0, 0, pc))['x']
            if np.linalg.norm(sol[:-1]) - np.linalg.norm(i1[:-1]) > 0:
                break
            sols.append(sol[:-1])
            para.append(sol[-1])
            i0 = i1
            i1 = sol
            bar.next()
        bar.finish()
    return para, np.array(sols) 

# npc, x_npc = natural_parameter_continuation(fsolve,algebraic_cubic, np.array([1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
# pac, x_pac = pseudo_arclength_continuation(fsolve,algebraic_cubic, np.array([1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('x')
# plt.legend()
# plt.show()

# npc, x_npc = natural_parameter_continuation(fsolve,hopf, np.array([1,1,1]),[2],0,[2, 0], 30, shooting.shoot,hopf_phase_condition)
# pac, x_pac = pseudo_arclength_continuation(fsolve, hopf, np.array([1,1,1]),[2],0,[2, 0], 30, shooting.shoot,hopf_phase_condition)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('x')
# plt.legend()
# plt.show()

# npc, x_npc = natural_parameter_continuation(fsolve,modifyhopf, np.array([1,1,1]),[2],0,[2, -1], 30, shooting.shoot,modifyhopf_phase_condiiton)
# pac, x_pac = pseudo_arclength_continuation(fsolve, modifyhopf, np.array([1,1,1]),[2],0,[2, -1], 30, shooting.shoot,modifyhopf_phase_condiiton)
# plt.plot(npc, scipy.linalg.norm(x_npc, axis=1, keepdims=True)[:, 0], label='Natural parameter continuation')
# plt.plot(pac, scipy.linalg.norm(x_pac, axis=1, keepdims=True)[:, 0], label='Pseudo arclength continuation')
# plt.xlabel('c')
# plt.ylabel('x')
# plt.legend()
# plt.show()

# k = 0.5
# l = 5
# mx = 10
# x = np.linspace(0, l, mx + 1)

# npc, x_npc = natural_parameter_continuation(pde_solve,heater_equation_pde,np.ones(mx + 1), np.array([k]),0,[0.5,5], 5 , discretisation=lambda x: x)
# pac, x_pac = pseudo_arclength_continuation(pde_solve,heater_equation_pde,np.ones(mx + 1), np.array([k]),0,[0.5,5], 5 , discretisation=lambda x: x)


# plt.plot(x, np.transpose(x_npc))
# plt.xlabel('x')
# plt.ylabel(f'u')
# labels = [f"k = {k}" for k in npc]
# plt.legend(labels)
# plt.show()

# plt.plot(x, np.transpose(x_pac))
# plt.xlabel('x')
# plt.ylabel(f'u')
# labels = [f"k = {k}" for k in pac]
# plt.legend(labels)
# plt.show()