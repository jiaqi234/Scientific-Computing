import numpy as np
import odesolver
from scipy.optimize import fsolve, newton
import shooting
import continuation
import math
import pdesolver
# Copyright: Jiaqi Wei(rf21798@bristol.ac.uk)
def f_true(t):
    # true value of ODE systems with dxdt =y, dydt = -x
    return np.array([np.sin(t)+np.cos(t),np.cos(t)-np.sin(t)])

t1 = np.linspace(0,10,100)
euler = odesolver.solve_ode(odesolver.euler_step,odesolver.f,[1,1],t1,0.005)
rk4 = odesolver.solve_ode(odesolver.rk4_step,odesolver.f,[1,1],t1,0.005)

def period_orbit(t,x,y):
    # periof obrbit given x, y and time value
    xround = np.around(x,4)
    ls,counts = np.unique(xround,return_counts=True)
    xrepeated = ls[np.where(counts==np.max(counts))][0]
    tarray =[]
    for i in range(len(t[np.where(xround == xrepeated)])-1):
        tarray.append(t[np.where(xround == xrepeated)][i+1]-t[np.where(xround== xrepeated)][i])
    time = np.min(tarray)
    xtimevalue = t[np.where(xround == xrepeated)[0][0]]
    return xrepeated,y[np.where(t == xtimevalue)[0]][0],time

t2 = np.linspace(0, 100, 1000)
u0 = np.array([0.1, 0.2, 100])
sol = odesolver.solve_ode(odesolver.rk4_step,shooting.predator_prey,u0[:-1],t2,0.01,[0.4])
orbit = shooting.limit_cycle(fsolve,shooting.predator_prey,shooting.phase_condition,u0,[0.4])

def algebraic_cubic_solution(x,val):
    return fsolve(continuation.algebraic_cubic,x,[val])

npc, x_npc = continuation.natural_parameter_continuation(fsolve,continuation.algebraic_cubic, np.array([1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
pac, x_pac = continuation.pseudo_arclength_continuation(fsolve,continuation.algebraic_cubic, np.array([1]),[2],0,[-2, 2], 200,discretisation=lambda x: x)
truesolnatrual = np.array([algebraic_cubic_solution(x,val) for val,x in list(zip(npc, x_npc))])
truesolpseudo = np.array([algebraic_cubic_solution(x,val) for val,x in list(zip(pac, x_pac))])

def hopf_sol(beta,t,phase):
    return np.array([np.sqrt(beta)*np.cos(t+phase),np.sqrt(beta)*np.sin(t+phase)])

hnpc, hx_npc = continuation.natural_parameter_continuation(fsolve,continuation.hopf, np.array([1,1,1]),[2],0,[2, 0], 30, shooting.shoot,continuation.hopf_phase_condition)
hpac, hx_pac = continuation.pseudo_arclength_continuation(fsolve, continuation.hopf, np.array([1,1,1]),[2],0,[2, 0], 30, shooting.shoot,continuation.hopf_phase_condition)
htruesolnatrual = np.array([hopf_sol(beta,0,phase) for beta,phase in list(zip(hnpc, hx_npc[:,-1]))])
htruesolpseudo = np.array([hopf_sol(beta,0,phase) for beta,phase in list(zip(hpac, hx_pac[:,-1]))])

def f_pde(x,t,k,l):
    return np.exp(-k *(math.pi **2 / l**2 )*t) * np.sin(math.pi*x/l)
k = 0.5
l = 5
t = 2
mx = 10
mt = 100
f_x, f_u = pdesolver.solve_pde('forward', k, l, t, mx, mt,'dirichlet', pdesolver.f, None, pdesolver.left_boundary, pdesolver.right_boundary, fargs=l)
b_x, b_u = pdesolver.solve_pde('backward', k, l, t, mx, mt,'dirichlet', pdesolver.f,None, pdesolver.left_boundary, pdesolver.right_boundary,fargs=l)
c_x, c_u = pdesolver.solve_pde('crank', k, l, t, mx, mt,'dirichlet', pdesolver.f,None, pdesolver.left_boundary, pdesolver.right_boundary, fargs=l)

#Testing
#Compare the true solution with euler method
if np.allclose(f_true(t1),euler,atol=1e-1):
    print("Euler method test passed")
else:
    print("Euler method test failed")
# Compare the true solution with rk4 method
if np.allclose(f_true(t1),rk4,atol=1e-1):
    print("RK4 method test passed")
else:
    print("RK4 method test failed")
#Compare the true solution with shooting method
if np.allclose(period_orbit(t2,sol[0],sol[1])[:-1],orbit[:-1],atol=1e-1):
    print("Shooting method test passed")
else:
    print("Shooting method test failed")
# Compare the true solution with algebraic cubic function using natural parameter ontinuation
if np.allclose(truesolnatrual,x_npc,atol=1e-5):
    print("Algebraic cubic function using natural parameter ontinuation test passed")
else:
    print("Algebraic cubic function using natural parameter ontinuation test failed")
# Compare the true solution with algebraic cubic function using pseudo arclength continuation
if np.allclose(truesolpseudo,x_pac,atol=1e-5):
    print("Algebraic cubic function using pseudo arclength continuation test passed")
else:
    print("Algebraic cubic function using pseudo arclength continuation test failed")
# Compare the true solution with Hopf bifurcation normal form using natural parameter ontinuation
if np.allclose(htruesolnatrual,hx_npc[:,:-1],atol=1e-5):
    print("Hopf bifurcation normal form using natural parameter ontinuation test passed")
else:
    print("Hopf bifurcation normal form using natural parameter ontinuation test failed")
# Compare the true solution with Hopf bifurcation normal form using pseudo arclength continuation
if np.allclose(htruesolpseudo,hx_pac[:,:-1],atol=1e-5):
    print("Hopf bifurcation normal form using pseudo arclength continuation test passed")
else:
    print("Hopf bifurcation normal form using pseudo arclength continuation test failed")

# Compare the ture solution of heat pde equation with forward euler method and dirichlet boundary condition
if np.allclose(f_pde(f_x,t,k,l),f_u,atol=1e-1):
    print("Heat pde equation with forward euler method and dirichlet boundary condition test passed")
else:
    print("Heat pde equation with forward euler method and dirichlet boundary condition test failed")
# Compare the ture solution of heat pde equation with backward euler method and dirichlet boundary condition
if np.allclose(f_pde(b_x,t,k,l),b_u,atol=1e-1):
    print("Heat pde equation with backward euler method and dirichlet boundary condition test passed")
else:
    print("Heat pde equation with backward euler method and dirichlet boundary condition test failed")
# Compare the ture solution of heat pde equation with crank method and dirichlet boundary condition
if np.allclose(f_pde(c_x,t,k,l),c_u,atol=1e-1):
    print("Heat pde equation with crank method and dirichlet boundary condition test passed")
else:
    print("Heat pde equation with crank method and dirichlet boundary condition test failed")