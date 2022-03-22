import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import math
def sparse_matrix(size,d1,d2,d3):
    return scipy.sparse.diags(np.array([d1*np.ones(size-1),d2*np.ones(size),d3*np.ones(size-1)],dtype = object),[-1,0,1],format = 'csr')

def methods(method,mtsize,lam):
    if method == "forward":
        mt1 = sparse_matrix(mtsize,0,1,0)
        mt2 = sparse_matrix(mtsize,lam,1-2*lam,lam)
    elif method == "backward":
        mt1 = sparse_matrix(mtsize,-lam,1+2*lam,-lam)
        mt2 = sparse_matrix(mtsize,0,1,0)
    elif method == "crank":
        mt1 = sparse_matrix(mtsize,-lam/2,1+lam,-lam/2)
        mt2 = sparse_matrix(mtsize,lam/2,1-lam,lam/2)
    return mt1,mt2

def args(f,args):
    def convert(x,t):
        return f(x,t,args)
    return convert

def solve_pde(method,k,l,t,mx,mt,boundarytype,f,s,left_boundary,right_boundary,fargs = None):
    x = np.linspace(0,l,mx+1)
    t = np.linspace(0,t,mt+1)
    lam = k*(t[1] - t[0])/ (x[1]-x[0])** 2
    if fargs != None:
        f = args(f,fargs)
    if boundarytype == "dirichlet":
        mtsize = mx -1
        uj = f(x[1:mx],0)
        mt1,mt2 = methods(method,mtsize,lam)
        for i in range(mt):
            ti = t[i]
            if s != None:
                svalue = s(x[1:mx],ti)
            else:
                svalue = 0
            vec = np.zeros(mtsize)
            vec[0] = left_boundary(0,ti)
            vec[-1] = right_boundary(l,ti)
            vec = vec *lam + (t[1] - t[0])*svalue
            uj = scipy.sparse.linalg.spsolve(mt1,mt2*uj+vec)
        uj = np.concatenate(([left_boundary(0,t)],uj,[right_boundary(l,t)]))
    elif boundarytype == "periodic":
        mtsize = mx
        uj = np.append(f(x[:mx-1],0),f(x[:mx-1],0)[-1])
        mt1,mt2 = methods(method,mtsize,lam)
        mt1[0, mtsize - 1] = mt1[0, 1]
        mt1[mtsize - 1, 0] = mt1[0, 1]
        mt2[0, mtsize - 1] = mt2[0, 1]
        mt2[mtsize - 1, 0] = mt2[0, 1]
        for i in range(mt):
            ti = t[i]
            if s != None:
                svalue = np.append(s(x[:mx-1],t),s(x[:mx-1],t)[-1])
            else:
                svalue = 0
            vec = (t[1] - t[0])*svalue
            uj = scipy.sparse.linalg.spsolve(mt1,mt2*uj+vec)
        uj = np.append(uj,uj[0])
    elif boundarytype == "neumann":
        mtsize = mx +1
        uj = f(x,0)
        mt1,mt2 = methods(method,mtsize,lam)
        mt1[0, 1] = mt1[0, 1] * 2
        mt1[mtsize - 1, mtsize - 2] = mt1[mtsize - 1, mtsize - 2 ] * 2
        mt2[0, 1] = mt2[0, 1] * 2
        mt2[mtsize - 1, mtsize - 2] = mt2[mtsize - 1, mtsize - 2] * 2
        for i in range(mt):
            ti = t[i]
            if s != None:
                svalue = s(x,ti)
            else:
                svalue = 0
            vec = np.zeros(mtsize)
            vec[0] = -left_boundary(0,ti)
            vec[-1] = right_boundary(l,ti)
            vec = 2 * vec *lam *(t[1] - t[0]) + (t[1] - t[0])*svalue
            uj = scipy.sparse.linalg.spsolve(mt1,mt2*uj+vec)
    return x, uj

# k = 1
# l = 2
# t = 0.5
# mx = 100
# mt = 1000

# def left_boundary(x,t):
#     return 0

# def right_boundary(x,t):
#     return 1

# def f(x,t,l):
#     return np.sin(math.pi*x/l)

# def s(x,t):
#     return x+t

# f_x, f_u = solve_pde('forward', k, l, t, mx, mt,'dirichlet', f, None, left_boundary, right_boundary, fargs=l)
# b_x, b_u = solve_pde('backward', k, l, t, mx, mt,'dirichlet', f,None, left_boundary, right_boundary,fargs=l)
# c_x, c_u = solve_pde('crank', k, l, t, mx, mt,'dirichlet', f,None, left_boundary, right_boundary, fargs=l)

# plt.plot(f_x, f_u, label='forward')
# plt.plot(b_x, b_u, label='backward')
# plt.plot(c_x, c_u, label='crank')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('u(x,0.5)')
# plt.show()


# f_x, f_u = solve_pde('forward', k, l, t, mx, mt,'neumann', f, s, left_boundary, right_boundary, fargs=l)
# b_x, b_u = solve_pde('backward', k, l, t, mx, mt,'neumann', f,s, left_boundary, right_boundary,fargs=l)
# c_x, c_u = solve_pde('crank', k, l, t, mx, mt,'neumann', f,s, left_boundary, right_boundary, fargs=l)

# plt.plot(f_x, f_u, label='forward')
# plt.plot(b_x, b_u, label='backward')
# plt.plot(c_x, c_u, label='crank')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('u(x,0.5)')
# plt.show()