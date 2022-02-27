def deltax (xn,x0,n):
    return (xn - x0)/n

def f(x):
    return x

def euler(x0,y0,xn,n):
    for i in range(n):
        yn = y0 + deltax(xn,x0,n) * f(x0)
        print("Step "+ str(i) + str((x0,y0,deltax(xn,x0,n),yn)))
        y0 = yn
        x0 = x0 + deltax(xn,x0,n)
    print(xn,yn)

def dydx(x,y):
    return 1

def RK4(x0, y0, xn, h):
    n = int((xn - x0)/h)
    y = y0
    for i in range (n + 1):
        k1 = h * dydx(x0,y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h,y + 0.5 * k2)
        k4 = h * dydx(x0 + h,y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4)/6
        x0 = x0 + h
    return y

x0 = 0
y0 = 1
xn = 10
n = 101
h = 0.2

euler(x0,y0,xn,n)
print(RK4(x0, y0, xn, h))