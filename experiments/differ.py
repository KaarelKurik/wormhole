import numpy as np

def sphery(v):
    return v[0]*v[0] + 3*v[1]*v[1] - 1.0

def grad(v):
    return np.array([2*v[0], 6*v[1]])

def hessian(v):
    return np.array([[2.0, 0.0],[0.0, 6.0]])

def boxy(bulk):
    v = bulk[1:]
    return np.block([[np.zeros((1,1)), grad(v).reshape(1,2)], [grad(v).reshape(2,1), bulk[0]*hessian(v) + np.identity(2)]])

def bulk_func(x, bulk):
    return np.array([sphery(bulk[1:]), *(bulk[0]*grad(bulk[1:])-x+bulk[1:])])

def true_newton_step(x, bulk):
    return bulk - np.linalg.inv(boxy(bulk)) @ bulk_func(x,bulk)

def step(x,n,l):
    spot = x-l*n
    new_n = grad(spot)
    new_l = l + sphery(spot)/np.dot(n,n)
    return new_n, new_l

def different_step(x,n,l):
    spot = x-l*n
    new_n = grad(spot)
    new_l = l + sphery(spot)/np.dot(new_n, new_n)
    return new_n, new_l

def do_thing(start, ksteps):
    print("thing")
    n = grad(start)
    l = sphery(start)/np.dot(n,n)
    for i in range(1,ksteps):
        print(i, n, l)
        n,l = step(start, n, l)
    print(i+1, n, l)
    return n,l

def other_thing(start, ksteps):
    print("other thing")
    n = grad(start)
    l = sphery(start)/np.dot(n,n)
    for i in range(1,ksteps):
        print(i, n, l)
        n,l = different_step(start, n, l)
    print(i+1, n, l)
    return n,l

def more_direct_step(x,y,n):
    new_y = x - (sphery(x)/(np.dot(grad(x),n)))*grad(x)
    new_n = grad(new_y)
    return new_y, new_n

def fourth_step(x,y,l):
    gy = grad(y)
    new_y = x - l*gy
    new_l = l + sphery(new_y)/np.dot(grad(new_y), gy)
    return new_y, new_l

def fourth_thing(start, ksteps):
    print("fourth")
    y = start
    l = 0.0
    for i in range(1,ksteps):
        print(i, y, l)
        y,l = fourth_step(start,y,l)
    print(i+1, y, l)
    return y,l

def direct_thing(start, ksteps):
    print("direct")
    start_grad = grad(start)
    y = start - start_grad/np.dot(start_grad, start_grad)
    n = grad(y)
    for i in range(1,ksteps):
        print(i, y, n)
        y,n = more_direct_step(start,y,n)
    print(i+1, y, n)
    return y,n


def newton_thing(start, ksteps):
    print("newton")
    start_grad = grad(start)
    y = start
    l = 1/np.dot(start_grad, start_grad)
    bulk = np.array([l,*y])
    for i in range(1,ksteps):
        print(i, bulk)
        bulk = true_newton_step(start,bulk)
    print(i+1, bulk)
    return bulk
    

start = np.array([1.1, 0.1])
do_thing(start, 32)
other_thing(start, 32)
bulky = newton_thing(start, 32)
cunky = fourth_thing(start, 32)
