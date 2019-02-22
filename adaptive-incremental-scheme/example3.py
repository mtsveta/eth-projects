import math

def y(t):
    return t*math.sin(math.pi*t)

def f(t):
    return y(t)*(1-y(t))

def fy(t):
    return 1-2*y(t)

def fprime(t):
    return f(t)*fy(t)

def f_n(tn, yn):
    return yn*(1-yn)

def fy_n(tn, yn):
    return 1-2*yn

def fprime_n(tn, yn):
    return f_n(yn)*fy_n(yn)