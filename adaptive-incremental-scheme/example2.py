import math

def y(t):
    return math.exp(-10*t)

def f(t):
    return -10*y(t)

def fprime(t):
    return 100*y(t)

def f_n(tn, yn):
    return -10*yn

def fprime_n(tn, yn):
    return 100*yn