import math

def y(t):
    return math.exp(-t)

def f(t, y):
    return -y(t)

def fprime(t, y):
    return y(t)

def f_n(tn, yn):
    return -yn

def fprime_n(tn, yn):
    return yn