import math

def y(t):
    return t * math.sin(math.pi * t)

def f(t):
    return math.sin(math.pi * t) + t * math.pi * math.cos(math.pi * t)

def fprime(t):
    return 2 * math.pi * math.cos(math.pi * t) - t * math.pi**2 * math.sin(math.pi * t)
