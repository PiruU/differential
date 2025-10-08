import math

from .differentiable import Differentiable

def Sqrt(f: Differentiable) -> Differentiable:
    value = lambda x: math.sqrt(f.value(x))
    derivative = lambda x: 0.5 * f.derivative(x) / math.sqrt(f.value(x))
    return Differentiable(value=value, derivative=derivative)

def Cos(f: Differentiable) -> Differentiable:
    value = lambda x: math.cos(f.value(x))
    derivative = lambda x: -math.sin(f.value(x)) * f.derivative(x)
    return Differentiable(value=value, derivative=derivative)

def Sin(f: Differentiable) -> Differentiable:
    value = lambda x: math.sin(f.value(x))
    derivative = lambda x: math.cos(f.value(x)) * f.derivative(x)
    return Differentiable(value=value, derivative=derivative)

def Constant(k: float) -> Differentiable:
    value, derivative= lambda _: k, lambda _: 0
    return Differentiable(value=value, derivative=derivative)

def Linear(k: float) -> Differentiable:
    value, derivative= lambda x: k * x, lambda _: k
    return Differentiable(value=value, derivative=derivative)