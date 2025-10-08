import typing, dataclasses

class _Visitable:
    def __call__(self, visitor: typing.Any) -> typing.Any:
        return visitor.visit(self)

@dataclasses.dataclass
class Differentiable(_Visitable):
    value     : typing.Callable[[float], float]
    derivative: typing.Callable[[float], float]

class _AddConstant:
    def __init__(self, constant: float):
        self.constant = constant

    def visit(self, one: Differentiable) -> Differentiable:
        value = lambda x: one.value(x) + self.constant
        return Differentiable(value=value, derivative=one.derivative)

class _AddDifferentiable:
    def __init__(self, other: Differentiable):
        self.other = other

    def visit(self, one: Differentiable) -> Differentiable:
        value      = lambda x: one.value(x) + self.other.value(x)
        derivative = lambda x: one.derivative(x) + self.other.derivative(x)
        return Differentiable(value=value, derivative=derivative)

class _SubConstant:
    def __init__(self, constant: float):
        self.constant = constant

    def visit(self, one: Differentiable) -> Differentiable:
        value = lambda x: one.value(x) - self.constant
        return Differentiable(value=value, derivative=one.derivative)

class _RSubConstant:
    def __init__(self, constant: float):
        self.constant = constant

    def visit(self, one: Differentiable) -> Differentiable:
        value = lambda x: self.constant - one.value(x)
        derivative = lambda x: -one.derivative(x)
        return Differentiable(value=value, derivative=derivative)

class _SubDifferentiable:
    def __init__(self, other: Differentiable):
        self.other = other

    def visit(self, one: Differentiable) -> Differentiable:
        value      = lambda x: one.value(x) - self.other.value(x)
        derivative = lambda x: one.derivative(x) - self.other.derivative(x)
        return Differentiable(value=value, derivative=derivative)

class _ScaleDifferentiable:
    def __init__(self, scaling: float):
        self.scaling = scaling

    def visit(self, one: Differentiable) -> Differentiable:
        value      = lambda x: self.scaling * one.value(x)
        derivative = lambda x: self.scaling * one.derivative(x)
        return Differentiable(value=value, derivative=derivative)

class _MultiplyDifferentiable:
    def __init__(self, other: Differentiable):
        self.other = other

    def visit(self, one: Differentiable) -> Differentiable:
        value      = lambda x: self.other.value(x) * one.value(x)
        derivative = lambda x: self.other.derivative(x) * one.value(x) + one.derivative(x) * self.other.value(x)
        return Differentiable(value=value, derivative=derivative)

class _TrueDivDifferentiable:
    def __init__(self, other: Differentiable):
        self.other = other

    def visit(self, one: Differentiable) -> Differentiable:
        value      = lambda x: one.value(x) / self.other.value(x)
        derivative = lambda x: (one.derivative(x) * self.other.value(x) - self.other.derivative(x) * one.value(x)) / self.other.value(x)**2
        return Differentiable(value=value, derivative=derivative)

add_dispatch = {
    float    : _AddConstant,
    Differentiable: _AddDifferentiable
}

mul_dispatch = {
    float    : _ScaleDifferentiable,
    Differentiable: _MultiplyDifferentiable
}

sub_dispatch = {
    float    : _SubConstant,
    Differentiable: _SubDifferentiable
}

rsub_dispatch = {
    float    : _RSubConstant,
    Differentiable: _SubDifferentiable
}

def _add_operator(self: Differentiable, other: Differentiable|float) -> Differentiable:
    return self(add_dispatch[type(other)](other))

def _sub_operator(self: Differentiable, other: Differentiable|float) -> Differentiable:
    return self(sub_dispatch[type(other)](other))

def _rsub_operator(self: Differentiable, other: Differentiable|float) -> Differentiable:
    return self(rsub_dispatch[type(other)](other))

def _mul_operator(self: Differentiable, other: Differentiable|float) -> Differentiable:
    return self(mul_dispatch[type(other)](other))

def _truediv_operator(self: Differentiable, other: Differentiable|float) -> Differentiable:
    return self(_TrueDivDifferentiable(other))

Differentiable.__add__     = _add_operator
Differentiable.__radd__    = _add_operator
Differentiable.__sub__     = _sub_operator
Differentiable.__rsub__    = _rsub_operator
Differentiable.__mul__     = _mul_operator
Differentiable.__rmul__    = _mul_operator
Differentiable.__truediv__ = _truediv_operator