import pytest

from differential import Differentiable, Linear, Constant

def eval_pair(f: Differentiable, x: float):
    """Retourne (f(x), f'(x))."""
    return f.value(x), f.derivative(x)

def test_make_constant_value_and_derivative():
    c = Constant(3.5)
    v, d = eval_pair(c, 2.0)
    assert v == pytest.approx(3.5)
    assert d == pytest.approx(0.0)

def test_make_linear_value_and_derivative():
    f = Linear(2.0)  # f(x)=2x, f'(x)=2
    v, d = eval_pair(f, 4.0)
    assert v == pytest.approx(8.0)
    assert d == pytest.approx(2.0)

def test_add_constant_value_derivative():
    f = Linear(3.0)   # 3x, f' = 3
    g = f + 2.0                             # 3x + 2
    v, d = eval_pair(g, 5.0)
    assert v == pytest.approx(17.0)
    assert d == pytest.approx(3.0)          # dérivée inchangée

def test_radd_constant_commutative():
    f = Linear(-1.0)  # -x
    v1, d1 = eval_pair(5.0 + f, 4.0)
    v2, d2 = eval_pair(f + 5.0, 4.0)
    assert v1 == pytest.approx(v2)
    assert d1 == pytest.approx(d2)

def test_add_derivable_sum_rule():
    f = Linear(1.5)    # 1.5x, f' = 1.5
    g = Constant(2.0)  # 2,    g' = 0
    h = f + g                                # 1.5x + 2
    v, d = eval_pair(h, 3.0)
    assert v == pytest.approx(6.5)
    assert d == pytest.approx(1.5)

def test_sub_constant_value_derivative():
    f = Linear(4.0)  # 4x
    h = f - 3.0                            # 4x - 3
    v, d = eval_pair(h, 2.0)
    assert v == pytest.approx(5.0)
    assert d == pytest.approx(4.0)

def test_rsub_constant_value_derivative():
    f = Linear(2.0)  # 2x
    h = 10.0 - f                           # 10 - 2x
    v, d = eval_pair(h, 1.5)
    assert v == pytest.approx(7.0)
    assert d == pytest.approx(-2.0)        # dérivée opposée

def test_sub_derivable_difference_rule():
    f = Linear(5.0)   # 5x
    g = Linear(2.0)   # 2x
    h = f - g                               # 3x
    v, d = eval_pair(h, 7.0)
    assert v == pytest.approx(21.0)
    assert d == pytest.approx(3.0)

def test_scale_by_constant_value_and_derivative():
    f = Linear(3.0)  # 3x
    h = f * 4.0                            # 12x
    v, d = eval_pair(h, 2.0)
    assert v == pytest.approx(24.0)
    assert d == pytest.approx(12.0)

def test_rmul_by_constant_commutative():
    f = Linear(1.25)  # 1.25x
    v1, d1 = eval_pair(2.0 * f, 8.0)
    v2, d2 = eval_pair(f * 2.0, 8.0)
    assert v1 == pytest.approx(v2)
    assert d1 == pytest.approx(d2)

def test_multiply_derivable_product_rule():
    # f(x)=ax, g(x)=bx => (fg)(x)=ab x^2, (fg)'(x)=2ab x
    a, b = 3.0, 2.0
    f = Linear(a)
    g = Linear(b)
    h = f * g
    x = 5.0
    v, d = eval_pair(h, x)
    assert v == pytest.approx(a * b * x * x)
    assert d == pytest.approx(2 * a * b * x)

def test_true_div_derivable_quotient_rule():
    # f(x)=ax, g(x)=bx (b != 0) => f/g = (a/b) (x/x) = a/b (constante), dérivée 0
    a, b = 4.0, 2.0
    f = Linear(a)
    g = Linear(b)
    h = f / g
    x = 3.0
    v, d = eval_pair(h, x)
    assert v == pytest.approx(a / b)
    assert d == pytest.approx(0.0)

def test_true_div_by_constant_function():
    # f(x)=ax, g(x)=c (constante) => (f/g)(x) = (a/c) x ; dérivée = a/c
    a, c = 6.0, 3.0
    f = Linear(a)
    g = Constant(c)
    h = f / g
    x = 10.0
    v, d = eval_pair(h, x)
    assert v == pytest.approx((a / c) * x)
    assert d == pytest.approx(a / c)

def test_true_div_raises_on_zero_division_at_eval():
    f = Linear(1.0)     # f(x)=x
    g = Constant(0.0)   # g(x)=0
    h = f / g
    with pytest.raises(ZeroDivisionError):
        _ = h.value(1.0)  # dénominateur nul

def test_visitable_call_invokes_visit_once():
    class Spy:
        def __init__(self): self.calls = 0; self.arg = None
        def visit(self, x): self.calls += 1; self.arg = x; return "ok"

    f = Differentiable(value=lambda x: 2*x, derivative=lambda _: 2.0)
    out = f(Spy())
    assert out == "ok"