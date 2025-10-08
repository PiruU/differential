import math
import pytest

from differential import Differentiable, Sqrt, Sin, Cos, Constant, Linear

def eval_pair(f: Differentiable, x: float):
    """Retourne (f(x), f'(x))."""
    return f.value(x), f.derivative(x)

def test_sqrt_of_positive_constant():
    # f(x) = c > 0  => sqrt(f) = sqrt(c), (sqrt(f))' = 0
    c = 9.0
    f = Constant(c)
    g = Sqrt(f)
    v, d = eval_pair(g, 2.0)
    assert v == pytest.approx(math.sqrt(c))
    assert d == pytest.approx(0.0)

def test_sqrt_linear_chain_rule():
    # f(x) = a x + b, supposé > 0 au point testé
    # (sqrt f)'(x) = 0.5 * f'(x) / sqrt(f(x)) = 0.5 * a / sqrt(a x + b)
    a, b = 3.0, 5.0
    f = Differentiable(
        value=lambda x: a * x + b,
        derivative=lambda _: a,
    )
    g = Sqrt(f)
    x = 7.0
    v, d = eval_pair(g, x)
    assert v == pytest.approx(math.sqrt(a * x + b))
    assert d == pytest.approx(0.5 * a / math.sqrt(a * x + b))

def test_sqrt_of_quadratic_positive_x():
    # f(x) = (a x)^2 => sqrt(f(x)) = |a x|
    # Pour x > 0, sqrt(f) = a x, dérivée = a
    a = 2.5
    f = Differentiable(
        value=lambda x: (a * x) ** 2,
        derivative=lambda x: 2 * (a * x) * a,  # 2 a^2 x
    )
    g = Sqrt(f)
    x = 3.0  # > 0
    v, d = eval_pair(g, x)
    assert v == pytest.approx(a * x)
    assert d == pytest.approx(a)

def test_sqrt_of_quadratic_negative_x():
    # Pour x < 0, sqrt(f) = |a x| = -a x (car a>0), dérivée = -a
    a = 1.75
    f = Differentiable(
        value=lambda x: (a * x) ** 2,
        derivative=lambda x: 2 * (a * x) * a,  # 2 a^2 x
    )
    g = Sqrt(f)
    x = -4.0  # < 0
    v, d = eval_pair(g, x)
    assert v == pytest.approx(-a * x)  # |a x| = -a x si x<0
    # Dérivée de |a x| vaut a * sign(x) ; ici sign(x) = -1
    assert d == pytest.approx(-a)

def test_sqrt_raises_on_negative_value():
    # f(x) = -1  => sqrt(f(x)) doit lever ValueError à l'évaluation
    f = Constant(-1.0)
    g = Sqrt(f)
    with pytest.raises(ValueError):
        _ = g.value(0.0)

def test_sqrt_derivative_raises_when_f_is_zero():
    # f(x) = x^2 : f(0)=0 et f'(0)=0, donc la dérivée 0.5*f'/sqrt(f) est indéfinie en x=0 (division par zéro)
    f = Differentiable(
        value=lambda x: x * x,
        derivative=lambda x: 2 * x,
    )
    g = Sqrt(f)
    # La valeur en 0 est bien 0
    assert g.value(0.0) == pytest.approx(0.0)
    # Mais la dérivée en 0 est singulière (ZeroDivisionError attendu)
    with pytest.raises(ZeroDivisionError):
        _ = g.derivative(0.0)

def test_sqrt_respects_visitable_semantics_indirectly():
    # Ce test s'assure que le Differentiable renvoyé reste bien "évaluable"
    # via ses callables value/derivative, ce qui est l'essentiel ici.
    f = Differentiable(
        value=lambda x: 16.0 + 2.0 * x,
        derivative=lambda _: 2.0,
    )
    g = Sqrt(f)
    x = 9.0
    # simple cohérence valeur/dérivée par différence finie
    v = g.value(x)
    d = g.derivative(x)
    eps = 1e-6
    fd = (g.value(x + eps) - g.value(x - eps)) / (2 * eps)
    assert v == pytest.approx(math.sqrt(16.0 + 2.0 * x))
    assert d == pytest.approx(fd, rel=1e-6, abs=1e-8)

def test_cos_of_constant():
    f = Constant(2.0)
    g = Cos(f)
    v, d = eval_pair(g, 1.23)
    assert v == pytest.approx(math.cos(2.0))
    assert d == pytest.approx(0.0)

def test_cos_of_linear():
    a, b = 2.0, 0.5
    f = Differentiable(value=lambda x: a * x + b, derivative=lambda _: a)
    g = Cos(f)
    x = 1.1
    v, d = eval_pair(g, x)
    assert v == pytest.approx(math.cos(a * x + b))
    assert d == pytest.approx(-math.sin(a * x + b) * a)

def test_sin_of_constant():
    f = Constant(-1.0)
    g = Sin(f)
    v, d = eval_pair(g, 3.21)
    assert v == pytest.approx(math.sin(-1.0))
    assert d == pytest.approx(0.0)

def test_sin_of_linear():
    a, b = -1.5, 0.25
    f = Differentiable(value=lambda x: a * x + b, derivative=lambda _: a)
    g = Sin(f)
    x = -0.7
    v, d = eval_pair(g, x)
    assert v == pytest.approx(math.sin(a * x + b))
    assert d == pytest.approx(math.cos(a * x + b) * a)

def test_chain_rule_cos_of_sin():
    # f(x) = x, g(x) = sin(f(x)) = sin(x), h(x) = cos(g(x)) = cos(sin(x))
    f = Differentiable(value=lambda x: x, derivative=lambda _: 1.0)
    g = Sin(f)
    h = Cos(g)
    x = 0.3
    v, d = eval_pair(h, x)
    assert v == pytest.approx(math.cos(math.sin(x)))
    # dérivée = -sin(sin(x)) * cos(x)
    expected_d = -math.sin(math.sin(x)) * math.cos(x)
    assert d == pytest.approx(expected_d)

def test_trig_identity_sin2_plus_cos2_equals_one():
    f = Differentiable(value=lambda x: x, derivative=lambda _: 1.0)
    s = Sin(f)
    c = Cos(f)
    x = 1.2
    lhs = s.value(x)**2 + c.value(x)**2
    # identité trigonométrique exacte
    assert lhs == pytest.approx(1.0, rel=1e-12, abs=1e-12)

def test_derivative_matches_finite_difference():
    f = Differentiable(value=lambda x: 2 * x, derivative=lambda _: 2.0)
    g = Sin(f)
    x = 0.5
    d_analytic = g.derivative(x)

    # approximation par différence finie
    eps = 1e-6
    d_numeric = (g.value(x + eps) - g.value(x - eps)) / (2 * eps)

    assert d_analytic == pytest.approx(d_numeric, rel=1e-6, abs=1e-8)

def test_constant_returns_derivable_instance():
    f = Constant(3.14)
    assert isinstance(f, Differentiable)
    assert callable(f.value) and callable(f.derivative)

@pytest.mark.parametrize("k", [0.0, 1.0, -2.5, 123456.0])
@pytest.mark.parametrize("x", [-10.0, -1.0, 0.0, 1.75, 100.0])
def test_constant_value_equals_k_for_any_x(k, x):
    f = Constant(k)
    v, _ = eval_pair(f, x)
    assert v == pytest.approx(k)

@pytest.mark.parametrize("k", [0.0, 3.0, -7.2])
@pytest.mark.parametrize("x", [-5.0, 0.0, 2.3, 9.9])
def test_constant_derivative_is_zero_for_any_x(k, x):
    f = Constant(k)
    _, d = eval_pair(f, x)
    assert d == pytest.approx(0.0)

def test_constant_is_immutable_wrt_x():
    k = -4.5
    f = Constant(k)
    xs = [-3.0, -1.0, 0.0, 2.0, 5.0]
    for x in xs:
        assert f.value(x) == pytest.approx(k)
        assert f.derivative(x) == pytest.approx(0.0)

def test_constant_derivative_matches_finite_difference():
    """
    Vérifie numériquement que la dérivée est ~ 0 par différences finies.
    """
    k = 7.0
    f = Constant(k)
    x0 = 1.2345
    eps = 1e-6
    d_numeric = (f.value(x0 + eps) - f.value(x0 - eps)) / (2 * eps)
    assert d_numeric == pytest.approx(0.0, abs=1e-10)
    # et la dérivée analytique est 0
    assert f.derivative(x0) == pytest.approx(0.0)

def test_linear_returns_derivable_instance():
    f = Linear(2.0)
    assert isinstance(f, Differentiable)
    assert callable(f.value) and callable(f.derivative)

@pytest.mark.parametrize("k", [0.0, 1.0, -3.5, 10.0])
@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 2.5, 100.0])
def test_linear_value_and_derivative_are_correct(k, x):
    f = Linear(k)
    v, d = eval_pair(f, x)
    assert v == pytest.approx(k * x)
    assert d == pytest.approx(k)

def test_linear_is_consistent_for_multiple_points():
    k = -4.2
    f = Linear(k)
    xs = [-3.0, -1.0, 0.0, 2.0, 5.0]
    for x in xs:
        v, d = eval_pair(f, x)
        assert v == pytest.approx(k * x)
        assert d == pytest.approx(k)

def test_linear_derivative_matches_finite_difference():
    """
    Vérifie numériquement que la dérivée est bien constante = k.
    """
    k = 7.5
    f = Linear(k)
    x0 = 1.2345
    eps = 1e-6
    d_numeric = (f.value(x0 + eps) - f.value(x0 - eps)) / (2 * eps)
    assert d_numeric == pytest.approx(k, rel=1e-6, abs=1e-8)
    # et la dérivée analytique est k
    assert f.derivative(x0) == pytest.approx(k)