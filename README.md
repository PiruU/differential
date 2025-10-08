# differential

Lightweight Python package for **symbolic and automatic differentiation**.  
It provides differentiable objects and elementary mathematical functions that can be combined to build complex differentiable expressions, compute derivatives, and evaluate results numerically.

---

## Features

- **Symbolic differentiation** of composed mathematical expressions  
- **Elementary differentiable functions** (`Sin`, `Cos`, `Sqrt`, `Linear`, `Constant`)  
- A base class `Differentiable` to define custom differentiable functions  
- Easy composition of operations (`+`, `-`, `*`, `/`, and more)  
- Clean and extensible object-oriented design  

---

## Installation

To install from source:

```bash
pip install .
```

For editable installs during development:

```bash
pip install -e .
```

Requirements: Python ≥ 3.9 and `pytest` for testing.

---

## Quickstart Example

```python
from differential import Differentiable, Sin, Cos, Constant, Linear

# Define a variable and a function
x = Linear(1.0)       # f(x) = x
f = Sin(x) + Constant(2) * Cos(x)

# Evaluate numerically
print("f(0.5) =", f.value(0.5))

# Compute derivative symbolically and evaluate it
print("f'(0.5) =", f.derivative(0.5)(0.5))
```

This example defines the function  
**f(x) = sin(x) + 2 · cos(x)**  
and computes both its value and its derivative at x = 0.5.

---

## API Overview

### 1. Class `Differentiable`

Base class for all differentiable entities.

---

### 2. Elementary Functions

| Function | Description | Example |
|-----------|--------------|----------|
| `Constant(c)` | Constant function f(x) = c | `Constant(5)` |
| `Linear(a=1, b=0)` | Linear function f(x) = a·x + b | `Linear(2, 3)` |
| `Sin(expr)` | Sine of an expression | `Sin(Linear())` |
| `Cos(expr)` | Cosine of an expression | `Cos(Sin(x))` |
| `Sqrt(expr)` | Square root of an expression | `Sqrt(Linear())` |

All functions are `Differentiable` and can be combined arithmetically.

---

### 3. Composition Example

```python
from differential import Linear, Sin, Cos, Sqrt

x = Linear()                 # f(x) = x
expr = Sqrt(Sin(x) + Cos(x)) # f(x) = √(sin(x) + cos(x))

val = expr.value(1.0)
deriv = expr.derivative(1.0)

print("f(1.0) =", val)
print("f'(1.0) =", deriv)
```

---

## Project Structure

```
differential/
├── src/
│   └── differential/
│       ├── __init__.py
│       ├── differentiable.py
│       └── functions.py
└── tests/
    ├── differentiable_test.py
    └── functions_test.py
```

---

## Testing

Run the unit tests using `pytest`:

```bash
pytest -q
```

All test files are located in the `tests/` directory.
