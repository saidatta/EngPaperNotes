## Intuition of derivatives
- Derivatives describe how functions change
- They tell us the slope of the function at each point
- Used in deep learning for optimization (minimizing error functions)
	
## Computing the derivative of a polynomial
- A polynomial function has a variable, a coefficient, and some power
- Example: x^2 (variable: x, coefficient: 1, power: 2)
- To compute the derivative of a polynomial:
  1. Multiply the coefficient by the power
  2. Subtract 1 from the power
- Formula: d(ax^n)/dx = n(ax^(n-1))
- Examples:
  - d(x^2)/dx = 2x
  - d(x^3)/dx = 3x^2
  - d(3x^3)/dx = 9x^2

## Visualizing derivatives with functions
- Derivative of x^2 is 2x
- Derivative of RELU and sigmoid functions can also be computed and visualized
- RELU function: max(0, x)
- Sigmoid function: 1 / (1 + e^(-x))

## Python code using SymPy library
- Import SymPy and SymPy plotting module
- Create symbolic variable x
- Compute derivative using SymPy's diff function
- Plot function and its derivative using SymPy

```python
import sympy as sp
from sympy.plotting import plot

x = sp.Symbol('x')

# Example: Function 2x^2
f = 2 * x**2
f_derivative = sp.diff(f, x)

# Plot function and its derivative
p1 = plot(f, show=False)
p2 = plot(f_derivative, show=False)
p1.extend(p2)
p1.show()

# RELU and Sigmoid functions
relu = sp.Max(0, x)
sigmoid = 1 / (1 + sp.exp(-x))

# Compute derivatives
relu_derivative = sp.diff(relu, x)
sigmoid_derivative = sp.diff(sigmoid, x)

# Plot functions and their derivatives
plot(relu, sigmoid, show=True)
plot(relu_derivative, sigmoid_derivative, show=True)
```

## Summary
- Derivatives help understand how functions change and are used in deep learning optimization
- Computing the derivative of a polynomial involves multiplying the coefficient by the power and subtracting 1 from the power
- SymPy library can be used to compute and visualize derivatives in Python