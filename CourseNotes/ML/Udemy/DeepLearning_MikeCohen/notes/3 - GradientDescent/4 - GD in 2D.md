## 2. High-Level Overview
1. **Goal**: Understand how gradient descent generalizes from 1D to 2D.  
2. **Key Ideas**:
   - Gradient = vector of partial derivatives \(\nabla f\).
   - Update rule in 2D is conceptually identical to 1D; just extended to multiple coordinates.
   - Gradient descent can still get stuck in local minima or find solutions far from the global minimum depending on the initial starting point.
3. **Why This Matters**: Modern machine learning involves very high-dimensional parameter spaces (often millions of dimensions). Understanding 2D generalizations is the next logical step before jumping to high-dimensional gradient-based optimization.

---
## 3. Definitions and Core Concepts
### 3.1 Derivatives vs. Partial Derivatives
- **Derivative (1D)**: For a function \( f(x) \), its derivative \( f'(x) \) gives the instantaneous rate of change at \( x \).  
- **Partial Derivative (ND)**: For a multivariate function 
    $f(x, y, \ldots) : \mathbb{R}^n \to \mathbb{R},$
  the partial derivative w.r.t. \( x \) is 
    $\frac{\partial f}{\partial x}(x, y) = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}.$
  Similarly for $\frac{\partial f}{\partial y}.$
### 3.2 Gradient
- **Gradient**: The gradient of  $f$  in $\mathbb{R}^2$ is the vector of its partial derivatives:
- ![[Screenshot 2024-12-27 at 10.38.07 PM.png]]
- **Notation**:
  - $\nabla$ f  is often denoted by an upside-down triangle, called **nabla**.
  - Partial derivatives $\frac{\partial}{\partial x}$ are denoted using “del” or curvy-d symbols.
### 3.3 Local vs. Global Minima
- **Global Minimum**: The absolute lowest point of \( f \) (in 2D or ND) across the entire domain.
- **Local Minimum**: A point where f is locally the smallest compared to nearby points, but not necessarily the smallest in the entire domain.
- **Multiple Local Minima**: In 2D or higher dimensions, complicated surfaces often have multiple local minima. Gradient descent might find different ones depending on the initialization.
### 3.4 Gradient Descent Algorithm

- **1D**: 
    $x_{\text{new}} \;=\; x_{\text{old}} \;-\; \eta \cdot \frac{d}{dx}f(x_{\text{old}}),$
  where $eta$ is the learning rate.

- **2D**:![[Screenshot 2024-12-27 at 10.39.41 PM.png]]
- **Learning Rate (\(\eta\))**: Controls how big each gradient step is.  
- **Number of Training Epochs / Iterations**: Determines how many times gradient descent updates the parameters.

---
## 4. Mathematical Function Example: “Peaks” Function
We have a function \( f(x, y) \) often called **Peaks**, famously used in MATLAB. It has multiple local minima and maxima, making it a good test function:

$\text{Peaks}(x, y) = 3(1 - x)^2 e^{-x^2 - (y + 1)^2}$
  - - $10\left(\frac{x}{5} - x^3 - y^5\right) e^{-x^2 - y^2}$
  -  - $\frac{1}{3} e^{-(x + 1)^2 - y^2}.$

(Exact forms vary; above is a representative shape.)
- **Global Minimum**: Located at a certain \((x^*, y^*)\).
- **Local Minima**: One or more additional “valleys” that are not as low as the global minimum.
- **Peaks**: Multiple local maxima.

Below is an **ASCII** representation (very rough) of a possible 2D surface shape. The “peaks” are ‘^’, valleys are ‘v’, and the global minimum is the lowest point.

```
               ^ Peak
             ^^^ ^^^
            ^       ^
           /         \
   Plane ->           ^ 
   0 ------- v  (global min)
           \         /
            ^       ^
             vvvvvvv
               v
```

---
## 5. Sympy / Python Workflow
### 5.1 Symbolic vs. Numeric Computation
- **Sympy**: Python library for symbolic math, allowing:
  1. Declaration of symbolic variables: `x = sym.Symbol('x')`
  2. Definition of symbolic expressions: `Z = define your function with x,y`
  3. Taking symbolic derivatives: `sym.diff(Z, x)` or `sym.diff(Z, y)`
  4. Converting symbolic expressions to numeric (NumPy) functions: `sym.lambdify((x, y), dZdx, 'numpy')`
- **Why Use Sympy?** 
  - Quickly get partial derivatives of complicated expressions.
  - Avoid manual differentiation errors.
### 5.2 Implementation Outline
1. **Define the Function**: 
   ```python
   import numpy as np
   def peaks(x, y):
       # This is a commonly used 'peaks' function
       return 3 * (1 - x)**2 * np.exp(-(x**2) - (y+1)**2) \
            - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
            - 1/3 * np.exp(- (x+1)**2 - y**2)
   ```
2. **Symbolic Derivatives**:
   ```python
   import sympy as sym
   
   # Symbolic variables
   x_sym = sym.Symbol('x', real=True)
   y_sym = sym.Symbol('y', real=True)
   
   # Symbolic definition of the same function
   Z = 3*(1 - x_sym)**2 * sym.exp(-(x_sym**2) - (y_sym+1)**2) \
       - 10*(x_sym/sym.Integer(5) - x_sym**3 - y_sym**5) * sym.exp(-x_sym**2 - y_sym**2) \
       - sym.Rational(1, 3)*sym.exp(-(x_sym+1)**2 - y_sym**2)
   
   # Partial derivatives
   dZdx_sym = sym.diff(Z, x_sym)
   dZdy_sym = sym.diff(Z, y_sym)
   
   # Convert to callable functions
   dZdx = sym.lambdify((x_sym, y_sym), dZdx_sym, 'numpy')
   dZdy = sym.lambdify((x_sym, y_sym), dZdy_sym, 'numpy')
   ```
3. **Gradient Descent in 2D**:
   ```python
   # Learning parameters
   eta = 0.1          # learning rate
   n_epochs = 100     # number of iterations
   
   # Initialize random starting point in [-2, 2]
   local_min = np.random.uniform(-2, 2, size=2)  # array([x0, y0])
   
   # To store trajectory
   trajectory = np.zeros((n_epochs+1, 2))
   trajectory[0] = local_min
   
   for i in range(n_epochs):
       # Compute gradient
       grad_x = dZdx(local_min[0], local_min[1])
       grad_y = dZdy(local_min[0], local_min[1])
       
       gradient = np.array([grad_x, grad_y])
       
       # Gradient descent update
       local_min = local_min - eta * gradient
       trajectory[i+1] = local_min
   
   print("Final local min estimate: ", local_min)
   ```

### 5.3 Visualizing the Result

```python
import matplotlib.pyplot as plt

# Create a grid for visualization
grid_x = np.linspace(-3, 3, 200)
grid_y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(grid_x, grid_y)
Z_vals = peaks(X, Y)

plt.figure(figsize=(8,6))
plt.imshow(Z_vals, extent=[-3, 3, -3, 3], origin='lower', cmap='jet')
plt.colorbar(label='Peaks function value')
plt.contour(X, Y, Z_vals, 15, colors='black', alpha=0.5)

# Plot the trajectory
plt.plot(trajectory[:,0], trajectory[:,1], 'k.-')  # black line with dots
plt.scatter([trajectory[0,0]], [trajectory[0,1]], color='green', label='Start')
plt.scatter([trajectory[-1,0]], [trajectory[-1,1]], color='red', label='End')
plt.legend()
plt.title("Gradient Descent on Peaks Function")
plt.show()
```

- **Observations**: 
  - If the initial random point is near the global minimum basin, gradient descent converges to the global minimum.  
  - If it starts near a local minimum, gradient descent settles there.

---

## 6. Potential Pitfalls and Discussion

1. **Local Minima and Saddles**:
   - In higher dimensions, there are many saddle points and local minima.  
   - Gradient descent can converge to these suboptimal points.

2. **Choice of Learning Rate (\(\eta\))**:
   - If \(\eta\) is **too large**, updates overshoot minima and can diverge.  
   - If \(\eta\) is **too small**, convergence can be very slow.

3. **Initialization Sensitivity**:
   - Random initialization can lead to different minima.  
   - In real-world machine learning, repeated runs or carefully chosen initializers are common practice.

4. **Extending to Higher Dimensions**:
   - The step from 2D to high dimensions is “just the same.”  
   - In large-scale ML, the cost function might have *millions* of parameters. The same logic still applies, albeit with more advanced optimization strategies (momentum, Adam, RMSProp, etc.).

---

## 7. Additional Theoretical Notes

1. **Gradient as the Direction of Steepest Ascent**  
   $\nabla f(\mathbf{x})$ points in the direction where  $f$  increases the most rapidly. Taking a step in the opposite direction $-\nabla f(\mathbf{x})$ leads to steepest descent locally.
2. **Convergence Criterion**  
   - Often, we iterate gradient descent until the gradient norm \(\|\nabla f(\mathbf{x})\|\) is below a threshold (\(\epsilon\)), or until a maximum number of iterations (epochs) is reached.

3. **Hessian and Second-Order Methods**  
   - For advanced studies: second-order methods (like Newton’s method) require the Hessian (matrix of second partial derivatives). This is typically expensive but can converge faster for well-conditioned problems.

4. **Stochastic vs. Batch Gradient Descent**  
   - In machine learning, we often can’t compute the exact gradient on the entire dataset. We approximate it using subsets (mini-batches). The underlying principles, however, remain identical.

---

## 8. Example Mathematical Detail

Let’s do a mini-derivation of partial derivatives for a simpler function \( f(x, y) = (x^2 + y^2) \). Suppose:
$f(x, y) = x^2 + y^2.$
Then
$\frac{\partial f}{\partial x} = 2x,$ 
$\quad$
$\frac{\partial f}{\partial y} = 2y,$
$\quad$
$\nabla f(x, y) = \begin{bmatrix} 2x \\ 2y \end{bmatrix}.$
Gradient descent steps:
\[
$\begin{bmatrix} x_{\text{new}} \\ y_{\text{new}} \end{bmatrix}$
= 
$\begin{bmatrix} x_{\text{old}} \\ y_{\text{old}} \end{bmatrix}$ - $\eta \begin{bmatrix} 2\,x_{\text{old}} \\ 2\,y_{\text{old}} \end{bmatrix}.$
Hence,
$x_{\text{new}} = x_{\text{old}} - 2\,\eta\, x_{\text{old}}$ ,
$\quad$
$y_{\text{new}} = y_{\text{old}} - 2\,\eta\, y_{\text{old}}.$
---

## 9. ASCII Quick-Reference

```
   2D Gradient Descent:

   Initialize (x, y) randomly
     |
     v
   while not converged:
       grad_x = dF/dx(x, y)
       grad_y = dF/dy(x, y)
       (x, y) = (x, y) - eta * (grad_x, grad_y)
     |
     v
   Output (x, y) as local minimum
```
---
## 10. Summary

- **Gradient Descent from 1D to 2D**: The same concept—just more components in the gradient vector.  
- **Sympy**: A powerful tool to automatically derive partial derivatives for complicated functions.  
- **Visualization**: In 2D, we can physically visualize the function surface, but the same principles generalize to very high dimensions.  
- **Further Study**: Explore advanced optimizers, second-order methods, or specialized GPU-accelerated libraries (like TensorFlow or PyTorch) for large-scale machine learning tasks.
---
## 11. References and Further Reading

- [*Sympy Documentation*](https://docs.sympy.org/latest/index.html)  
- [*Gradient Descent in Machine Learning* – Coursera or Udemy ML courses]  
- [*Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville – Chapter on Optimization]  
- [*Pattern Recognition and Machine Learning* by Christopher Bishop – Section on Gradient-based optimization]