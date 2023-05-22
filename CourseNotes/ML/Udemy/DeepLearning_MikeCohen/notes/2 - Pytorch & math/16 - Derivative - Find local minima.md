### Main Purpose of Derivatives in Gradient Descent
- Identify local minima for optimization
- Used in high-dimensional functions where visualization isn't possible
- Compute derivative, set it to zero, and solve for x to identify critical points (minima and maxima)

### Minima and Maxima
- Local minimum: point on a graph where the function is lower than its surroundings
- Local maximum: point on a graph where the function is higher than its surroundings
- Critical points: points where the derivative is zero
- To distinguish between minima and maxima, observe the sign of the derivative before and after the critical point

### How to Distinguish Local Minima from Local Máxima
- Local minimum: derivative is negative to the left and positive to the right
- Local maximum: derivative is positive to the left and negative to the right
- In gradient descent, the algorithm takes advantage of this distinction to identify minima and not máxima

### Vanishing Gradient Problem
- Occurs when the derivative is zero, but the function is neither at a minimum nor a maximum
- Can be problematic in deep learning
- Solution: Use specific techniques in deep learning to address the vanishing gradient problem

Equation:
- df/dx = 0 (derivative of the function equals zero at critical points)

Example:
1. Function: Third-order polynomial with three solutions for critical points
2. Compute derivative and set to zero
3. Solve for x to identify minima and maxima
4. Observe the sign of the derivative before and after the critical point to distinguish between minima and maxima

In this chapter, you learned the main purpose of using derivatives in gradient descent, how to identify minima and maxima points, and how to distinguish a minimum from a maximum. These concepts are essential prerequisites for understanding how gradient descent works.