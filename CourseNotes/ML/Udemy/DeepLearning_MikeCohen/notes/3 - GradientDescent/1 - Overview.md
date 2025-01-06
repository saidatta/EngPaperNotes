-   Gradient Descent is the most important algorithm in deep learning.
-   Without gradient descent, deep learning wouldn't exist.
-   It's used for optimization in other areas of machine learning and AI.
-   It involves using derivatives and some algebra.
## Motivation
-   Deep learning models learn in three steps:
    1.  Guess a solution randomly.
    2.  Compute the error (mistakes or whether the guess was correct or incorrect).
    3.  Learn from mistakes by modifying the parameters to give a better guess next time.
## Mathematical Framework
-   The goal is to convert the statement in step 3 into a mathematical framework that the model can implement and compute.
-   We need a mathematical description of the error landscape.
-   We need to discover a way to find the minimum of that error landscape.
-   Calculus, specifically derivatives, come to the rescue.
-   Gradient Descent finds the minimum of the error function.
## Gradient vs Derivative
-   Gradient is the same as a derivative.
-   The term "derivative" is used for one-dimensional functions.
-   The term "gradient" is used for multidimensional functions.
## Gradient Descent Algorithm
1.  Initialize a random guess of the minimum.
2.  Loop over some number of iterations.
3.  Compute the derivative at the initial guess of the function minimum.
4.  Update the guess by subtracting the derivative scaled by the learning rate.
### Example of Gradient Descent
-   Function: f(x) = 3x^2 - 6x + 4
-   Derivative: f'(x) = 6x - 6
-   Learning rate: 0.1

1.  Initialize random guess: x = 1.5
2.  Loop for 10 iterations:
    -   Compute the derivative: f'(1.5) = 3
    -   Update the guess: x = x - learning_rate * f'(x) = 1.5 - 0.1 * 3 = 1.2
    -   Repeat for 9 more iterations

After 10 iterations, the value of x converges close to the true minimum at x = 1.
## Issues with Gradient Descent
-   Gradient Descent is not guaranteed to give the correct solution.
-   Possible issues include:
    1.  Poor choice of model parameters.
    2.  Getting stuck in local minima.
    3.  Vanishing or exploding gradients.
-   Many problems have been overcome through research and development in deep learning.
-   There are tricks and techniques to get around some of the problems of gradient descent.
### GD: Local minima stuck issue 
-   Function: f(x) = x^4 - 4x^2 + 2x
-   Derivative: f'(x) = 4x^3 - 8x + 2
-   Learning rate: 0.01
1.  Initialize random guess: x = -0.5
2.  Loop for 10 iterations:
    -   Compute the derivative: f'(-0.5) = 3.5
    -   Update the guess: x = x - learning_rate * f'(x) = -0.5 - 0.01 * 3.5 = -0.535
    -   Repeat for 9 more iterations
After 10 iterations, x converges close to the local minimum at x = -0.577, but the global minimum is actually at x = 0.577. In this case, Gradient Descent got stuck in a local minimum instead of finding the global minimum.
### GD: Vanishing Gradients issue
-   A common issue in deep neural networks, especially when using activation functions like sigmoid and hyperbolic tangent (tanh).
-   Occurs when the gradients become very small during backpropagation, causing the weights to update very slowly or not at all.
-   This leads to slow convergence or a model that does not learn from the training data.
-   The problem is exacerbated in deeper networks as the gradients are multiplied through many layers.
#### Example of Vanishing Gradients
-   Consider a simple feedforward neural network with 4 layers and sigmoid activation functions.
-   Sigmoid function: σ(x) = 1 / (1 + exp(-x))
-   Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
1.  Assume that the weights and biases are randomly initialized.
2.  Perform a forward pass through the network and compute the error.
3.  Begin backpropagation to update the weights.

In the backpropagation process, we need to calculate the gradients of the error with respect to the weights. For this, we use the chain rule and compute the derivative of the error with respect to the output, times the derivative of the output with respect to the input (which is the derivative of the sigmoid function).

Since the derivative of the sigmoid function (σ'(x)) has a maximum value of 0.25, the gradients can become very small when multiplied through multiple layers. This results in vanishing gradients, and the weights in the earlier layers update very slowly, if at all.

Suppose we have the following values for the output of the first layer (before activation): x1 = 7, x2 = -1, x3 = 2

The corresponding sigmoid activation values are: σ(x1) = 0.999, σ(x2) = 0.269, σ(x3) = 0.881

The derivatives of the sigmoid function at these points are: σ'(x1) = 0.0009, σ'(x2) = 0.197, σ'(x3) = 0.105

When the backpropagation process multiplies these small gradient values across multiple layers, the resulting gradients get smaller and smaller, leading to the vanishing gradients issue. This can slow down or stall the learning process, making it difficult for the model to converge to an optimal solution.
# Key Takeaways
-   Gradient Descent is a simple and powerful algorithm.
-   It always works well but is not guaranteed to work perfectly.
-   Understanding the gradient descent algorithm is crucial for deep learning.