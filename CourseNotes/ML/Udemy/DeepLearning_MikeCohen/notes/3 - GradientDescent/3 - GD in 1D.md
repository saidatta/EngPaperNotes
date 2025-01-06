## Overview
In this lecture, we learn how to implement gradient descent in Python, focusing on one-dimensional (1D) problems for simplicity. We explore the importance of two key parameters in deep learning: learning rate and the number of training epochs. By adjusting these parameters, we understand why gradient descent isn't guaranteed to provide correct or optimal answers.
## Topics Covered
1.  Implementing gradient descent in Python
2.  Learning rate and training epochs
3.  Why gradient descent may not guarantee the correct answer
## Implementation
We use NumPy and Matplotlib to implement gradient descent, focusing on a simple function (f(x)) and its derivative (f'(x)). By doing so, we can visualize the results and understand the impact of learning rate and training epochs on the algorithm's performance.
## Gradient Descent Algorithm
1.  Compute the gradient (derivative) at the current estimate of the local minimum.
2.  Update the estimate of the local minimum by subtracting the learning rate multiplied by the computed gradient.
## Key Parameters
1.  **Learning rate**: A small value (e.g., 0.01) that scales the gradient as we update the local minimum estimate. A smaller learning rate requires more iterations to reach the minimum, while a larger learning rate can overshoot the minimum.
2.  **Number of training epochs**: The number of iterations for the gradient descent algorithm. A larger number of epochs allows the algorithm to explore the function more thoroughly, while a smaller number can lead to an inaccurate estimate of the minimum.
## Observations
-   With a very small learning rate, the algorithm takes tiny steps and may require more iterations (epochs) to reach the minimum.
-   A larger learning rate can lead to overshooting the minimum, resulting in an inaccurate estimate.
-   The optimal learning rate and number of training epochs depend on the problem, the dataset, and the model's architecture.
## Additional Explorations
1.  Experiment with different learning rates and numbers of training epochs to understand their impact on the performance of the gradient descent algorithm.
2.  Explore how the choice of the initial estimate for the local minimum affects the algorithm's convergence.
3.  Investigate the behavior of the gradient descent algorithm with different functions and their derivatives.
4.  Compare the results of the gradient descent algorithm with those obtained using analytical methods for finding the minimum of a function.

```python 
import numpy as np import matplotlib.pyplot as plt 

# Define the function and its derivative 
def f(x): return (x - 0.5) ** 2 def df(x): return 2 * (x - 0.5) 

# Gradient Descent Function 
def gradient_descent(lr, epochs): 
	x = np.random.choice(np.linspace(-2, 2, 2001)) 
	history = [] for _ in range(epochs): gradient = df(x) x = x - lr * gradient history.append(x) return x, history 
	
# Compare different learning rates and epochs learning_rates = [0.1, 0.01, 0.001] epochs_list = [100, 500, 1000] for lr in learning_rates: for epochs in epochs_list: x_min, history = gradient_descent(lr, epochs) plt.plot(history, label=f"LR: {lr}, Epochs: {epochs}")

plt.xlabel("Epochs") plt.ylabel("Estimate of Local Minimum") plt.legend() plt.show() ``` </pre>