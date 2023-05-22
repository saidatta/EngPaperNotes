## Summary

In this chapter, Mike Cohen discusses various ways of changing the learning rate during gradient descent to make it dynamic. He then provides a code challenge where you can implement one or more of these methods in Python. The goal is to explore the advantages and disadvantages of each method and understand how they might fail.

## Fixed Learning Rate vs. Dynamic Learning Rate

1. Fixed Learning Rate: The learning rate remains constant throughout the gradient descent process.
2. Dynamic Learning Rate: The learning rate changes at each iteration through the gradient descent process.

## Four Possible Ways to Change the Learning Rate

1. Adjusting the learning rate based on the training epoch.
2. Modulating the learning rate according to the derivative or gradient.
3. Modifying the learning rate based on the loss function.
4. Changing the learning rate according to the current minimum value.

### Method 1: Adjusting the Learning Rate Based on the Training Epoch

- As the learning progresses, the learning rate decreases.
- Advantage: It's a simple method and often used in practice.
- Disadvantage: The method is not related to model performance.

### Method 2: Modulating the Learning Rate According to the Derivative or Gradient

- The learning rate is scaled by the derivative or gradient.
- Advantage: The method is adaptive to the problem.
- Disadvantage: Requires additional parameters and appropriate scaling.

### Method 3: Modifying the Learning Rate Based on the Loss Function

- The learning rate is adjusted based on the difference between the correct answer and the model's prediction.
- Advantage: Adaptive to the problem.
- Disadvantage: Requires appropriate scaling.

### Method 4: Changing the Learning Rate According to the Current Minimum Value

- Not a viable solution as it makes too many assumptions.

## Industry Standard Methods

1. Learning Rate Decay: Adjusting the learning rate based on the training epoch. It is discussed further in the Meta Parameters section.
2. RMS Prop and Adam Optimizers: Modulating the learning rate according to the gradient. These optimization algorithms are also discussed in the Meta Parameters section.

## Code Challenge

The code challenge involves implementing one or more of the methods described above in Python. After implementing these methods, compare them against a fixed learning rate, and critically analyze the advantages and disadvantages of each method.


	### Example of those 4 methods disccusued
Here's an example in Python, implementing the four methods of modifying the learning rate mentioned earlier: constant learning rate, time-based decay, step decay, and exponential decay. We'll use a simple example of finding the minimum value of a quadratic function: `f(x) = (x - 2)^2 + 3`.

```python
import numpy as np
import matplotlib.pyplot as plt

# Objective function: f(x) = (x - 2)^2 + 3
def objective_function(x):
    return (x - 2)**2 + 3

# Gradient of the objective function: f'(x) = 2(x - 2)
def gradient_function(x):
    return 2 * (x - 2)

# Gradient descent algorithm
def gradient_descent(initial_x, epochs, lr_schedule, **lr_params):
    x = initial_x
    x_history = [x]

    for epoch in range(epochs):
        lr = lr_schedule(epoch, **lr_params)
        x = x - lr * gradient_function(x)
        x_history.append(x)

    return x_history

# Learning rate schedules
def constant_lr(epoch, lr):
    return lr

def time_based_decay_lr(epoch, lr, decay):
    return lr / (1 + decay * epoch)

def step_decay_lr(epoch, lr, decay, step_size):
    return lr * (decay ** np.floor(epoch / step_size))

def exponential_decay_lr(epoch, lr, decay):
    return lr * np.exp(-decay * epoch)

# Parameters
initial_x = 6
epochs = 50

# Learning rate schedules parameters
constant_params = {'lr': 0.1}
time_based_decay_params = {'lr': 0.1, 'decay': 0.01}
step_decay_params = {'lr': 0.1, 'decay': 0.5, 'step_size': 10}
exponential_decay_params = {'lr': 0.1, 'decay': 0.1}

# Perform gradient descent using different learning rate schedules
constant_history = gradient_descent(initial_x, epochs, constant_lr, **constant_params)
time_based_decay_history = gradient_descent(initial_x, epochs, time_based_decay_lr, **time_based_decay_params)
step_decay_history = gradient_descent(initial_x, epochs, step_decay_lr, **step_decay_params)
exponential_decay_history = gradient_descent(initial_x, epochs, exponential_decay_lr, **exponential_decay_params)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(constant_history, label='Constant LR', linestyle='--')
plt.plot(time_based_decay_history, label='Time-based Decay', linestyle='-.')
plt.plot(step_decay_history, label='Step Decay', linestyle=':')
plt.plot(exponential_decay_history, label='Exponential Decay', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('x')
plt.legend()
plt.show()
```

This code defines the objective function, its gradient, and the four learning rate schedules. It then performs gradient descent using each schedule and plots the results. The x-axis represents the number of epochs, and the y-axis represents the position (x value) in each epoch. The plot shows how each learning rate schedule converges to the minimum value of the objective function.

