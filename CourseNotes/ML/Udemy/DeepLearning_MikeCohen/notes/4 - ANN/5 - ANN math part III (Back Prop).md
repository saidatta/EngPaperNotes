
> **Overview:**  
> In this lecture, we put together all the pieces—from the perceptron model to a full deep network—and show how gradient descent is extended to handle non-linear activation functions via the chain rule. We examine how each node (unit) computes a weighted linear sum followed by a non-linearity, and then how the error propagates backward through the network. Ultimately, backpropagation is nothing more than gradient descent applied to a high-dimensional, composite function.

---

## Table of Contents

1. [Introduction & Motivation](#introduction--motivation)
2. [From Perceptron to Deep Networks](#from-perceptron-to-deep-networks)
3. [Gradient Descent Recap](#gradient-descent-recap)
4. [The Chain Rule in Backpropagation](#the-chain-rule-in-backpropagation)
5. [Example: Derivative of MSE with Activation](#example-derivative-of-mse-with-activation)
6. [Code Example: Implementing Backpropagation in PyTorch](#code-example-implementing-backpropagation-in-pytorch)
7. [Visualization: Loss Convergence Over Training](#visualization-loss-convergence-over-training)
8. [Discussion & Practical Considerations](#discussion--practical-considerations)
9. [Conclusion](#conclusion)
10. [References & Further Reading](#references--further-reading)

---
## Introduction & Motivation

- **Why Backpropagation?**  
  - Deep networks are built from many simple units (each performing a weighted sum and non-linear activation).  
  - To train these networks, we need to adjust the weights so that the network’s output approximates the target values.
  - **Backpropagation** is the algorithm that computes the gradient of the loss function with respect to every weight in the network, using the chain rule to "propagate" the error backward from the output layer to earlier layers.
  
- **Key Concept:**  
  - Although each unit's computation is simple, the interaction of thousands of units makes the network complex.  
  - Backpropagation is essentially gradient descent extended to a high-dimensional setting with nested functions (the activations).
---
## From Perceptron to Deep Networks
- **Perceptron Model:**  
  Each node computes:  
  $y = \sigma\left(\sum_{i} w_i x_i + b\right)$
  where $\sigma$ is a non-linear activation function (e.g., ReLU, sigmoid).
- **Deep Network:**  
  - Stacks many such nodes in layers.
  - The output of one layer becomes the input of the next.
  - **Key Insight:** Every node acts independently, computing its own weighted sum and activation.

  
- **Simplified Diagram:**  
![[Pasted image 20250131162048.png]]
  Instead of drawing each weight and node, we represent an entire perceptron (node) as a circle:
- In a deep network, these units are arranged in layers, and the error from the final output is propagated backward.

---
## Gradient Descent Recap

- **Gradient Descent Update Rule:**  
  For a single weight \(w\), the update rule is:
  $w \leftarrow w - \eta \frac{\partial L}{\partial w}$
  \]
  where:
  - \(\eta\) is the learning rate.
  - \(\frac{\partial L}{\partial w}\) is the derivative of the loss function \(L\) with respect to \(w\).

- **High-Dimensional Extension:**  
  - In deep networks, weights are vectors/matrices.
  - The update is applied to all weights simultaneously, using their respective partial derivatives.

---

## The Chain Rule in Backpropagation
- **Why the Chain Rule?**  
  - In a deep network, the loss \(L\) depends on weights indirectly via multiple nested functions.
  - For example, if a unit computes $y = \sigma(u)$ and $u = \mathbf{w} \cdot \mathbf{x} + b$, then by the chain rule:
    $\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial \mathbf{w}}$

- **Simplification Using a Substitution:**  
  - Often we let \(U = u\) so that:
    $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial U} \cdot \frac{\partial U}{\partial w}$
  - The derivative of the activation \(\sigma\) (e.g., for sigmoid, $\sigma'(u) = \sigma(u)(1-\sigma(u))\)$) comes into play.

- **Propagation Across Layers:**  
  - The error (gradient) computed at the output layer is propagated backward through each layer using the chain rule.

---
![[Pasted image 20250131162945.png]]
## Example: Derivative of MSE with Activation

- **Mean Squared Error (MSE):**  
  $L = \frac{1}{2}(y - \hat{y})^2$

  where $\hat{y} = \sigma(u)$ and $u = \mathbf{w} \cdot \mathbf{x} + b.$

- **Computing the Gradient:**
  1. **Output Layer:**  
    
     $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$
    
  2. **Activation Derivative:**  
    
     $\frac{\partial \hat{y}}{\partial u} = \sigma'(u)$
    
  3. **Weight Derivative:**  

     $\frac{\partial u}{\partial w_i} = x_i$
    
  4. **Combined via Chain Rule:**
     $\frac{\partial L}{\partial w_i} = (\hat{y} - y) \cdot \sigma'(u) \cdot x_i$

  - **Interpretation:**  
    Each weight is updated proportionally to the product of the error, the sensitivity of the activation, and the input feature.

---

## Code Example: Implementing Backpropagation in PyTorch

Below is a simple PyTorch example that constructs a one-layer perceptron, computes the forward pass, calculates the loss (using MSE for demonstration), and uses autograd to compute gradients (backpropagation).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a simple one-layer perceptron model
class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # computes u = Wx + b
    
    def forward(self, x):
        # Compute weighted sum and then apply a non-linear activation (sigmoid)
        u = self.linear(x)
        y_hat = torch.sigmoid(u)
        return y_hat

# Instantiate model
model = SimplePerceptron(input_dim=3, output_dim=1)

# Create sample data
torch.manual_seed(42)
x = torch.randn(10, 3)       # 10 samples, 3 features
y = torch.randn(10, 1)       # 10 target values

# Forward pass
y_hat = model(x)

# Compute Mean Squared Error (MSE) Loss
criterion = nn.MSELoss()
loss = criterion(y_hat, y)
print(f"Initial Loss: {loss.item():.4f}")

# Perform Backpropagation: compute gradients
loss.backward()

# Print gradients for the weights
print("Gradients for weights:")
print(model.linear.weight.grad)
print("Gradients for bias:")
print(model.linear.bias.grad)

# Example of a weight update (Gradient Descent Step)
learning_rate = 0.01
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad

# Forward pass after weight update to see new loss
y_hat_new = model(x)
loss_new = criterion(y_hat_new, y)
print(f"New Loss after one update: {loss_new.item():.4f}")
```

**Explanation:**

- We define a simple perceptron with one linear layer followed by a sigmoid activation.
- We compute the forward pass to get the predicted output \( \hat{y} \).
- The MSE loss is computed, and `loss.backward()` automatically computes the gradients via backpropagation.
- We then update the weights using a simple gradient descent step.
- Printing the loss before and after the update shows the effect of backpropagation.

---
## Visualization: Loss Convergence Over Training

Below is a sample training loop that tracks the loss over epochs and plots it, illustrating how gradient descent reduces the loss over time.

```python
# Reset model and optimizer for training demonstration
model = SimplePerceptron(input_dim=3, output_dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

loss_history = []
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Convergence Over Training")
plt.grid(True)
plt.show()
```

**Observation:**  
The plot should show a decrease in loss over epochs as gradient descent minimizes the cost function.

---

## Discussion & Practical Considerations

- **Backpropagation is Gradient Descent:**  
  - The core update rule is:  
    \[
    w \leftarrow w - \eta \frac{\partial L}{\partial w}
    \]
  - With deep networks, the chain rule is applied iteratively from the output layer back to the input layer.

- **Handling Non-linearities:**  
  - Each activation function (e.g., sigmoid, ReLU) has a corresponding derivative that is used in the chain rule.
  - Numerical stability is crucial—modern libraries implement optimizations to ensure stability and efficiency.
- **Complexity vs. Simplicity:**  
  - While the math can become “hairy” with many nested functions, the underlying principle remains simple.
  - Automatic differentiation (autograd) abstracts much of this complexity away in frameworks like PyTorch.
- **Interpretability:**  
  - Understanding backpropagation helps demystify how deep learning models learn.
  - Even if we can compute gradients by hand, large-scale networks rely on efficient, numerically stable implementations.
---
## Conclusion
- **Key Takeaways:**
  - **Backpropagation** is the extension of gradient descent using the chain rule to update weights in a deep network.
  - Each node computes a simple operation, but backpropagation aggregates these computations to adjust thousands of parameters.
  - Practical implementation using libraries like PyTorch hides much of the mathematical complexity via automatic differentiation.
  
- **Final Thought:**  
  While the detailed calculus can be challenging, a solid conceptual grasp of backpropagation is essential for understanding how deep learning models learn and adapt.

---

## References & Further Reading

- **Deep Learning Book:**  
  *Deep Learning* by Goodfellow, Bengio, and Courville – Chapters on Backpropagation and Optimization.
- **PyTorch Documentation:**  
  - [torch.autograd](https://pytorch.org/docs/stable/autograd.html)
  - [Neural Network Tutorials](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- **Online Resources:**  
  - Articles on the chain rule in backpropagation.
  - Videos and lectures on gradient descent and optimization in deep learning.

---

*End of Note*