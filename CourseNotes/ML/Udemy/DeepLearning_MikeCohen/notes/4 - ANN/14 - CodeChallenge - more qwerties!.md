
> This lecture is a code challenge where you learn to merge code from multiple notebooks into one working file. The goal is to modify your code so that your model generates and classifies three groups of qwerties—blue squares, black circles, and red triangles—instead of two. This challenge is an opportunity to enhance your deep learning, coding, and integration skills.

---

## Table of Contents

1. [Challenge Overview](#challenge-overview)
2. [Problem Description](#problem-description)
3. [Data Generation](#data-generation)
4. [Model Architecture](#model-architecture)
    - [Network Diagram](#network-diagram)
    - [Discussion of Softmax vs. CrossEntropyLoss](#discussion-of-softmax-vs-crossentropyloss)
5. [PyTorch Implementation](#pytorch-implementation)
    - [Data Preparation & Visualization](#data-preparation--visualization)
    - [Defining the Model](#defining-the-model)
    - [Testing Model Output Dimensions](#testing-model-output-dimensions)
6. [Training the Model](#training-the-model)
    - [Training Loop & Metrics](#training-loop--metrics)
    - [Training Dynamics Discussion](#training-dynamics-discussion)
7. [Visualizations](#visualizations)
    - [Loss and Accuracy Plots](#loss-and-accuracy-plots)
    - [Inspecting Softmax Outputs](#inspecting-softmax-outputs)
8. [Additional Explorations](#additional-explorations)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Challenge Overview

- **Objective:**  
  Integrate and modify code from two prior notebooks—one that generates two groups of qwerties and one that implements multioutput classification—to create a model that handles **three groups**.
  
- **Key Skills Developed:**
  - Merging code from multiple sources.
  - Modifying data generation to create three separable clusters.
  - Adjusting network architecture to output three probabilities.
  - Experimenting with explicit softmax inclusion and monitoring training dynamics.

---

## Problem Description

- **Dataset:**  
  300 samples (qwerties) with 2 features (x and y coordinates).  
  - **Clusters:**  
    - **Class 0:** Blue squares  
    - **Class 1:** Black circles  
    - **Class 2:** Red triangles
  
- **Task:**  
  Build and train a neural network that maps 2-dimensional input data to 3-class output, achieving high classification accuracy (target ~95%).

- **Challenge Note:**  
  While the baseline solution uses a 2-layer architecture with a hidden layer of 4 units, feel free to experiment with more hidden layers or different neuron counts.

---

## Data Generation

We generate three clusters by shifting the centers of normally distributed data. Ensure that the clusters are sufficiently separated to minimize overlap.

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 100  # samples per cluster

# Define cluster centers (ensure they are sufficiently separated)
center_A = [2, 2]
center_B = [7, 7]
center_C = [2, 7]

# Generate clusters using Gaussian distributions
cluster_A = np.random.randn(n_samples, 2) + center_A
cluster_B = np.random.randn(n_samples, 2) + center_B
cluster_C = np.random.randn(n_samples, 2) + center_C

# Combine the data into one dataset
X_data = np.vstack([cluster_A, cluster_B, cluster_C])
# Create labels: 0 for Class A, 1 for Class B, 2 for Class C
y_data = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.scatter(cluster_A[:, 0], cluster_A[:, 1], color='blue', marker='s', label='Class 0 (Blue Squares)')
plt.scatter(cluster_B[:, 0], cluster_B[:, 1], color='black', marker='o', label='Class 1 (Black Circles)')
plt.scatter(cluster_C[:, 0], cluster_C[:, 1], color='red', marker='^', label='Class 2 (Red Triangles)')
plt.title("Generated Qwerties Data")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.show()
```

---

## Model Architecture

### Overview

- **Inputs:** 2 features (x, y)
- **Hidden Layer:** 4 neurons (this number is a design choice; experiment if desired)
- **Output Layer:** 3 neurons (one for each class)

A ReLU activation follows the hidden layer. In our implementation, we include an explicit Softmax layer in the model (even though `CrossEntropyLoss` automatically applies a LogSoftmax) to explore its effect.

### Network Diagram

```mermaid
flowchart TD
    A[Input: 2 features]
    B[Hidden Layer: 4 neurons]
    C[ReLU Activation]
    D[Output Layer: 3 neurons]
    E[Softmax (explicit)]
    
    A --> B
    B --> C
    C --> D
    D --> E
```

### Discussion of Softmax vs. CrossEntropyLoss

- **Explicit Softmax:**  
  Applies a probability normalization so that each output row sums to 1.
  
- **CrossEntropyLoss:**  
  Combines LogSoftmax and Negative Log Likelihood Loss, so an explicit Softmax is redundant. Including it here is for exploration; later, you may try removing it to see if performance changes.

---

## PyTorch Implementation

### Data Preparation & Visualization

Convert the numpy arrays into PyTorch tensors.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.long)

# Confirm tensor shapes
print("X_tensor shape:", X_tensor.shape)  # Expected: (300, 2)
print("y_tensor shape:", y_tensor.shape)  # Expected: (300,)
```

### Defining the Model

Here we define a simple feedforward network that includes an explicit Softmax layer.

```python
class QwertiesNet(nn.Module):
    def __init__(self):
        super(QwertiesNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),       # Input layer: 2 -> 4
            nn.ReLU(),             # Non-linearity
            nn.Linear(4, 3),       # Output layer: 4 -> 3
            nn.Softmax(dim=1)      # Explicit Softmax (for exploration)
        )
    
    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = QwertiesNet()
print(model)
```

### Testing Model Output Dimensions

Perform a test forward pass to verify output dimensions and that each sample’s probabilities sum to 1.

```python
model.eval()  # Set to evaluation mode
with torch.no_grad():
    test_output = model(X_tensor)
    print("Output shape:", test_output.shape)  # Expected: (300, 3)
    # Check that probabilities sum to 1 for each sample
    print("Sum of probabilities (first 5 samples):", test_output.sum(dim=1)[:5])
```

---

## Training the Model

We train for 10,000 epochs and compute loss and accuracy at each epoch.

### Training Loop & Metrics

```python
num_epochs = 10000
loss_history = []
accuracy_history = []

# Use CrossEntropyLoss (note: this expects raw logits normally, but we are using explicit Softmax)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss_history.append(loss.item())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy (using argmax to get predicted classes)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_tensor).sum().item()
    accuracy = correct / y_tensor.size(0)
    accuracy_history.append(accuracy)
    
    # Log every 1000 epochs
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")

final_accuracy = accuracy_history[-1] * 100
print(f"\nFinal Accuracy after {num_epochs} epochs: {final_accuracy:.2f}%")
```

### Training Dynamics Discussion

- **Early Epochs:**  
  Expect rapid improvements as the network starts learning the basic structure of the data.
  
- **Mid-to-Late Training:**  
  The loss decreases smoothly, yet the accuracy may only show sudden improvements—indicating that the model might be reorganizing internal weights without immediate gains in accuracy.
  
- **Final Results:**  
  In the provided example, the final accuracy converges around 93–95% after 10,000 epochs.

---

## Visualizations

### Loss and Accuracy Plots

Plot the training loss and accuracy to inspect model convergence.

```python
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, color='green')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
```

### Inspecting Softmax Outputs

Visualize the softmax probabilities for each sample (the output of the final layer).

```python
model.eval()
with torch.no_grad():
    final_outputs = model(X_tensor)
    final_probs = final_outputs.numpy()  # Convert tensor to NumPy array

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(final_probs[:, i], 'o', label=f'Class {i}')
plt.title("Softmax Probabilities for Each Sample")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

> **Note:**  
> Each row in the output should sum to 1, which you can verify with:
> ```python
> print("Row sums (first 5 samples):", final_outputs.sum(dim=1)[:5])
> ```

---

## Additional Explorations

1. **Explicit Softmax vs. Implicit Application:**  
   - Remove the `nn.Softmax(dim=1)` layer from the model definition and retrain. Compare performance when relying on `CrossEntropyLoss` to perform LogSoftmax internally.
   
2. **Architecture Variations:**  
   - Experiment with different hidden layer sizes or additional hidden layers. How does the capacity affect convergence and accuracy?
   
3. **Weight Dynamics Analysis:**  
   - Track and plot the norm or distribution of weights over training epochs to gain insight into internal learning dynamics.
   
4. **Data Randomization:**  
   - Shuffle the dataset before each epoch to study the impact on training performance.

---

## Conclusion

In this code challenge:
- You merged code from previous notebooks to generate three groups of qwerties.
- You built a neural network in PyTorch with two inputs, one hidden layer (4 neurons), and an output layer (3 neurons).
- You trained the model for 10,000 epochs, observing how the loss decreased and accuracy increased (with final accuracy around 93–95%).
- You explored the role of an explicit Softmax layer and gained insight into training dynamics.

This challenge reinforces your ability to integrate multiple codebases, modify model architectures, and troubleshoot deep learning implementations—essential skills for any deep learning researcher.

---

## References

- **PyTorch Documentation:**  
  [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)  
  [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- **Matplotlib Documentation:**  
  [Matplotlib Pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
- **Deep Learning Texts:**  
  Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press.

---

*End of Note*

Feel free to expand on these notes as you experiment with different architectures and further explore the behavior of deep networks. Happy coding and good luck with your challenge!