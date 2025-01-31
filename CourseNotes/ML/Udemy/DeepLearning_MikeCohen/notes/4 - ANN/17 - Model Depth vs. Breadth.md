Below is an extensive set of Obsidian notes in Markdown format covering the lecture “ANN: Model depth vs. breadth.” These notes explain the key concepts of depth versus breadth (width), detail the design and implementation of a parametric experiment in PyTorch (using the Iris dataset as an example), include code examples for defining flexible model classes, show how to train and evaluate models with varying depth and breadth, and provide visualizations and discussions of the results.

> **"A model’s performance is not simply a function of the number of parameters it has; the architecture is also important."**  
> — *Mike*

This lecture explores two crucial dimensions of deep learning model complexity:
- **Depth:** The number of hidden layers (i.e., the levels of abstraction).
- **Breadth (or Width):** The number of units (neurons) per hidden layer.

We will see that simply making a model deeper is not automatically beneficial. Furthermore, even if one model has more trainable parameters than another, that does not guarantee better performance. In today’s experiment, we manipulate both the number of hidden layers and the number of hidden units per layer.

---

## Table of Contents

1. [Key Concepts & Terminology](#key-concepts--terminology)
2. [Experiment Design Overview](#experiment-design-overview)
3. [PyTorch Implementation](#pytorch-implementation)
   - [Data Preparation](#data-preparation)
   - [Flexible Model Definition (Custom Class)](#flexible-model-definition-custom-class)
   - [Training Function](#training-function)
   - [Parametric Experiment Loop](#parametric-experiment-loop)
4. [Visualizations & Results](#visualizations--results)
5. [Discussion & Insights](#discussion--insights)
6. [Additional Explorations](#additional-explorations)
7. [References & Further Reading](#references--further-reading)

---

## Key Concepts & Terminology

- **Depth:**  
  - Defined as the number of hidden layers in a network (i.e., the layers between the input and the output).
  - More layers can allow a network to learn more abstract representations.

- **Breadth (or Width):**  
  - The number of units (neurons) in each hidden layer.
  - A layer with many neurons is considered “wide” and can increase the total number of parameters substantially.

- **Trade-offs:**  
  - **Shallow (low depth) but wide models:**  
    - Often learn quickly due to fewer layers.
    - May have a large number of parameters if each layer is very wide.
  - **Deep (high depth) but narrow models:**  
    - Can capture more complex and abstract representations.
    - May have fewer total parameters if each layer has fewer neurons.
  - **Key Observation:**  
    - Model performance is not solely determined by the total number of parameters.  
    - Other factors (learning rate, training epochs, optimizer, etc.) and architectural details play a critical role.

---

## Experiment Design Overview

We will conduct a parametric experiment using the Iris dataset. The two primary variables are:

1. **Number of hidden layers (depth):**  
   - We vary from 1 to 5 hidden layers.

2. **Number of units per hidden layer (breadth/width):**  
   - We vary the number of units from a lower bound (e.g., 4) to an upper bound (e.g., 100) in steps (e.g., every 3 units).

**Goal:**  
For each combination, train the model for a fixed number of epochs, record the final classification accuracy, and store the total number of trainable parameters. Later, plot the accuracy as a function of the number of hidden units for different depths.

---

## PyTorch Implementation

### Data Preparation

We use the Iris dataset (with 4 features and 3 classes). The following code shows how to load and convert the data into PyTorch tensors.

```python
import pandas as pd
import numpy as np
import torch

# Load the Iris dataset (using an online CSV for convenience)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(url)

# Extract features and labels
X_np = iris_df.iloc[:, 0:4].values    # shape: (150, 4)
# Map species to integers: setosa->0, versicolor->1, virginica->2
y_np = iris_df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.long)

print("X_tensor shape:", X_tensor.shape)  # Expected: (150, 4)
print("y_tensor shape:", y_tensor.shape)    # Expected: (150,)
```

---

### Flexible Model Definition (Custom Class)

Here we define a flexible model class that accepts two parameters: `n_units` (number of units per hidden layer) and `n_layers` (number of hidden layers). This approach gives us the flexibility to vary both depth and breadth.

```python
import torch.nn as nn
import torch.nn.functional as F

class ANNIris(nn.Module):
    def __init__(self, n_units, n_layers):
        """
        Initializes the ANNIris model.
        - n_units: Number of units in each hidden layer.
        - n_layers: Number of hidden layers.
        The input layer has 4 features (Iris dataset) and the output layer has 3 units (3 classes).
        """
        super(ANNIris, self).__init__()
        self.n_layers = n_layers
        
        # Dictionary (ModuleDict) to hold layers
        self.layers = nn.ModuleDict()
        
        # Input layer: from 4 features to n_units
        self.layers['input'] = nn.Linear(4, n_units)
        
        # Create hidden layers: Each hidden layer maps from n_units to n_units.
        for i in range(n_layers):
            self.layers[f'hidden{i}'] = nn.Linear(n_units, n_units)
        
        # Output layer: from n_units to 3 outputs
        self.layers['output'] = nn.Linear(n_units, 3)
    
    def forward(self, x):
        # Forward pass: Input layer then each hidden layer with ReLU activation, then output.
        x = self.layers['input'](x)
        x = F.relu(x)
        
        # Loop over hidden layers
        for i in range(self.n_layers):
            x = self.layers[f'hidden{i}'](x)
            x = F.relu(x)
        
        # Output layer (typically, CrossEntropyLoss expects raw logits)
        x = self.layers['output'](x)
        return x

# Example: Create a model with 40 units per hidden layer and 3 hidden layers
model_example = ANNIris(n_units=40, n_layers=3)
print("Example Model:")
print(model_example)
```

---

### Training Function

Below is a simple training function that trains a given model on the Iris dataset for a fixed number of epochs and returns the final accuracy and total trainable parameters.

```python
import torch.optim as optim

def train_model(model, X, y, num_epochs=2500, lr=0.01):
    """
    Trains the given model on (X, y) for num_epochs using SGD.
    Returns the final accuracy and the total number of trainable parameters.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, dim=1)
        correct = (preds == y).sum().item()
        accuracy = correct / y.size(0)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return accuracy, total_params

# Test the training function with our example model
accuracy_example, params_example = train_model(model_example, X_tensor, y_tensor, num_epochs=2500, lr=0.01)
print(f"Example Model Accuracy: {accuracy_example*100:.2f}%")
print(f"Example Model Total Parameters: {params_example}")
```

---

### Parametric Experiment Loop

We now create a loop to systematically vary the number of hidden layers (depth) and the number of hidden units (breadth). For each combination, we train the model and record the final accuracy and total trainable parameters.

```python
import matplotlib.pyplot as plt

# Define ranges for hidden layers and hidden units
hidden_layers_range = [1, 2, 3, 4, 5]         # Number of hidden layers
hidden_units_range = range(4, 101, 3)           # Number of units from 4 to 100 in steps of 3

# Store results: Use a dictionary to map n_layers -> (list of accuracies, list of parameters, list of n_units)
results = {}

for n_layers in hidden_layers_range:
    accuracies = []
    params_list = []
    for n_units in hidden_units_range:
        model = ANNIris(n_units=n_units, n_layers=n_layers)
        acc, params = train_model(model, X_tensor, y_tensor, num_epochs=2500, lr=0.01)
        accuracies.append(acc * 100)  # Convert to percentage
        params_list.append(params)
    results[n_layers] = {
        "n_units": list(hidden_units_range),
        "accuracies": accuracies,
        "params": params_list
    }
    print(f"Completed: {n_layers} hidden layer(s)")

# Example: Plot accuracy vs. number of hidden units for each number of hidden layers
plt.figure(figsize=(10, 6))
for n_layers in hidden_layers_range:
    plt.plot(results[n_layers]["n_units"], results[n_layers]["accuracies"],
             marker='o', label=f'{n_layers} hidden layer(s)')
plt.xlabel("Number of Hidden Units (per layer)")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs. Hidden Units for Varying Depth")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Visualizations & Results

### Accuracy vs. Hidden Units for Different Depths

The above code produces a plot where:
- The **x-axis** represents the number of hidden units per layer.
- The **y-axis** represents the final model accuracy (in percent).
- Each line (different color) represents a different number of hidden layers (depth).

### Additional Analysis: Total Parameters vs. Accuracy

We can also examine if there is any correlation between the total number of trainable parameters and accuracy.

```python
# Flatten data to plot total parameters vs. accuracy across all models
all_params = []
all_accuracies = []

for n_layers in hidden_layers_range:
    all_params.extend(results[n_layers]["params"])
    all_accuracies.extend(results[n_layers]["accuracies"])

plt.figure(figsize=(10, 6))
plt.scatter(all_params, all_accuracies, c='blue', alpha=0.6)
plt.xlabel("Total Number of Trainable Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Total Number of Parameters")
plt.grid(True)
plt.show()

# Compute correlation coefficient
correlation = np.corrcoef(all_params, all_accuracies)[0, 1]
print(f"Correlation between total parameters and accuracy: {correlation:.3f}")
```

> **Observation:**  
> In our experiment, the correlation between the total number of parameters and accuracy might be very low—indicating that more parameters do not necessarily guarantee better performance.

---

## Discussion & Insights

- **Flexible Model Definition:**  
  Defining your own model classes (instead of using `nn.Sequential`) allows you to easily vary both the depth and breadth of your network. This flexibility is critical for exploring model architectures beyond simple stacking.

- **Depth vs. Breadth Trade-off:**  
  - **Shallow Models:** Tend to learn faster and might achieve high accuracy quickly, but may be limited in representing complex data patterns.
  - **Deep Models:** Can capture more abstract representations, but increasing depth too far can lead to diminishing returns or even lower accuracy if not trained adequately.
  
- **Total Parameters:**  
  Our experiment shows that the total number of trainable parameters does not have a straightforward relationship with performance. Architecture (the arrangement of layers and units) and other hyperparameters (learning rate, epochs, optimizer) are equally important.

- **Practical Takeaway:**  
  When designing deep learning models, carefully consider both depth and breadth. Empirical experiments like this are invaluable for understanding the optimal architecture for a given problem.

---

## Additional Explorations

- **Vary Training Epochs:**  
  Investigate how longer or shorter training durations interact with model depth and breadth.
- **Learning Rate & Optimizer:**  
  Experiment with different learning rates or optimizers to see their effect on deeper vs. shallower networks.
- **Regularization Techniques:**  
  Later in the course, explore methods such as dropout, weight decay, and batch normalization to see how they affect performance in models of different complexity.

---

## References & Further Reading

- **PyTorch Documentation:**  
  - [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)  
  - [torch.nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)
- **Deep Learning Books:**  
  - *Deep Learning* by Goodfellow, Bengio, and Courville.
- **Relevant Tutorials:**  
  - [PyTorch Tutorials: Building Custom Models](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

---

*End of Note*

These comprehensive notes cover the conceptual background of model depth versus breadth, the practical implementation of a flexible model class in PyTorch, and the design of a parametric experiment to study how these factors affect performance. Happy experimenting and deep learning!