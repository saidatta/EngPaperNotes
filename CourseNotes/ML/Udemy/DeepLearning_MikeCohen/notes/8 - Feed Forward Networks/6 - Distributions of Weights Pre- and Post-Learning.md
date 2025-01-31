aliases: [Weight Distributions, FFN, Pre-training, Post-training, Histograms]
tags: [Deep Learning, MNIST, Weights, Visualization, PyTorch]
## Overview
In this lecture, we explore **how the distribution of weights changes** throughout the training process of a Feedforward Network (FFN) on MNIST. We'll visualize how initially **randomly initialized** weights evolve as the model learns to classify handwritten digits. This approach provides deeper insights into the model's **internal dynamics** and lays a foundation for further studies on topics like **weight initialization** strategies and **model interpretability**.

---
## 1. Motivation

1. **Beyond Just Running Models**  
   - We aim to develop **intuitive understanding** of neural network internals—how weights evolve, why they assume certain distributions, and how they relate to model performance.  
2. **Random Initialization**  
   - We typically begin with weights drawn from some distribution (e.g., uniform or Gaussian).  
   - **Learning** via backpropagation **shifts** and **sculpts** this distribution to separate classes effectively.  
3. **Preview of Future Topics**  
   - Later in the course, you will learn more about **weight initialization** methods (e.g., **Xavier**, **He** initialization) and additional ways to investigate trained weights.

---

## 2. Data Setup & Normalization

Below, we assume we have the **MNIST** training data loaded in a standard way, normalized to \([0,1]\). The code is similar to previous MNIST feedforward tutorials.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Load partial MNIST (e.g., ~20k images, CSV format)
mnist_path = "/content/sample_data/mnist_train_small.csv"
mnist_data = np.loadtxt(mnist_path, delimiter=",")

# 2. Separate labels and pixels
labels_np = mnist_data[:, 0].astype(int)
pixels_np = mnist_data[:, 1:].astype(float)

# 3. Normalize pixel values to [0,1]
pixels_np /= 255.0

# 4. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# 5. Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 6. Create Dataset and DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
```

---

## 3. Defining the Feedforward Model

We define an FFN with:
- **Input Layer**: 784 → 64
- **Hidden Layer**: 64 → 32
- **Output Layer**: 32 → 10

```python
class MNIST_FFN(nn.Module):
    def __init__(self):
        super(MNIST_FFN, self).__init__()
        self.input = nn.Linear(784, 64)
        self.fc1   = nn.Linear(64, 32)
        self.fc2   = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)  # or output raw logits + CrossEntropyLoss
```

### 3.1 Inspecting the Model’s Layers

```python
net = MNIST_FFN()
print(net)  # summary of layers

# Example: look at net.input
print(net.input)
# This object holds .weight, .bias, and other attributes
print("Weight shape in the input layer:", net.input.weight.shape)
```

- By default, **PyTorch** initializes weights using specific heuristics (often a form of **uniform** or **Kaiming** approach).  
- We can inspect the actual weight **tensor**:

```python
print("Initial input layer weights:\n", net.input.weight)
```

---

## 4. Visualizing the Initial Weight Distributions

We can plot a **histogram** of these weights:

```python
weights_input = net.input.weight.detach().flatten().numpy()

plt.hist(weights_input, bins=40, color='steelblue', alpha=0.7)
plt.title("Histogram of Initial Input Layer Weights")
plt.xlabel("Weight Value")
plt.ylabel("Count")
plt.show()
```

> Typically, initial distributions are near-zero, often uniform or normal depending on PyTorch defaults. Here, it may appear roughly **uniform** or slightly trapezoidal.

---

## 5. Aggregating All Weights

Instead of just one layer, we can gather **all layer weights** in a single histogram.

**Helper Function**: Returns histogram data (\(x\) and \(y\)) of all network weights.

```python
def get_weight_histogram(model, num_bins=100, min_val=-0.8, max_val=0.8):
    """
    Stacks all trainable weights from all layers into a single 1D vector,
    then computes a histogram.
    
    Returns:
      histx (bin centers),
      histy (bin counts).
    """
    # Collect weights from each layer into one tensor
    all_weights = []
    for param_name, param in model.named_parameters():
        if 'weight' in param_name:  # we skip biases here
            w = param.detach().flatten()
            all_weights.append(w)
    all_weights = torch.cat(all_weights)
    
    histy, bin_edges = np.histogram(all_weights.numpy(), bins=num_bins, range=(min_val, max_val))
    # mid-points of bins for plotting on x-axis
    histx = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return histx, histy

# Quick test
net = MNIST_FFN()
histx, histy = get_weight_histogram(net)
plt.bar(histx, histy, width=0.015, color='steelblue')
plt.title("All Weights Histogram (Initial)")
plt.show()
```

---

## 6. Training the Model & Tracking Weights Over Epochs

We modify our **training loop** to:
1. Train for N epochs.
2. **At each epoch**:  
   - Compute histogram data for the **current** model weights.  
   - Store for later analysis/visualization.

### 6.1 Training Function

```python
def train_and_track_weights(model, train_loader, test_loader, num_epochs=100):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    # For logging
    train_accs, test_accs = [], []
    histx, histy_epochs = None, []
    
    for epoch in range(num_epochs):
        # --- Record histogram at the start of each epoch ---
        hx, hy = get_weight_histogram(model, num_bins=100, min_val=-0.8, max_val=0.8)
        histx = hx  # same bin centers each time
        histy_epochs.append(hy)
        
        # Training
        model.train()
        batch_accs = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred_log = model(Xb)
            loss = criterion(y_pred_log, yb)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(y_pred_log, dim=1)
            acc = (preds == yb).float().mean().item()
            batch_accs.append(acc)
        
        train_acc = np.mean(batch_accs)
        train_accs.append(train_acc)
        
        # Evaluation on test
        model.eval()
        batch_accs_test = []
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                y_pred_log_test = model(Xb_t)
                _, preds_t = torch.max(y_pred_log_test, dim=1)
                acc_t = (preds_t == yb_t).float().mean().item()
                batch_accs_test.append(acc_t)
        test_acc = np.mean(batch_accs_test)
        test_accs.append(test_acc)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
            
    return histx, histy_epochs, train_accs, test_accs
```

### 6.2 Run the Training & Log Histograms

```python
model = MNIST_FFN()
histx, histy_epochs, train_accs, test_accs = train_and_track_weights(
    model, train_loader, test_loader, num_epochs=100
)

# Final accuracy
print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
print(f"Final Test Accuracy:  {test_accs[-1]*100:.2f}%")
```

Expect ~**95%** test accuracy, consistent with a simple feedforward approach on a partial MNIST dataset.

---

## 7. Results: Visualizing Weight Distributions Over Time

### 7.1 Overlaid Histograms at Different Epochs

We can plot **multiple lines**—each line is a weight histogram at a given epoch. We'll color them from an earlier epoch (warm color) to a later epoch (cool color).

```python
import matplotlib.cm as cm

# histy_epochs: list of histogram counts over epochs
num_epochs = len(histy_epochs)

plt.figure(figsize=(10,6))
for i, hy in enumerate(histy_epochs):
    # Create a color that transitions from orange to blue
    color_frac = i / (num_epochs - 1)
    color = cm.coolwarm(color_frac)
    plt.plot(histx, hy, color=color, alpha=0.5)

plt.title("Evolution of Weight Distributions Over Training")
plt.xlabel("Weight Value")
plt.ylabel("Count")
plt.show()
```

> **Interpretation**:  
> - Initially, you might see a roughly **uniform** distribution near zero.  
> - As training progresses, **tails** become heavier (more large positive/negative weights).  
> - The histogram often skews towards a shape more resembling a **Gaussian**.

### 7.2 A “Stacked” Visualization (Density Over Epochs)

We can visualize each histogram as a row in a 2D “heatmap” or “waterfall” plot.

```python
# Convert histy_epochs into a 2D array
histy_matrix = np.array(histy_epochs)

# Plot as image
plt.figure(figsize=(8,6))
plt.imshow(histy_matrix.T, origin='lower', aspect='auto',
           extent=[0, num_epochs, histx[0], histx[-1]], cmap='hot')
plt.colorbar(label='Count')
plt.title("2D Visualization of Weight Histograms")
plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.show()
```

> **Note**:  
> - The horizontal axis = epochs (0 to num_epochs).  
> - The vertical axis = weight bin centers.  
> - Color intensity shows how many weights fall in each bin at each epoch.  
> - You might see the center portion lighten over time (fewer near zero) and the tails strengthen (more large weights).

---

## 8. Interpretation & Insights

1. **Initial Uniform Distribution**  
   - PyTorch often initializes layer weights uniformly in a small range (\([-a, a]\)).  
   - This avoids symmetry-breaking issues in training.

2. **Drift Toward Larger Magnitudes**  
   - As the network learns, certain weights grow larger (positive/negative) to capture the discriminative features for digit classification.

3. **Gaussian-Like Distribution**  
   - Over many updates, the distribution can resemble a **rough Gaussian**, though it may not be perfectly symmetrical.

4. **Practical Relevance**  
   - Monitoring weight distributions can diagnose **over-saturation** (weights growing too large) or **under-training** (weights staying near zero).  
   - In advanced topics (like batch normalization, dropout, etc.), weight dynamics can differ significantly.

---

## 9. Key Takeaways

- **Inspecting Weights**:  
  - Provides a window into the network’s internal representations.  
  - Helps understand how initialization and training shape the parameter space.
- **Dynamic Evolution**:  
  - Weights often spread out from small initial ranges, matching the complexity of the classification task.
- **Relates to Performance**:  
  - While this doesn’t directly give the classification accuracy, it *does* correlate with how the model organizes itself to distinguish classes.
- **Future Topics**:  
  - Various **initialization** schemes will yield different starting distributions.  
  - **Regularization** (like weight decay) can constrain how large weights become.

---

## 10. Further Exploration

- **Compare Different Optimizers**: Adam vs. SGD. Notice how the distribution evolves differently (Adam might converge faster).  
- **Check Per-Layer Histograms**: Some layers may have narrower or broader weight distributions.  
- **Add Weight Regularization**: L2-regularization (weight decay) affects the histogram’s tail behavior.  
- **View Biases**: Also inspect bias distributions to see if they converge to particular offsets.

---

**End of Notes – “Distributions of Weights Pre- and Post-Learning”**  
```