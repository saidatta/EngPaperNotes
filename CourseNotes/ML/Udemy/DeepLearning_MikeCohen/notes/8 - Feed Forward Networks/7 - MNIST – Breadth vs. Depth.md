aliases: [Breadth vs. Depth, MNIST, CodeChallenge, Hidden Layers, Hidden Units]
tags: [Deep Learning, Classification, Feedforward, Architecture]
## Overview
This CodeChallenge explores how **model complexity** (in terms of **depth** and **breadth**) impacts classification performance on **MNIST**. Specifically, we will:
- **Vary the number of hidden layers** (1, 2, or 3).
- **Vary the number of hidden units per layer** (\(50, 100, 150, 200, 250\)).

The experiment systematically tests each combination and measures **train/test accuracy**. The results highlight how **deeper** and **wider** networks can often yield better performance for sufficiently complex tasks (like digit classification).

> **Note**: Because we train multiple models (5 possible widths \(\times\) 3 possible depths = 15 runs), **this experiment can be time-consuming** (~20–30 minutes) if each run takes a few minutes.

---
## 1. Key Concepts

1. **Breadth (Width)**  
   - The **number of units per hidden layer**.  
   - In this challenge, we keep the same number of units for every hidden layer in a given model.

2. **Depth**  
   - The **number of hidden layers**.  
   - We will try **1, 2, and 3** hidden layers between input and output.

3. **Performance**  
   - Typically measured via **accuracy** on the train and test sets.  
   - Deeper/wider networks often lead to higher accuracy on **complex** data (like MNIST).  
   - For **simpler** tasks (e.g., Iris dataset), additional layers/units can lead to **overfitting** or yield diminishing returns.

---

## 2. Data Setup

We assume you have a partial MNIST dataset loaded from CSV (e.g., `mnist_train_small.csv` with ~20,000 samples). The standard steps:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Load partial MNIST
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# 2. Separate labels and pixel values
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

# 3. Normalize to [0,1]
pixels_np /= 255.0

# 4. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# 5. Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 6. Create Dataset & DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)

print("Train set size:", len(train_loader)*batch_size)
print("Test set size: ", len(test_loader)*batch_size)
```

---

## 3. Creating a Flexible Model Class

We want a model class that:
1. Takes an **arbitrary** number of hidden layers.
2. Each hidden layer has the **same** number of units (breadth).

### 3.1 Using `nn.ModuleDict`
We can dynamically construct multiple layers inside a **ModuleDict**. Here’s an example approach:

```python
class FlexibleFFN(nn.Module):
    def __init__(self, num_hidden_layers=1, units_per_layer=50):
        super(FlexibleFFN, self).__init__()
        
        # Store config
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer   = units_per_layer
        
        # A ModuleDict to store each layer
        # Input layer: 784 -> (units_per_layer)
        layers = nn.ModuleDict()
        layers["input"] = nn.Linear(784, units_per_layer)
        
        # Hidden layers: (units_per_layer) -> (units_per_layer)
        for i in range(1, num_hidden_layers):
            layer_name = f"hidden_{i}"
            layers[layer_name] = nn.Linear(units_per_layer, units_per_layer)
        
        # Output layer: (units_per_layer) -> 10
        layers["output"] = nn.Linear(units_per_layer, 10)
        
        self.layers = layers
    
    def forward(self, x):
        # Pass through input layer
        x = self.layers["input"](x)
        x = torch.relu(x)
        
        # Pass through hidden layers
        for i in range(1, self.num_hidden_layers):
            layer_name = f"hidden_{i}"
            x = self.layers[layer_name](x)
            x = torch.relu(x)
        
        # Final output layer
        x = self.layers["output"](x)
        
        # Apply log_softmax for classification
        return torch.log_softmax(x, dim=1)
```

- **Input**: 784 features (28x28).
- **Hidden**: \(\text{units\_per\_layer}\) repeated for `num_hidden_layers - 1` layers.
- **Output**: 10 classes.

---

## 4. Training Function

We’ll define a function that:
1. Instantiates a **`FlexibleFFN`** with given `num_hidden_layers` and `units_per_layer`.
2. Trains for a set number of epochs on the MNIST train loader.
3. Returns final **train** and **test** accuracy.

```python
def train_model(num_hidden_layers, units_per_layer, train_loader, test_loader, epochs=20):
    # 1) Create model
    model = FlexibleFFN(num_hidden_layers=num_hidden_layers, units_per_layer=units_per_layer)
    
    # 2) Define loss + optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 3) Train
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred_log = model(Xb)
            loss = criterion(y_pred_log, yb)
            loss.backward()
            optimizer.step()
    
    # 4) Evaluate on training set
    model.eval()
    correct_train, total_train = 0, 0
    with torch.no_grad():
        for Xb, yb in train_loader:
            preds_log = model(Xb)
            _, preds = torch.max(preds_log, dim=1)
            correct_train += (preds == yb).sum().item()
            total_train   += len(yb)
    train_acc = correct_train / total_train
    
    # 5) Evaluate on test set
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for Xb_t, yb_t in test_loader:
            preds_log_t = model(Xb_t)
            _, preds_t = torch.max(preds_log_t, dim=1)
            correct_test += (preds_t == yb_t).sum().item()
            total_test   += len(yb_t)
    test_acc = correct_test / total_test
    
    return train_acc, test_acc
```

> **Note**: For computational reasons, we pick ~**20 epochs**. You can adjust as needed.

---

## 5. Running the Experiment

We systematically vary:
- **Depth**: \( \{1,2,3\} \)
- **Breadth**: \( \{50, 100, 150, 200, 250\} \)

We store **train_acc** and **test_acc** in a results structure:

```python
layers_to_try = [1, 2, 3]                   # number of hidden layers
units_to_try  = [50, 100, 150, 200, 250]    # units per hidden layer

results_train = {}
results_test  = {}

for depth in layers_to_try:
    results_train[depth] = []
    results_test[depth]  = []
    
    for width in units_to_try:
        print(f"Training model with depth={depth}, width={width}...")
        train_acc, test_acc = train_model(
            num_hidden_layers=depth,
            units_per_layer=width,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=20
        )
        results_train[depth].append(train_acc)
        results_test[depth].append(test_acc)

print("Experiment complete.")
```

> This loop may take **20–30+ minutes** depending on hardware and dataset size.

---

## 6. Plotting the Results

We want to plot **train** and **test** accuracy vs. **width**, with separate lines for **depth**.

```python
plt.figure(figsize=(12,5))

# 6.1 Plot Training Acc
plt.subplot(1,2,1)
for depth in layers_to_try:
    plt.plot(units_to_try, results_train[depth], label=f"{depth} hidden layers")
plt.xlabel("Units per layer")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy vs. Width (for different depths)")
plt.legend()

# 6.2 Plot Testing Acc
plt.subplot(1,2,2)
for depth in layers_to_try:
    plt.plot(units_to_try, results_test[depth], label=f"{depth} hidden layers")
plt.xlabel("Units per layer")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Width (for different depths)")
plt.legend()

plt.tight_layout()
plt.show()
```

> **Interpretation**:  
> - Expect **train accuracy** to rise significantly as we add more units/layers.  
> - **Test accuracy** also rises, but might see diminishing returns if the model overfits.  
> - For MNIST, more depth/breadth typically **boosts** performance.

---

## 7. Example Results

A typical outcome might look like:

- **Train Accuracy** approaches **100%** for large networks (200–250 units, 2–3 hidden layers).
- **Test Accuracy** climbs from ~**92–93%** (small single-layer) to **94–97%** (larger/deeper).  

You might see a figure akin to:

> ![Sample result concept](https://via.placeholder.com/600x300?text=Train+and+Test+Accuracy+Plot)

Where each line represents a different depth (1, 2, or 3 layers), and the horizontal axis is the width (50 to 250). The lines generally slope upward as width increases, and the deeper lines might lie above the shallower lines.

---

## 8. Discussion & Insights

1. **Complex Tasks Benefit from Larger Models**  
   - MNIST is **complex enough** that deeper/wider networks continue to boost accuracy.
2. **Contrast with Simpler Datasets**  
   - On simpler tasks (e.g., **Iris**), extra layers yield **minimal gains** or can **hurt** performance.  
   - Deeper architecture can be **overkill** for low-dimensional tasks.
3. **Experimental Mindset**  
   - There's **no universal** “optimal depth or width.”  
   - Each dataset can respond differently, so we often rely on experiments & tuning.

---

## 9. Key Takeaways

- **Breadth vs. Depth**: For MNIST, adding more layers (depth) and more units (width) **significantly** improves performance—unlike simpler tasks (e.g., Iris) where benefits saturate quickly.
- **Computational Cost**: More layers/units → more parameters → **longer training**. Expect trade-offs in real-world constraints.
- **General Advice**: 
  - Start with moderate architecture (1–2 hidden layers, ~100–200 units).  
  - Increase gradually to see diminishing returns or overfitting signs.  
  - Use **experiment-driven** approach, because deep learning performance is data/task dependent.

---

## 10. Further Explorations

- **Regularization**: Try adding dropout or weight decay for bigger networks to see if that further improves generalization.
- **Different Optimizers**: Adam vs. SGD might converge faster or yield different final accuracies.
- **Time vs. Accuracy**: Plot how training time scales with width/depth. This is crucial in practical deployments.
- **Even Deeper**: Evaluate 4–5 hidden layers (with caution about training time) to observe further improvements or overfitting.

---

**End of Notes – "CodeChallenge: MNIST and Breadth vs. Depth"**  
```