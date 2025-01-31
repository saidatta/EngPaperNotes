aliases: [Data Size vs Network Size, Model Complexity, PyTorch Experimentation]
tags: [Data, Deep Learning, Model Architecture, Experiment, PyTorch]
## Overview
This lecture explores how **the amount of training data** and **the size (breadth & depth) of a feedforward network** interact to affect model performance. We revisit a synthetic “QWERTY” 3-class dataset and systematically vary:
1. **Number of layers** (network depth)
2. **Number of units per layer** (network breadth)
3. **Total samples** in the training data

We fix the **total number of hidden units** across the network at 80, but distribute them differently across the hidden layers. You’ll see that some architectures (e.g., fewer layers with more units each) can outperform deeper, narrower networks, especially when data is limited—highlighting the **unpredictable** nature of how models perform without empirical testing.

---

## 1. Motivation

- **Previously**: We saw that **wider** or **deeper** networks can each improve performance in different settings.  
- **Now**: We fix the total “budget” of **80 hidden units** and vary how many layers share those units.
  - E.g., 1 layer w/ 80 units vs. 20 layers w/ 4 units each.
- **Also** vary the **data size**: from 50 samples per class up to 550 per class.  

**Goal**: Understand how **architecture** (wide vs. deep) interacts with **training data size** to produce drastically different learning outcomes.

---

## 2. Generating the QWERTY Dataset

We create a function that:
1. Constructs a **3-class** dataset in 2D space (like previous QWERTY examples).
2. Returns a **dictionary** containing train/test splits and relevant PyTorch dataloaders.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def create_qwerty_data(nPerClust=50, test_size=0.5):
    """
    Generates a 3-class dataset ('QWERTY' style) with 2D features.
    Returns a dictionary with:
      - data (numpy array, shape [3*nPerClust, 2])
      - labels (numpy array, shape [3*nPerClust,])
      - train_loader, test_loader (DataLoader objects)
    """
    # 3 cluster centers
    A_center = [ 1,  2]
    B_center = [ 4,  4]
    C_center = [ 4, -2]
    
    # Generate data for each cluster
    A_data = np.random.randn(nPerClust, 2) + A_center
    B_data = np.random.randn(nPerClust, 2) + B_center
    C_data = np.random.randn(nPerClust, 2) + C_center
    
    # Labels: 0, 1, 2
    A_labels = np.zeros((nPerClust, ))
    B_labels = np.ones((nPerClust, ))
    C_labels = 2 * np.ones((nPerClust, ))
    
    # Concatenate
    data = np.vstack((A_data, B_data, C_data))
    labels = np.concatenate((A_labels, B_labels, C_labels))
    
    # Convert to PyTorch tensors
    dataT   = torch.tensor(data, dtype=torch.float32)
    labelsT = torch.tensor(labels, dtype=torch.long)
    
    # Create train/test splits
    dataset = TensorDataset(dataT, labelsT)
    data_size = len(labels)
    test_size_abs = int(data_size * test_size)
    
    # Shuffle index
    indices = np.random.permutation(data_size)
    train_idx, test_idx = indices[test_size_abs:], indices[:test_size_abs]
    
    train_dataset = TensorDataset(dataT[train_idx], labelsT[train_idx])
    test_dataset  = TensorDataset(dataT[test_idx],  labelsT[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, drop_last=True)
    
    # Return everything in a dictionary
    return {
        'data': data,
        'labels': labels,
        'train_loader': train_loader,
        'test_loader': test_loader
    }

# Quick test
nSamples_per_class = 50
dataset_info = create_qwerty_data(nSamples_per_class)
X, Y = dataset_info['data'], dataset_info['labels']

# Plot the data
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.5)
plt.title(f"QWERTY Dataset with {nSamples_per_class} samples/class")
plt.show()
```

- We now have **3 classes**, each with `nPerClust` samples. For \(\text{nPerClust} = 50\), total is \(150\) samples.

---

## 3. Defining a Flexible Network for Different Depths

We create a class that:
1. Uses a **variable** number of hidden layers, each having `nUnits` units.
2. Has a final output layer with **3 outputs** (since 3 classes).
3. Allows for an **adjustable** number of hidden layers.

```python
class FFN_Flexible(nn.Module):
    def __init__(self, num_layers=1, num_units=80):
        """
        num_layers: how many hidden layers
        num_units:  how many units in each hidden layer
        (2 inputs -> hidden layers -> 3 outputs)
        """
        super(FFN_Flexible, self).__init__()
        
        self.num_layers = num_layers
        self.num_units  = num_units
        
        # We store layers in a ModuleDict
        # Input layer: 2 -> num_units
        layers = nn.ModuleDict()
        layers["input"] = nn.Linear(2, num_units)
        
        # Hidden layers: each num_units -> num_units
        for i in range(1, num_layers):
            layer_name = f"hidden_{i}"
            layers[layer_name] = nn.Linear(num_units, num_units)
        
        # Output layer: num_units -> 3
        layers["output"] = nn.Linear(num_units, 3)
        
        self.layers = layers
    
    def forward(self, x):
        # Forward pass
        x = self.layers["input"](x)
        x = torch.relu(x)
        
        for i in range(1, self.num_layers):
            layer_name = f"hidden_{i}"
            x = self.layers[layer_name](x)
            x = torch.relu(x)
        
        # final output (raw logits)
        x = self.layers["output"](x)
        return x

# Quick check
test_net = FFN_Flexible(num_layers=1, num_units=80)
print("test_net:", test_net)
```

---

## 4. Training Loop

Below is a generic function that:
1. Creates the model with given `num_layers`, `num_units`.
2. Uses a **CrossEntropyLoss** and **SGD** with `lr=0.01`.
3. Trains for fixed epochs, returning final **train/test accuracy** (or average last few epochs).

```python
def train_model(num_layers, num_units, train_loader, test_loader, epochs=100):
    model = FFN_Flexible(num_layers=num_layers, num_units=num_units)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # We'll store results
    train_accs, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        batch_accs = []
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(logits, 1)
            acc = (preds == yb).float().mean().item()
            batch_accs.append(acc)
        
        train_accs.append(np.mean(batch_accs))
        
        # Evaluate on test
        model.eval()
        batch_accs_test = []
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                logits_t = model(Xb_t)
                _, preds_t = torch.max(logits_t, 1)
                acc_t = (preds_t == yb_t).float().mean().item()
                batch_accs_test.append(acc_t)
        test_accs.append(np.mean(batch_accs_test))
    
    return model, train_accs, test_accs
```

---

## 5. Fixing Total Units to 80

### 5.1 Wide vs. Deep
- If we have **`num_layers=1`**, that layer has **all 80** units.  
- If we have **`num_layers=20`**, each layer might have only **4** units (since \(20 \times 4 = 80\)).  
- We do this for 4 possible layer counts: `[1, 5, 10, 20]`.

### 5.2 Data Variation
- We vary **nPerClust** in `[50, 100, 150, ..., 550]`.  
- This means total samples = `3 * nPerClust` per dataset.

We’ll store the final metrics for each combination.

```python
layer_options = [1, 5, 10, 20]  # hidden layers
data_options  = range(50, 601, 50)  # nPerClust from 50..600 in steps of 50

results = {}
for nL in layer_options:
    results[nL] = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[], 'nData':[]}
```

**Why store?** We will plot them later.

---

## 6. Running the Experiment

```python
def run_experiment():
    # We do 100 epochs of training for each combination
    epochs = 100
    
    for nL in layer_options:
        for nPerC in data_options:
            # Generate data
            data_info = create_qwerty_data(nPerC, test_size=0.5)
            
            # We'll fix the #units by dividing 80 among the #layers
            nUnits_per_layer = 80 // nL  # integer division
            model, train_accs, test_accs = train_model(
                num_layers=nL, 
                num_units=nUnits_per_layer,
                train_loader=data_info['train_loader'],
                test_loader=data_info['test_loader'],
                epochs=epochs
            )
            
            # We'll average over the last 5 epochs to reduce noise
            avg_train = np.mean(train_accs[-5:])
            avg_test  = np.mean(test_accs[-5:])
            
            # Store
            results[nL]['train_acc'].append(avg_train)
            results[nL]['test_acc'].append(avg_test)
            results[nL]['nData'].append(nPerC * 3)  # total data
    return results

# Warning: this can take several minutes
results = run_experiment()
print("Experiment complete!")
```

> This might run for a while depending on your system.

---

## 7. Visualization of Results

Let’s plot **test accuracy** vs. total number of data points (x-axis), with separate lines for each layer configuration.

```python
plt.figure(figsize=(10,5))

for nL in layer_options:
    # x-axis: total data per dataset
    x_data = results[nL]['nData']
    y_acc  = results[nL]['test_acc']
    plt.plot(x_data, y_acc, marker='o', label=f"{nL} layers")

plt.xlabel("Total Training Samples")
plt.ylabel("Test Accuracy (avg last 5 epochs)")
plt.title("Data Size vs. Network Depth (80 total units)")
plt.legend()
plt.show()
```

Optionally, we could also plot **train accuracy** or **loss** similarly.

---

## 8. Example Outcomes & Interpretation

You might see results like:

- **1 hidden layer (80 units)** achieving **high accuracy** with moderate data (e.g., ~80–90%).  
- **5 hidden layers (16 units each)** performing **reasonably well**.  
- **10 hidden layers (8 units each) or 20 hidden layers (4 units each)** might perform **poorly**, especially with smaller data. They can converge slowly or get stuck near chance.

**Key Observations**:
1. **Fewer, wider layers** can do quite well with limited data.  
2. **Deeper, narrower** networks often struggle with the same data.  
3. We see yet again that **empirical testing** is crucial. There’s no universal “best” shape for a network.

---

## 9. Why Does This Happen?

- **Same # of total units** does **not** imply same **# of parameters**.  
  - Deeper networks can end up with fewer parameters or many parameters, depending on how you distribute connections, but often the arrangement yields less capacity to represent the data effectively or they are harder to train with limited data.  
- **Data** limitations: Deeper models may be harder to train well if you don’t have enough samples or if you don’t carefully tune hyperparameters/initialization.

---

## 10. Conclusion & Next Steps

- **Takeaway**: Model size vs. data size is a **crucial** interplay.  
- Sometimes **wider** is better, sometimes **deeper** helps—**always test** empirically.
- In future topics, we’ll see how **regularization**, **batch normalization**, **learning rate schedules**, and **advanced optimizers** can improve deep network training even with smaller datasets.

**End of Notes – "Data: Data Size and Network Size"** 
```