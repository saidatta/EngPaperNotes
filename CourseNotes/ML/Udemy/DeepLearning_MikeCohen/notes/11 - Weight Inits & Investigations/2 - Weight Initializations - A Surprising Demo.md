title: "Weight Initializations: A Surprising Demo"
tags: [deep-learning, weight-initialization, pytorch, experiments]

## 1. Overview
In deep learning, **weight initialization** can have a profound impact on **if** and **how** a model **learns**. This lecture demonstrates a **surprising** phenomenon:  
- **All-zero** (or all-constant) weight initialization → The model **fails** to learn anything meaningful.  
- **Random** weight initialization → Normal, expected learning.

### Key Takeaways
- Deep networks **require** at least some variation or randomness in initial weights to **break symmetry** and allow gradients to flow properly.  
- Setting **all** weights to the **same** value (e.g., 0 or 1) can completely stall learning.

---

## 2. Motivation and Experiment Setup

### 2.1. Typical MNIST Performance
- Using a **feedforward** neural network on MNIST, we often see around **95%** test accuracy with a small architecture and standard training (~10 epochs).  

### 2.2. What Happens if We Initialize Weights to Zero?
- You might expect the network to still adjust weights via **backpropagation**—but in practice, the model is stuck at **~10% accuracy** (random guessing) and never improves.

**Question**: *Why does uniform initialization kill learning?*  
- We will defer the detailed explanation to later, but keep in mind it has to do with **symmetry** and **gradient** updates.

---

## 3. Demonstration in PyTorch

Below is a **toy code** illustrating the difference between **normal** (random) initialization and **constant** initialization (all zeros or ones). The code uses **MNIST** for a quick experiment.

### 3.1. Data Preparation

```python
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST dataset with standard transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
```
```

**Note**: This is standard MNIST preprocessing. Normalizing with mean ~0.1307 and std ~0.3081 is common.

---

### 3.2. Model Definition

We define a **simple feedforward** network with two hidden layers:

```python
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)       # raw logits
        return x
```
```

---

### 3.3. Training Function

```python
```python
def train_model(net, train_loader, test_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc_history = []
    test_acc_history  = []
    
    for epoch in range(epochs):
        net.train()
        correct = 0
        total   = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = net(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            
            _, pred_labels = torch.max(y_hat, dim=1)
            correct += (pred_labels == y_batch).sum().item()
            total   += len(y_batch)
        
        train_acc = correct / total
        train_acc_history.append(train_acc)
        
        # Evaluate on test set
        net.eval()
        correct_test = 0
        total_test   = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                out_t = net(X_t)
                _, preds_t = torch.max(out_t, dim=1)
                correct_test += (preds_t == y_t).sum().item()
                total_test   += len(y_t)
        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
    
    return train_acc_history, test_acc_history
```
```

**Explanation**:  
- We track **train accuracy** per epoch and also compute **test accuracy** to measure performance.

---

## 4. Experiments and Results

### 4.1. Baseline Model (Default Random Initialization)

```python
```python
# 1) Create a fresh network - default random initialization
net_baseline = MNISTNet()

# 2) Train
epochs = 10
train_acc_base, test_acc_base = train_model(net_baseline, train_loader, test_loader, epochs=epochs, lr=0.001)
```
```

**Typical Outcome**:  
- Final test accuracy around **90–95%** after ~10 epochs.  
- The model shows normal **learning** curves.

### 4.2. Initialize Only One Layer to Zeros

Let's see how partial zero initialization affects learning:

```python
```python
net_zero_layer = MNISTNet()

# Force the weights of fc1 to be all zeros
with torch.no_grad():
    net_zero_layer.fc1.weight.data = torch.zeros_like(net_zero_layer.fc1.weight.data)
    net_zero_layer.fc1.bias.data   = torch.zeros_like(net_zero_layer.fc1.bias.data)

train_acc_zeroL, test_acc_zeroL = train_model(net_zero_layer, train_loader, test_loader, epochs=epochs, lr=0.001)
```
```

**Observation**:  
- The network may still **learn** somewhat, because **subsequent layers** have random initialization.  
- Performance might degrade slightly, but not always drastically if **other layers** can still adjust.

### 4.3. Initialize **All** Parameters to Zeros

```python
```python
net_all_zero = MNISTNet()

# Set all learnable parameters to zero
with torch.no_grad():
    for name, param in net_all_zero.named_parameters():
        param.data = torch.zeros_like(param.data)

train_acc_allZ, test_acc_allZ = train_model(net_all_zero, train_loader, test_loader, epochs=epochs, lr=0.001)
```
```

**Expected Behavior**:  
- **No** improvement in accuracy beyond random chance (~10%).  
- The weights remain at zero throughout training (no gradient-based updates effectively break the symmetry).

#### Checking the Final Weights

```python
```python
for name, param in net_all_zero.named_parameters():
    print(name, param.data.unique())  # should show only tensor([0.])
```
```

You’ll see **only zeros** in the final weights.

### 4.4. Initialize All Parameters to Ones (or Another Constant)

```python
```python
net_ones = MNISTNet()

with torch.no_grad():
    for _, param in net_ones.named_parameters():
        param.data = torch.ones_like(param.data)  # Now everything is 1

train_acc_ones, test_acc_ones = train_model(net_ones, train_loader, test_loader, epochs=epochs, lr=0.001)
```
```

**Observation**:  
- The network might learn **slightly** more than the all-zero case, but still **far below** normal performance.  
- Often stuck at 20–40% accuracy even after many epochs.

---

## 5. Visualizing the Learning Curves

Compare the different initializations:

```python
```python
plt.figure(figsize=(8,5))

plt.plot(range(1, epochs+1), test_acc_base, label='Baseline (Random Init)')
plt.plot(range(1, epochs+1), test_acc_zeroL, label='Zero Init (Only fc1)')
plt.plot(range(1, epochs+1), test_acc_allZ, label='All Zero')
plt.plot(range(1, epochs+1), test_acc_ones,  label='All Ones')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Different Weight Initializations on MNIST')
plt.legend()
plt.show()
```
```

**Typical Graph**:  
- **Baseline** quickly approaches ~90–95%.  
- **All Zero** remains ~10%.  
- **All Ones** might rise slightly above chance but plateaus far below ~95%.  
- **Zero Init (One Layer)** may partially learn but is generally worse than the baseline.

---

## 6. Why Does This Happen?

Although the lecture **teases** the explanation, the main points are:

1. **Symmetry**: With identical initial weights, each neuron in a layer receives **identical** gradients, so they all remain identical throughout training. No meaningful learning can differentiate them.  
2. **Zero Grad** Phenomenon**: When all weights are zero, the backprop gradients for those parameters can end up zero (especially in symmetrical architectures), preventing any updates.  
3. **Bias to a Single Value**: There's no variance to “push” different neurons to learn different features.

**We’ll** dive deeper into the **theory** and **correct** random weight initialization strategies in subsequent videos.

---

## 7. Summary & Next Steps

1. **Constant Initialization** (all 0, all 1, or any single value) → The network **fails** to train or severely underperforms.  
2. **Random Initialization** is crucial to break neuron **symmetry** and let each weight learn unique features.  
3. In **future** lessons, we’ll learn about advanced strategies (e.g., **Xavier/Glorot**, **He/Kaiming** initialization) that improve convergence by controlling variance in forward/backward passes.

**Action Item**: *Ponder* the deeper reasons behind these observations. The next lectures will formalize **why** random initialization is essential and how to **implement** it properly.

---

## 8. Further Reading

1. **Goodfellow, I., Bengio, Y., & Courville, A.**: *Deep Learning* – Chapter on **parametrized models**, emphasizing the importance of proper initialization.  
2. **Krizhevsky et al.** (AlexNet paper, 2012) – Early demonstration of random initialization for CNNs.  
3. **Xavier/Glorot Initialization**: [Original Paper by Glorot and Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a.html)  
4. **He (Kaiming) Initialization**: [He et al. (2015)](https://arxiv.org/abs/1502.01852)

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your vault (e.g., `WeightInit_SurprisingDemo.md`).  
2. **Copy and paste** the entire content (including frontmatter `---`) above.  
3. Optionally adjust headings, tags, or add your own links like `[[Regularization Notes]]`, etc.  

These notes provide a **thorough** demonstration of how **all-constant** weight initialization prevents or severely hinders learning, underlining the importance of **random** initializations in **deep neural networks**.