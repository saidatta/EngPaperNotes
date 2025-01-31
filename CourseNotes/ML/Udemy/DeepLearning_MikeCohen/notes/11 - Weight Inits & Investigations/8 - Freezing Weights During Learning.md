## 1. Overview

**Freezing weights** in a deep neural network means **preventing** certain layers' weights from updating via backpropagation. Instead, those layers remain **fixed** (unchanged) throughout training. In PyTorch, this is achieved by setting `requires_grad=False` for the parameters you want to freeze.

### Key Points
- **When**: Typically done in **transfer learning** or when certain layers are *pre-trained* and shouldn’t be modified.  
- **How**: Flip the `requires_grad` toggle to **False** for the relevant parameters.  
- **Result**: Those layers *do not learn*, so only the remaining layers (with `requires_grad=True`) update their weights.

---

## 2. Conceptual Background

1. **Why Freeze?**  
   - In **transfer learning**, you might have a **large pre-trained model** (e.g., on ImageNet) where lower layers represent general features. You keep those intact (freeze) and **fine-tune** only higher (or output) layers.  
   - This saves **time**, **compute**, and preserves **knowledge** gained in the original training.

2. **What Happens?**  
   - When you freeze a layer, its gradients become zero. That layer’s parameters **never update**.  
   - The rest of the network can still learn if their `requires_grad=True` flags remain set.

3. **Implementation**:  
   - In PyTorch, each parameter (weight, bias) is a `torch.Tensor` with an attribute `.requires_grad`.  
   - Default is `True` for trainable parameters, but setting `param.requires_grad = False` *freezes* them.

---

## 3. PyTorch Example: Freezing a Layer

We’ll illustrate with a **MNIST** feedforward network. Our plan:
1. Build the network.  
2. Disable (`requires_grad=False`) certain layers mid-training.  
3. Observe the **impact** on training curves.

### 3.1. Data Loading (MNIST)

```python
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST transforms and data
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

### 3.2. Model Architecture

```python
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)        # raw logits
        return x
```
```

**Note**: This is a moderate architecture for demonstration.

### 3.3. Simple Training Function

We define a function that allows us to **turn layers on/off** mid-training. Specifically, we freeze certain layers for the first half of epochs, then unfreeze them in the second half.

```python
```python
def train_model(net, train_loader, test_loader, epochs=100, lr=0.001):
    optimizer = optim.SGD(net.parameters(), lr=lr)  # small LR, slow training
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc_hist = []
    test_acc_hist  = []
    
    for epoch_i in range(epochs):
        
        # Example condition: freeze all but output layer for first half
        if epoch_i < (epochs//2):
            # Freeze everything except 'out' layer
            for name, param in net.named_parameters():
                if 'out' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # unfreeze entire network
            for name, param in net.named_parameters():
                param.requires_grad = True
        
        # TRAIN
        net.train()
        correct_train = 0
        total_train   = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_hat = net(Xb)
            loss  = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()
            
            preds = y_hat.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train   += len(yb)
        
        train_acc = correct_train / total_train
        train_acc_hist.append(train_acc)
        
        # EVAL
        net.eval()
        correct_test = 0
        total_test   = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                out_t = net(X_t)
                preds_t = out_t.argmax(dim=1)
                correct_test += (preds_t == y_t).sum().item()
                total_test   += len(y_t)
        test_acc = correct_test / total_test
        test_acc_hist.append(test_acc)
    
    return train_acc_hist, test_acc_hist
```
```

**Key**:
- For epoch < (epochs//2), we freeze fc1, fc2 (by `requires_grad=False`), allowing only `out` layer to learn.  
- For epoch >= (epochs//2), we unfreeze everything.

---

## 4. Running the Experiment

### 4.1. Initialization and Training

```python
```python
net_freeze = MNISTNet()
train_acc, test_acc = train_model(net_freeze, train_loader, test_loader, epochs=100, lr=0.001)

# Plot
epochs_arr = np.arange(len(train_acc))
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(epochs_arr, train_acc, label='Train Acc')
plt.plot(epochs_arr, test_acc,  label='Test Acc')
plt.axvline(x=(len(train_acc)//2), color='r', linestyle='--', label='Unfreeze point')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Freezing fc1, fc2 for the first half, then unfreezing')
plt.legend()
plt.show()
```
```

**What to Expect**:
- First half of training → Only output layer (`out`) updates. Performance may barely improve or remain low.  
- Once layers are **unfrozen** → All layers can learn, leading to a sudden improvement in training and test accuracy.

---

## 5. Observations & Interpretations

1. **Frozen Period**:  
   - If only the final (output) layer trains, the model can only **learn** a linear mapping from the **frozen** embeddings to classes.  
   - Accuracy remains low unless the final layer alone can glean enough from the fixed features.

2. **Unfreeze**:  
   - Once all layers have `requires_grad=True`, the deeper parts can adjust, enabling normal improvement.  
   - Accuracy then climbs, though it might take **more** epochs to reach typical ~95% (for MNIST) because half the training time was “wasted.”

3. **Practical Relevance**:  
   - Usually, we **freeze** layers when they are **pre-trained** and we trust those features. We only fine-tune the higher layers or output layers for a new task.  
   - In this **toy** example, we froze layers that started from random initialization → it’s unlikely to help.  
   - The real usage is in **transfer learning** scenarios.

---

## 6. How to Freeze Specific Layers

### 6.1. Direct Access

If you want to freeze only `fc1`:

```python
```python
for name, param in net.named_parameters():
    if 'fc1' in name:
        param.requires_grad = False  # freeze fc1
    else:
        param.requires_grad = True   # default for others
```
```

### 6.2. Using `.eval()` vs. `.train()`

Be aware that:
- `.eval()` vs. `.train()` modes relate to **inference** vs. **training** behaviors (dropout, batchnorm), not weight freezing.  
- Setting `param.requires_grad=False` is the *only* way to freeze a parameter.

---

## 7. Summary and Next Steps

- **Freezing Weights** = Setting `requires_grad=False` to skip backprop updates for those parameters.  
- Main usage is **transfer learning**, where we preserve learned features in lower layers.  
- **Demos**: Letting only part of the network learn can hamper performance unless those frozen parts are already well-trained.

**Next**:
- You’ll see how **transfer learning** uses this concept heavily, especially in **computer vision** (e.g., freezing large CNN backbones) and other pre-trained models (e.g., Transformers in NLP).

---

## 8. References

- **PyTorch**: [Handling `requires_grad`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.requires_grad).  
- **Transfer Learning**: [Official PyTorch Tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html).  
- **Deep Learning** Book (Goodfellow et al.): Sections on advanced training and knowledge transfer.

---
```

**How to Use These Notes in Obsidian**:

1. **Create a new note** named something like `Freezing_Weights.md`.  
2. **Paste** the above markdown (including the frontmatter `---`).  
3. Add or alter any **internal links** (e.g., `[[AdvancedTopicsInTraining]]`) as needed.  
4. Optionally add your own headings, summary, or references.

These notes convey the conceptual rationale for **freezing weights** and demonstrate **how** to do it in PyTorch, highlighting why it’s especially relevant to **transfer learning**.