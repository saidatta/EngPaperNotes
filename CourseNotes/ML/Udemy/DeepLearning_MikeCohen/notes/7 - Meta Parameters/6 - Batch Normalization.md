
aliases: [Batch Normalization, BN, Activation Normalization, Deep Networks]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Normalization]

**Batch Normalization** extends the concept of normalizing data from **just the input** of the network to **the outputs/activations of each hidden layer**. By dynamically learning scaling and shifting parameters per layer, batch normalization can stabilize and accelerate training, particularly in **deep** networks.

---
## 1. Rationale

### 1.1 From Data Normalization to Layer Normalization

Previously, we learned that **data normalization** (e.g., z-scoring) helps ensure inputs are on comparable scales:
- Speeds convergence.
- Improves numerical stability.

However, **inside** a deep network, each subsequent layer also transforms data. Even if the **input** is normalized:
1. Weights and biases **shift** and **scale** these features.  
2. Non-linear activations (e.g., ReLU) skew or “clip” values.

Thus, the intermediate activations for deeper layers can also become **unbalanced**. Batch normalization aims to **continuously re-normalize** the activations that flow between layers.

### 1.2 Internal Covariate Shift

- As activations flow through a multi-layer network, their distribution changes (mean, variance, etc.).  
- This shifting distribution can cause:
  - **Vanishing gradients** (if magnitudes shrink too much).  
  - **Exploding gradients** (if values inflate too much).  
  - Slower training (optimizers must repeatedly adjust for changing scales).

**Batch Norm** addresses these issues by **dynamically normalizing** layer inputs during training, effectively reducing “internal covariate shift.”

---

## 2. Batch Normalization Mechanism

### 2.1 Learned Scaling and Shifting

In **standard** data normalization (like z-scoring), we compute:
\[
x' = \frac{x - \mu}{\sigma}
\]
where \(\mu\) and \(\sigma\) come from the feature’s mean and standard deviation.

In **batch normalization**, we replace the simple z-score with **learnable** parameters \(\gamma\) and \(\beta\):

\[
\hat{x} = \frac{x - \mu_\text{batch}}{\sigma_\text{batch}}, 
\quad
x^\prime = \gamma \hat{x} + \beta
\]

- \(\mu_\text{batch}\) and \(\sigma_\text{batch}\) are the **mean** and **std** of the **current mini-batch** for the layer’s activations.
- \(\gamma\) (gamma) and \(\beta\) (beta) are **trainable parameters**:
  - \(\beta\) shifts the mean (like a learned “offset”).
  - \(\gamma\) scales the variance (like a learned “gain”).

Thus, each layer can **adapt** the normalized activations to the scale that best suits the network’s needs.

### 2.2 Forward Pass Modification

Consider a typical layer output:
\[
\mathbf{z} = W \mathbf{x} + b
\]
with some activation \(\sigma(\mathbf{z})\).

#### With Batch Norm:

1. Compute mini-batch mean & std for \(\mathbf{z}\).  
2. Normalize \(\mathbf{z}\) using \(\mu_\text{batch}\) and \(\sigma_\text{batch}\).  
3. Scale and shift with \(\gamma\) and \(\beta\).  
4. Apply activation function \(\sigma\).

\[
\mathbf{z}_\text{norm} = \frac{\mathbf{z} - \mu_\text{batch}}{\sigma_\text{batch}}
\quad\rightarrow\quad
\mathbf{z}^\prime = \gamma \mathbf{z}_\text{norm} + \beta
\quad\rightarrow\quad
\mathbf{y} = \sigma(\mathbf{z}^\prime).
\]

### 2.3 Training vs. Inference Mode

- During **training**: 
  - Each mini-batch has its own \(\mu_\text{batch}\) and \(\sigma_\text{batch}\).  
  - \(\gamma\) and \(\beta\) are updated via **backprop**.  
- During **testing/validation**:
  - We typically use a **running average** of \(\mu_\text{batch}\) and \(\sigma_\text{batch}\) (accumulated during training) so that inference does not depend on the batch.  
  - \(\gamma\) and \(\beta\) are **fixed** at their trained values.  
  - In PyTorch, this is managed automatically by `model.eval()`, which changes BN layers to “evaluation” mode.

---

## 3. PyTorch Implementation

PyTorch offers `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`, etc., depending on dimensionality of data:
- **For MLPs** (fully-connected layers), we often use `BatchNorm1d`.
- **For CNNs** (2D images), we use `BatchNorm2d`.

### 3.1 Example: MLP with BatchNorm1d

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BNNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super(BNNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # BN after fc1
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # BN after fc2
        
        self.out = nn.Linear(hidden_dim, 1)    # final layer for binary classification
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)      # apply BN
        x = self.relu(x)     # activation
        
        x = self.fc2(x)
        x = self.bn2(x)      # BN again
        x = self.relu(x)
        
        x = self.out(x)
        return x

# Example training loop snippet
model = BNNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()            # BN is active
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits.squeeze(), yb.float())
        loss.backward()
        optimizer.step()
    
    model.eval()             # BN uses running statistics
    # Evaluate on test data...
```

- `nn.BatchNorm1d(hidden_dim)` learns a separate \(\gamma\) and \(\beta\) for each neuron in the hidden layer (i.e., dimension = `hidden_dim`).

### 3.2 Placement of Batch Norm

Common practice:
1. **Linear** → **BatchNorm** → **ReLU**  
2. or **Linear** → **ReLU** → **BatchNorm**  

In practice:
- Many frameworks/libraries suggest **BN before activation**.  
- Results can vary by dataset and architecture. **Experimentation** is key.

---

## 4. Empirical Observations

### 4.1 Faster Training, Stable Gradients

- BN reduces **internal covariate shift**:
  - Minimizes risk of **vanishing** or **exploding** gradients.  
  - Usually leads to **faster** or **more stable** convergence.

### 4.2 Form of Regularization

- BN’s “noise” (due to using batch statistics) can **regularize** the network.  
- Larger batch sizes produce more stable estimates of mean/std.  
- Smaller batches produce more variation in these estimates, sometimes providing a mild regularization effect.

### 4.3 Potential Edge Cases

- Very small batch sizes (e.g., <8) can lead to **unstable** BN statistics.  
- For **shallow** networks, BN still helps but the gains might be smaller.  
- Advanced norms: **Layer Normalization**, **Group Normalization**, etc., can handle certain corner cases better.

---

## 5. Example Visualization

Imagine training an MLP on a moderate dataset:

- **Loss Comparison**: 
  - BN: Lower, faster drop.
  - No BN: Higher or slower to converge.
- **Training Accuracy**:
  - BN: Reaches higher accuracy quickly.  
  - No BN: May lag behind or show more instabilities.
- **Test Accuracy**:
  - BN: Gains can be model/data-dependent. Sometimes clearly improved, sometimes not drastically different.

Example hypothetical curves:

```
Loss (train) w/ BN         |   Loss (train) w/o BN
   \         |      \      |      \         |
    \        |       \     |       \        |
     \       |        \    |        \       |
      \______|_________\___|_________\______ 

Accuracy (train) w/ BN     |  Accuracy (train) w/o BN
         /\                |       /\
        /  \               |      /  \
 BN => /    \              |     /    \
```

*(Exact results vary, but BN often helps with speed and stability.)*

---

## 6. Discussion & Key Points

1. **It’s a Learned Normalization**:
   - \(\gamma\) and \(\beta\) are part of the model’s parameters, updated by **backprop**.
2. **BN != “Data Normalization”**:
   - They share conceptual similarities, but BN is applied to **layer activations** inside the network, across each mini-batch, and is **learned**.
3. **Deeper Networks Benefit More**:
   - BN is especially potent in large CNNs (e.g., image classification) or deep MLPs.
4. **Enable & Disable**:
   - BN layers do one thing in **training mode** (use mini-batch stats), and another in **eval mode** (use running averages).
5. **Alternative Terms**:
   - Some argue “activation normalization” or “layer normalization” might be more precise.  
   - Standard term in the field remains: **Batch Normalization**.

---

## 7. Practical Tips

- **Always** use `model.train()` during training loops to ensure BN tracks batch statistics.
- **Switch** to `model.eval()` before running validation/testing to ensure BN uses running (global) means and variances.
- **Tune** the learning rate, batch size, and BN usage. Certain combinations can yield big improvements.

---

## 8. References & Further Reading

1. **Original BN Paper**: Sergey Ioffe, Christian Szegedy. 2015. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.  
2. **PyTorch Docs**: [BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)  
3. **Other Normalizations**:
   - *LayerNorm*, *GroupNorm*, *InstanceNorm*, etc.  

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: Batch Normalization”  
```