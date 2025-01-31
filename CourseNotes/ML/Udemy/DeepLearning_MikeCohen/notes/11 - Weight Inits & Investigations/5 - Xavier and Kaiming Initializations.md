## 1. Overview

This lecture demonstrates how to **implement** the widely used **Xavier** (Glorot) and **Kaiming** (He) initialization schemes in **PyTorch**. We also examine **default** PyTorch initialization behavior (which is essentially Kaiming for `nn.Linear`) and how to manually adjust weights to **Xavier**. By the end, you’ll understand:

1. How PyTorch’s **default** `nn.Linear` initialization works.  
2. How to access and **manipulate** model parameters for custom initialization.  
3. How to **verify** that the distribution of weights matches theoretical formulas (e.g., range for uniform, variance for normal).

---

## 2. Setup and Model Definition

### 2.1. Model Architecture

We define a multi-layer architecture with **100** units per layer (except final output has 2 units). The reason is to produce a *large set* of weights and biases to analyze:

```python
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Large hidden layers (100 -> 100) for more interesting weight distributions
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)  # small output: 2 classes
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net = BigNet()
print(net)
```
```

**Note**: We aren’t using any dataset here. We just want to investigate the **initialized weights**.

---

## 3. Inspecting Default PyTorch (Kaiming) Initialization

### 3.1. Extract and Plot Histograms

Below, we gather **all** weights and **all** biases across layers into separate arrays, then plot histograms:

```python
```python
all_weights = np.array([])
all_biases  = np.array([])

for name, param in net.named_parameters():
    pdata = param.detach().flatten().cpu().numpy()
    if 'bias' in name:
        all_biases = np.concatenate((all_biases, pdata))
    else:  # weight
        all_weights = np.concatenate((all_weights, pdata))

print(f"Total biases: {len(all_biases)}, Total weights: {len(all_weights)}")

plt.figure(figsize=(8,4))
# We convert counts to probability density to compare different scales easily
weights_hist, w_bins = np.histogram(all_weights, bins=100, density=True)
biases_hist, b_bins  = np.histogram(all_biases,  bins=100, density=True)

w_bincenters = 0.5*(w_bins[:-1] + w_bins[1:])
b_bincenters = 0.5*(b_bins[:-1] + b_bins[1:])

plt.plot(w_bincenters, weights_hist, label='Weights')
plt.plot(b_bincenters, biases_hist,  label='Biases')
plt.xlabel('Parameter Value')
plt.ylabel('Probability Density')
plt.title('Default Initialization Distributions')
plt.legend()
plt.show()
```
```

**Observations**:
- PyTorch’s **default** `nn.Linear` typically uses a **uniform** distribution in \(\left[-\sqrt{\frac{1}{n_\text{in}}}, +\sqrt{\frac{1}{n_\text{in}}}\right]\).  
- If each hidden layer is 100 → 100, \(\sqrt{\frac{1}{100}} = 0.1\).  
- So we expect uniform \([-0.1, 0.1]\) for **weights**.  
- Bias distribution might look less “flat,” especially if fewer bias parameters exist (leading to sampling variance).

### 3.2. Layer-by-Layer Comparison

We can confirm that each **hidden** layer (100→100) shows roughly the same distribution, while the final layer (100→2) has a different range or “fan in.” We also note that if a layer is **square** (e.g., 100×100), the default range is \(\sqrt{1/100}\).

```python
```python
plt.figure(figsize=(10,6))

colors = plt.cm.viridis(np.linspace(0,1,6))  # color map

i = 0
for name, param in net.named_parameters():
    pdata = param.detach().flatten().cpu().numpy()
    hist, bins = np.histogram(pdata, bins=50, density=True)
    bin_cent = 0.5*(bins[:-1] + bins[1:])
    plt.plot(bin_cent, hist, label=name, color=colors[i], alpha=0.7)
    i += 1

plt.xlabel("Parameter Value")
plt.ylabel("Density")
plt.title("Per-Layer Parameter Distributions")
plt.legend()
plt.show()
```
```

For the **final layer** with output size 2, the range might differ from \(\pm0.1\) because `n_in=100`, `n_out=2`. The default formula is:  
\[
\text{Uniform}\left(-\sqrt{\frac{1}{n_\text{in}}}, +\sqrt{\frac{1}{n_\text{in}}}\right).
\]

---

## 4. Verifying PyTorch Documentation

To see the exact formula, we can run something like `?nn.Linear` or check the [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). It states:

> The layer’s weight is randomly initialized from
> \[
> \text{Uniform}\Bigl(-\sqrt{\frac{1}{\text{fan\_in}}}, \sqrt{\frac{1}{\text{fan\_in}}}\Bigr)
> \]

Hence, we confirm that by default, **Kaiming**-style uniform is used.

---

## 5. Modifying Weights to **Xavier** Initialization

### 5.1. Xavier Normal Example

Xavier (Glorot) initialization for a **normal** distribution:  
\[
w \sim \mathcal{N}\Bigl(0,\;\sigma^2 = \frac{2}{n_\text{in} + n_\text{out}}\Bigr).
\]

In PyTorch, we can manually **overwrite** the weights for each layer using `nn.init.xavier_normal_`:

```python
```python
import torch.nn.init as init

# Create a fresh network
net_xavier = BigNet()

# We'll override ONLY the weights with Xavier normal, keep biases as default
with torch.no_grad():
    for name, param in net_xavier.named_parameters():
        if 'weight' in name:
            init.xavier_normal_(param)  # in-place operation
        # bias remains at default (Kaiming uniform)

# Inspect the resulting distribution
all_w_xav = []
all_b_xav = []
for name, param in net_xavier.named_parameters():
    vals = param.detach().flatten().cpu().numpy()
    if 'weight' in name:
        all_w_xav.extend(vals)
    else:
        all_b_xav.extend(vals)

plt.figure(figsize=(8,4))
w_hist, w_bins = np.histogram(all_w_xav, bins=100, density=True)
b_hist, b_bins = np.histogram(all_b_xav, bins=100, density=True)

plt.plot(0.5*(w_bins[:-1]+w_bins[1:]), w_hist, label='Xavier Weight (Normal)')
plt.plot(0.5*(b_bins[:-1]+b_bins[1:]), b_hist, label='Default Bias (Uniform)')
plt.title("Xavier Normal for Weights & Default Bias Distribution")
plt.legend()
plt.show()
```
```

We should see **weights** roughly distributed like a normal with mean=0 and some variance ~\(\frac{2}{n_\text{in}+n_\text{out}}\). The **biases** remain uniform in \([-0.1,0.1]\) (since `fan_in=100` for the hidden layers).

#### 5.1.1. Checking Theoretical Variance

For a layer with `in=100, out=100`, Xavier normal yields:
\[
\sigma^2 = \frac{2}{100 + 100} = \frac{2}{200} = 0.01.
\]
Hence \(\sigma = 0.1\).

We can compare to the **empirical** variance:

```python
```python
# e.g., pick first layer weights
first_layer_w = net_xavier.fc1.weight.detach().flatten().cpu().numpy()
empirical_var = first_layer_w.var()
print("Theoretical var=0.01, Empirical var=", empirical_var)
```
```

They should match **closely**, acknowledging random sampling differences.

---

### 5.2. Other PyTorch Init Methods

In addition to `kaiming_uniform_` (default) and `xavier_normal_`, PyTorch offers:

- `xavier_uniform_`
- `kaiming_normal_`
- `orthogonal_`
- `sparse_`
- etc.

Each addresses different model needs. You can **mix** them as you see fit, or implement a fully **custom** method:

```python
```python
with torch.no_grad():
    net_xavier.fc2.weight.data.uniform_(-0.05, 0.05)
```
```
(This overrides the `fc2.weight` with a uniform distribution of your choice.)

---

## 6. Practical Notes

1. **Default is Kaiming** (fan_in version) for `nn.Linear` in PyTorch. Often good enough for many tasks.  
2. **Xavier** recommended for older architectures or those using **tanh/sigmoid**.  
3. **Deep architectures**: careful initialization helps mitigate vanishing/exploding gradients.  
4. **Bias** initialization is often less critical; typically small uniform or zero.

---

## 7. Summary

- **Kaiming** (He) initialization is **PyTorch’s default** for `nn.Linear`.  
- **Xavier** initialization can be set **manually** using `nn.init.xavier_normal_` or `nn.init.xavier_uniform_`.  
- Weight initializations matter more for **deep** or **specialized** architectures.  
- If you change **layer sizes**, the **range/variance** of your initial distributions changes. You can confirm it by examining the histogram of parameters.  

---

## 8. Further Reading

1. **He et al. (2015)**: *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*  
2. **Glorot & Bengio (2010)**: *Understanding the difficulty of training deep feedforward neural networks*  
3. **PyTorch nn.init** Docs: [https://pytorch.org/docs/stable/nn.init.html](https://pytorch.org/docs/stable/nn.init.html)  
4. **Deep Learning Book** (Goodfellow et al.), Chapter on *Numerical Computation* and *Initialization*.  

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** (e.g., `XavierKaimingInits.md`) and paste the entire markdown above (including the frontmatter).  
2. Add any **internal links** or references (e.g., `[[Weight Init Theory]]`, `[[PyTorch Custom Initialization]]`).  
3. Adjust headings or tags to fit your vault structure.

These notes demonstrate how to **access and modify** PyTorch parameters for **Xavier** or **Kaiming** initialization, confirm distributions, and compare with **default** PyTorch initialization.