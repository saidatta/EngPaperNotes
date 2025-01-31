
aliases: [PyTorch Loss Functions, BCE vs. BCEWithLogits, Custom Loss]
tags: [Deep Learning, Lecture Notes, PyTorch, Loss Functions]

In this note, we **explore** how PyTorch implements **common loss functions** (MSE, BCE, Cross Entropy, etc.) and show **practical examples**. Finally, we see how to write a **custom** loss if the built-in options don’t suffice.

---
## 1. Setup

We only need **PyTorch** and **Matplotlib** for plotting results. We’re **not** building a full model or using datasets—just illustrating how **loss** functions behave on sample inputs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
```

---

## 2. Mean Squared Error (MSE) Demo

### 2.1 Basic Usage

```python
# 1) Create the loss function object
mse_loss_fun = nn.MSELoss()

# 2) Suppose our target is a single value y=0.5
y_true = torch.tensor([0.5])

# Let's sample model predictions from -2 to 2
y_hats = torch.linspace(-2, 2, steps=101)

# Compute MSE for each predicted value
mse_values = []
for y_pred in y_hats:
    loss = mse_loss_fun(y_pred, y_true)
    mse_values.append(loss.item())

# Visualize
plt.figure(figsize=(6,4))
plt.plot(y_hats.numpy(), mse_values, label="MSE Loss")
plt.axvline(0.5, color='gray', linestyle='--', label="Target = 0.5")
plt.title("MSE Loss vs. Prediction")
plt.xlabel("y_hat")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

#### Observations

- **Parabola** shape, minimized at `y_hat == y_true == 0.5`.  
- Loss grows **quadratically** as the model prediction moves away from 0.5.

### 2.2 Why MSE?

- Used for **regression** tasks with continuous targets.  
- In PyTorch, `nn.MSELoss()` expects predicted and target tensors of the same shape.

---

## 3. Binary Cross-Entropy (BCE)

### 3.1 BCE Loss Manually

```python
bce_loss_fun = nn.BCELoss()

# Our 'model outputs' should be in (0,1), simulating a Sigmoid
y_hats = torch.linspace(1e-5, 1-1e-5, steps=100)

# We consider two possible targets: 0 or 1
y0 = torch.tensor([0.0])
y1 = torch.tensor([1.0])

bce_vals_y0 = []
bce_vals_y1 = []

for y_pred in y_hats:
    # Must keep shape consistent with bce_loss_fun
    y_pred = y_pred.view(1)
    loss0 = bce_loss_fun(y_pred, y0)
    loss1 = bce_loss_fun(y_pred, y1)
    bce_vals_y0.append(loss0.item())
    bce_vals_y1.append(loss1.item())

plt.figure(figsize=(8,4))
plt.plot(y_hats.numpy(), bce_vals_y0, label="Target=0", color='red')
plt.plot(y_hats.numpy(), bce_vals_y1, label="Target=1", color='blue')
plt.title("BCE Loss vs. y_hat for y=0 or y=1")
plt.xlabel("y_hat (Sigmoid Output)")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

**Observations**:

- **Loss is minimal** if `y_hat ~ y`.  
- If `y=0`, pushing `y_hat` to 1 causes **high** loss, and vice versa.

### 3.2 Why Need BCEWithLogitsLoss?

- **BCE** expects inputs \(\in (0,1)\).  
- Many models output **logits** (raw scores, \((-\infty,\infty)\)).  
- **`nn.BCEWithLogitsLoss()`** = Sigmoid + BCE in one, more **numerically stable**.

```python
# Example: raw output = 2.0 (logit)
bce_logits_loss_fun = nn.BCEWithLogitsLoss()

y_pred_logit = torch.tensor([2.0])  # raw logistic output
y_target = torch.tensor([1.0])

loss = bce_logits_loss_fun(y_pred_logit, y_target)
print("BCEWithLogitsLoss when logit=2, target=1:", loss.item())
```

Internally, PyTorch does `sigmoid(2.0) ~ 0.8808`, then BCE.

---

## 4. Categorical Cross-Entropy (Multi-Class)

### 4.1 CrossEntropyLoss

```python
ce_loss_fun = nn.CrossEntropyLoss()

# Suppose model output for 3 classes (e.g., cat/dog/giraffe)
# shape [1,3]
logits = torch.tensor([[2.0, 4.0, 3.0]])

# 'target' is the index of the correct class, e.g. 0,1,2
# shape []
target_class_0 = torch.tensor([0])
target_class_1 = torch.tensor([1])
target_class_2 = torch.tensor([2])

loss_0 = ce_loss_fun(logits, target_class_0)  # correct class is 0
loss_1 = ce_loss_fun(logits, target_class_1)  # correct class is 1
loss_2 = ce_loss_fun(logits, target_class_2)  # correct class is 2

print("CE Loss if true class=0:", loss_0.item())
print("CE Loss if true class=1:", loss_1.item())
print("CE Loss if true class=2:", loss_2.item())
```

**Note**: `CrossEntropyLoss` in PyTorch expects **raw logits**. It internally does `LogSoftmax + NLLLoss`.

#### Double Softmax? Don’t do it.

If you manually apply `F.softmax(logits, dim=1)` then pass to `CrossEntropyLoss`, you’d be **softmaxing twice**, which leads to incorrect results. Just feed the **raw** logits into `CrossEntropyLoss`.

---

## 5. Softmax vs. LogSoftmax

### 5.1 Example

```python
logits = torch.tensor([[2.0, 4.0, 3.0]])

# 1) Raw logits
print("Raw logits:", logits)

# 2) Softmax probabilities
softmax_output = F.softmax(logits, dim=1)
print("Softmax:", softmax_output)

# 3) LogSoftmax
logsoftmax_output = F.log_softmax(logits, dim=1)
print("LogSoftmax:", logsoftmax_output)
```

- **Softmax** → each entry in \((0,1)\), sums to 1 across the dimension.  
- **LogSoftmax** → sums to 0 in *log-space*, typically negative values.

### 5.2 Why LogSoftmax?

- More **numerically stable** when dealing with very small probabilities.  
- Stronger gradient signals for “confident but wrong” predictions.

---

## 6. Creating a Custom Loss Function

While PyTorch has many **built-in** losses, occasionally you need a **custom** approach. You can define your own by **inheriting** from `nn.Module` and overriding `forward()`.

```python
class MyCustomL1Loss(nn.Module):
    def __init__(self):
        super(MyCustomL1Loss, self).__init__()
    
    def forward(self, pred, target):
        # Simple absolute difference
        return torch.mean(torch.abs(pred - target))

# Usage:
l1_loss = MyCustomL1Loss()

pred_val = torch.tensor([2.0])
true_val = torch.tensor([3.2])

loss_val = l1_loss(pred_val, true_val)
print("Custom L1 Loss:", loss_val.item())
```

**In practice**:
- Usually rely on existing `nn.MSELoss`, `nn.L1Loss`, `nn.BCEWithLogitsLoss`, etc.  
- Custom is needed for specialized tasks (e.g., perceptual losses in GANs, advanced distribution-based losses, etc.).

---

## 7. Summary & Best Practices

1. **MSELoss**: For **regression**.  
2. **BCEWithLogitsLoss**: For **binary** classification with **logits**.  
3. **CrossEntropyLoss**: For **multi-class** classification with **logits** (PyTorch does LogSoftmax inside).  
4. **LogSoftmax** vs **Softmax**: Typically prefer **LogSoftmax** + NLL for **numerical stability**.  
5. **Custom Loss**: Inherit `nn.Module`, implement `forward()`. Do so only if you have a genuine need beyond standard losses.

---

## 8. References

- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)  
- **Goodfellow et al. (Deep Learning Book)** – Chapter on **Loss Functions**  
- [BCEWithLogitsLoss vs BCE + Sigmoid in PyTorch discussion](https://discuss.pytorch.org/t/bcewithlogits-vs-bce-sigmoid/)

**Created by**: [Your Name / Lab / Date]  
**Lecture Reference**: “Meta parameters: Loss functions in PyTorch”  
```