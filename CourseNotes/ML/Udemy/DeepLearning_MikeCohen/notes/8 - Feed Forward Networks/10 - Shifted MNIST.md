aliases: [Shifted MNIST, Image Translation, Feedforward Limitations]
tags: [Deep Learning, MNIST, Data Augmentation, Visualization, PyTorch]
## Overview
This lecture investigates how **horizontal shifts** (translations) in MNIST images affect feedforward networks’ (FFNs) performance. We’ll see that even small random shifts in **test images** can significantly degrade accuracy—even though a **full pixel scramble** did **not** hurt FFN performance. This highlights a key limitation of feedforward networks for image tasks and motivates the need for **Convolutional Neural Networks (CNNs)**, which handle spatial shifts more gracefully.

---
## 1. Shifting (Rolling) Images Concept

1. **Rolling (Shifting) Horizontally**:  
   - In many libraries (including PyTorch), shifting an image’s columns is called *rolling*.  
   - For an \(n\)-pixel shift, columns are moved right/left by \(n\) columns, with edges **wrapping around**.
2. **Effect on Feedforward Nets**:  
   - FFNs flatten images into **1D vectors** and thus “see” each pixel as an independent feature.  
   - If **train** and **test** distributions differ (e.g., test images are shifted), performance can plummet.

---

## 2. Data Loading & Preparation

We assume a partial MNIST dataset (e.g., `mnist_train_small.csv` with ~20k samples) already available.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1) Load partial MNIST data
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# Separate labels and pixel values
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

# Normalize to [0,1]
pixels_np /= 255.0

# Create train/test splits
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# Create Dataset and DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
```

---

## 3. Brief Intro to Dataloaders & Accessing Their Data

### 3.1 Inspect a DataLoader Object

```python
print(train_loader)
# Something like: <torch.utils.data.dataloader.DataLoader object at 0x7f96d8f698d0>

# We can access the underlying dataset
print(train_loader.dataset)
# <torch.utils.data.dataset.TensorDataset object at 0x7f96d8f69d30>

# The dataset stores .tensors (images, labels)
print(train_loader.dataset.tensors)
# (tensor([...]), tensor([...]))  # (images, labels)
```

- The first element of the tuple is **all image data** (\(N \times 784\)).  
- The second element is the **corresponding labels** (\(N \times 1\)).

---

## 4. Rolling (Shifting) Test Images

We keep **train images** *as-is* and shift only **test images**. Each test image is shifted horizontally by a random integer \(\delta\) in \([-10, +10]\).

### 4.1 Rolling in PyTorch

```python
# Example: Rolling a single 28x28 image by 5 pixels to the right
test_img_1d = test_loader.dataset.tensors[0][0]  # first image in test set, shape (784,)
img_2d = test_img_1d.reshape(28,28)

# PyTorch calls shifting "rolling"
shifted_img_2d = torch.roll(img_2d, shifts=5, dims=1)  # shift 5 columns to the right
```

1. `shifts=5`: number of pixels to roll.  
2. `dims=1`: axis along which we shift (1 = columns, 0 = rows).

### 4.2 Applying Random Shifts to **All** Test Images

```python
# Loop through entire test set, shifting each image
for i in range(len(test_loader.dataset.tensors[0])):
    # original image (1D)
    original_img_1d = test_loader.dataset.tensors[0][i]
    
    # random shift ∈ [-10,10]
    shift_amount = np.random.randint(-10, 11)
    
    # reshape to 28x28
    img_2d = original_img_1d.reshape(28,28)
    
    # shift horizontally
    shifted_2d = torch.roll(img_2d, shifts=shift_amount, dims=1)
    
    # flatten back to 784
    shifted_1d = shifted_2d.flatten()
    
    # replace in dataset
    test_loader.dataset.tensors[0][i] = shifted_1d
```

> Now every test sample is shifted by some integer from -10 to +10 columns.

---

## 5. Visualizing Shifted Samples

```python
# Let's peek at a single test sample before/after shift
index = 0  # just an example index
original_img = X_test_t[index].reshape(28,28)
shifted_img  = test_loader.dataset.tensors[0][index].reshape(28,28)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(original_img, cmap='gray')
plt.title(f"Original digit: {y_test[index]}")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(shifted_img, cmap='gray')
plt.title(f"Shifted digit: {y_test[index]}")
plt.axis('off')
plt.show()
```

---

## 6. Defining & Training the Model

Use a **standard feedforward** network (e.g., two hidden layers). We train on **non-shifted** train images while testing on **shifted** test images.

```python
class ShiftedMNIST_FFN(nn.Module):
    def __init__(self):
        super(ShiftedMNIST_FFN, self).__init__()
        self.fc0 = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

model = ShiftedMNIST_FFN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
def train_model(model, train_loader, test_loader, epochs=60):
    train_accs, test_accs, losses = [], [], []
    
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred_log = model(Xb)
            loss = criterion(y_pred_log, yb)
            loss.backward()
            optimizer.step()
        
        # Evaluate on train data
        model.eval()
        correct_train, total_train = 0, 0
        with torch.no_grad():
            for Xb_t, yb_t in train_loader:
                y_train_pred = model(Xb_t)
                _, train_preds = torch.max(y_train_pred, 1)
                correct_train += (train_preds == yb_t).sum().item()
                total_train   += len(yb_t)
        train_acc = correct_train / total_train
        
        # Evaluate on shifted test data
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for Xb_te, yb_te in test_loader:
                y_test_pred = model(Xb_te)
                _, test_preds = torch.max(y_test_pred, 1)
                correct_test += (test_preds == yb_te).sum().item()
                total_test   += len(yb_te)
        test_acc = correct_test / total_test
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    return train_accs, test_accs, losses

train_accs, test_accs, losses = train_model(model, train_loader, test_loader, epochs=60)
```

---

## 7. Results & Observations

### 7.1 Accuracy Drops Significantly on Shifted Test Data
- Typical test accuracies might drop to **30–40%**, a large deviation from the ~95% we get on the **unshifted** dataset.
- Training accuracy remains high (~98–99%) because the training set is **unshifted**.

```python
print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
print(f"Final Test Accuracy:  {test_accs[-1]*100:.2f}%")
```

### 7.2 Visualizing Curves

```python
epochs_range = range(1, 61)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, test_accs,  label="Test Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

> Typically:
> - **Train Accuracy** \(\to\) ~**99%**.  
> - **Test Accuracy** \(\to\) ~**30–40%** (depending on how large the shift range is).

---

## 8. Why Does Shifting Hurt So Much?

1. **Train vs. Test Distribution Mismatch**  
   - The model learns to recognize digits in very specific pixel positions.  
   - At inference, if the digit is shifted horizontally, the **pixels** shift to different indices in the 1D input vector, violating the learned associations.

2. **No Spatial Invariance**  
   - A feedforward net doesn’t “see” local context—only raw pixel intensities in certain positions.  
   - Even small translations cause large changes in the input vector layout.

3. **Contrast with Scrambling**  
   - When we scrambled **all** images consistently (train + test), there was no mismatch.  
   - Here, train is unshifted, test is shifted → big mismatch in distributions.

---

## 9. Foreshadowing CNNs

- **Convolutional Neural Networks** introduce **translation invariance** (or at least **translation tolerance**) by learning **local features** that shift across the image.  
- Shifting an image in CNN input often has far **less impact** on performance, because convolutional filters “slide” over the image to detect patterns anywhere in the spatial plane.

---

## 10. Key Takeaways

1. **Feedforward Limitation**: A plain FFN cannot handle shifts unless it has explicitly seen those shifts during training (data augmentation).
2. **Severe Accuracy Drop**: Even small shifts (±10 pixels) can reduce test accuracy from ~95% to ~30–40%.  
3. **Data Mismatch**: If train images differ in distribution from test images (e.g., shift, rotation, scale), FFNs fail to generalize.  
4. **CNN Motivation**: Convolutional layers drastically improve robustness to shifts, making them the go-to for image-related tasks.

---

## 11. Further Exploration

- **Shifting Rows**: Try rolling vertically (`dims=0`) or both horizontally & vertically.  
- **Data Augmentation**: Add shifting to *training* images, measure if test accuracy improves.  
- **Vary Shift Range**: e.g., ±1 pixel vs. ±10 vs. ±14. Observe how accuracy changes.  
- **Compare With CNN**: Once you learn CNNs, see how they handle shifted MNIST. Expect **much higher** test accuracy.

---

**End of Notes – "Shifted MNIST"**
```