aliases: [Scrambled MNIST, Spatial Information, Feedforward, FFN]
tags: [Deep Learning, MNIST, CNN Introduction, Code, Visualization]

## Overview
This lecture demonstrates that standard feedforward networks (FFNs) **do not inherently use spatial arrangement** in image data. By **scrambling** the pixels in MNIST images (applying the *same* random permutation of pixels to every image), we see that a feedforward network can still learn the classification task with high accuracy. This is surprising at first glance, but it highlights that FFNs treat input images as **unstructured** 1D vectors—spatial relationships are not preserved.

> **Motivation**: This sets the stage for **Convolutional Neural Networks (CNNs)**, which *do* leverage spatial locality.

---

## 1. Scrambling the MNIST Dataset

### 1.1 Conceptual Idea
1. **Original MNIST**: Each \(28 \times 28\) image has a natural 2D spatial layout.  
2. **Scrambled MNIST**: 
   - Permute the 784 pixels into a random order, but apply *one* fixed permutation across **all** images.  
   - The result is a “scrambled” version of the digit, unrecognizable to human eyes, yet consistently permuted for all samples.

### 1.2 Code for Scrambling

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1) Load partial MNIST (~20k images) from CSV
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# Separate labels and pixel data
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

# Normalize pixel intensities to [0,1]
pixels_np /= 255.0

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# 2) Create a single random permutation for columns
num_pixels = X_train.shape[1]  # should be 784
perm_indices = np.random.permutation(num_pixels)

# Apply the same permutation to training and test sets
X_train_scrambled = X_train[:, perm_indices]
X_test_scrambled  = X_test[:, perm_indices]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_scrambled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test_scrambled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# Create Datasets and DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
```

> We only **permute once** (line `perm_indices = np.random.permutation(num_pixels)`), then apply that same permutation to every image.

---

## 2. Visualizing Scrambled Images

To see how these scrambled digits look:

```python
def show_scrambled_examples(X_data, y_data, num=6):
    indices = np.random.choice(len(X_data), num, replace=False)
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(indices):
        img = X_data[idx].reshape(28, 28)
        label = y_data[idx]
        plt.subplot(1, num, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()

# Show some scrambled samples from the training set
show_scrambled_examples(X_train_scrambled, y_train, 6)
```

**Observation**:  
- Looks like random noise to humans.  
- In principle, images of the same digit now appear similarly “random,” but we can’t easily see it.

---

## 3. Feedforward Model & Training

We use the *same* feedforward network as before (e.g., two hidden layers), no changes necessary:

```python
import torch.nn as nn
import torch.optim as optim

class ScrambledMNIST_FFN(nn.Module):
    def __init__(self):
        super(ScrambledMNIST_FFN, self).__init__()
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

# Instantiate model, define optimizer & loss
model = ScrambledMNIST_FFN()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 3.1 Training Loop

```python
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
        
        # Evaluate training accuracy
        model.eval()
        correct_train, total_train = 0, 0
        with torch.no_grad():
            for Xb_t, yb_t in train_loader:
                y_pred_train = model(Xb_t)
                _, preds_train = torch.max(y_pred_train, 1)
                correct_train += (preds_train == yb_t).sum().item()
                total_train   += len(yb_t)
        train_acc = correct_train / total_train
        
        # Evaluate test accuracy
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for Xb_te, yb_te in test_loader:
                y_pred_test = model(Xb_te)
                _, preds_test = torch.max(y_pred_test, 1)
                correct_test += (preds_test == yb_te).sum().item()
                total_test   += len(yb_te)
        test_acc = correct_test / total_test
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    return train_accs, test_accs, losses

# Train the model
epochs = 60
train_accs, test_accs, losses = train_model(model, train_loader, test_loader, epochs=epochs)
```

---

## 4. Results & Analysis

### 4.1 Final Performance
Even with **scrambled pixels**, feedforward networks can reach **\~95–96% test accuracy** on MNIST:

```python
print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
print(f"Final Test Accuracy:  {test_accs[-1]*100:.2f}%")
```

> Often this matches or even *slightly exceeds* the same network on **unscrambled** MNIST, revealing that **FFNs do not leverage local spatial structure**.

### 4.2 Loss & Accuracy Curves

```python
import matplotlib.pyplot as plt

epochs_range = range(1, epochs+1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, losses, label='Loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label='Train Acc')
plt.plot(epochs_range, test_accs,  label='Test Acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 5. Why Does This Happen?

1. **FFN Sees a 1D Vector**  
   - By design, we flatten images into a vector of length 784 before feeding into a feedforward layer.  
   - The network has **no awareness** that adjacent pixels in the vector were adjacent in the original 2D image.  

2. **Scrambled = Same Mapping for All Images**  
   - As long as each digit’s pixels are **permuted consistently**, the network can still learn patterns (co-occurrences of pixel intensities) that correspond to each label.  
   - Spatial locality is **not** exploited, but it’s also **not** essential for the feedforward network’s classification capability on MNIST.

3. **But** …  
   - This approach would not scale well to more complex images where **local patterns** matter significantly (e.g., color edges, textures, shapes).  
   - For complex real-world images, **CNNs** drastically outperform FFNs because they **do** leverage spatial structure and fewer parameters.

---

## 6. Implications for CNNs

- **Convolutional Neural Networks** introduce **convolutional filters** that respect local spatial proximity, drastically reducing parameters and focusing on **spatially correlated** features.  
- **Scrambled data** breaks that local structure advantage. CNNs would not handle scrambled images the same way as normal images.  
- Thus, **CNN** excels at tasks where **spatial arrangement** is crucial, while a plain FFN is effectively “blinded” to 2D arrangement but can still learn from the raw pixel intensities in simpler tasks like MNIST.

---

## 7. Key Takeaways

1. **Spatially Blind**: FFNs do not encode any 2D structure. The pixels are seen purely as a 784-length vector.  
2. **High Accuracy Despite Scrambling**: MNIST is “easy” enough that a feedforward net can still learn the classification from random permutations of pixels.  
3. **CNN Motivation**: In real-world image tasks, local spatial features are crucial. CNNs **exploit** these, generally outperforming FFNs while using fewer parameters.  
4. **Looking Ahead**: This result foreshadows how CNNs are designed to **leverage spatial correlations** and is a stepping stone to understanding convolutional layers.

---

## 8. Further Exploration

- **Vary Permutations**: Try multiple permutations and compare test accuracy.  
- **Partial Scrambling**: Randomly shuffle only half the pixels to observe intermediate performance.  
- **Larger Datasets**: Check if scrambling severely impacts tasks like CIFAR-10 or ImageNet (it should!).  
- **CNN Comparison**: Train a CNN on scrambled vs. unscrambled MNIST to see the difference in performance.

---

**End of Notes – "Scrambled MNIST"**  
```