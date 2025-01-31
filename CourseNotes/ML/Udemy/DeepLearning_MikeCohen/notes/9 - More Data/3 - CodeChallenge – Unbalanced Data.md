aliases: [Unbalanced Data, Class Imbalance, Binary Classification, Wine Quality]
tags: [Data, Deep Learning, CodeChallenge, Imbalanced Classes, PyTorch]
## Overview
In this challenge, we explore how **unbalanced classes** affect model training and evaluation. We’ll revisit the **wine quality** dataset but **binarize** the wine ratings with **different thresholds**:
1. **Low threshold**: e.g., wines rated \(\leq 4\) vs. \(\geq 5\)
2. **Medium threshold**: our usual split, \(\leq 5\) vs. \(\geq 6\)
3. **High threshold**: \(\leq 6\) vs. \(\geq 7\)

Depending on the threshold, one class becomes much larger than the other. You will see how the model might **learn a strong bias**—predicting the majority class almost all the time—leading to deceptively high overall accuracy but very poor accuracy on the minority class.

---
## 1. Theoretical Background

### 1.1 What Is Unbalanced Data?
- In classification, **unbalanced** or **imbalanced** data occurs when one class significantly outnumbers the other(s).  
- Example: 99% cats, 1% ships. A trivial model that **always predicts “cat”** can achieve **99%** accuracy.

### 1.2 Why Is This a Problem?
- The model fails to learn meaningful **feature distinctions**.  
- **Overall accuracy** can be misleadingly high.  
- Minority class performance often plummets.

### 1.3 Real-World Examples
- **Medical diagnosis**: Few positive disease cases vs. many negatives.  
- **Fraud detection**: Legitimate transactions far outnumber fraudulent.

---

## 2. Wine Quality Dataset Refresher

We have a dataset of ~1599 red wines, each labeled with a quality score \(\in \{3,4,5,6,7,8\}\). Previously, we **binarized** at **5.5**:

- **Low quality**: \(\{3,4,5\}\)  
- **High quality**: \(\{6,7,8\}\)

This yields a roughly balanced distribution. But in this challenge, we’ll choose **three different thresholds**:

1. **Threshold = 4.5**  
   - “Low” = \(\{3,4\}\)  
   - “High” = \(\{5,6,7,8\}\)  
   - Very few low-quality wines.  
2. **Threshold = 5.5**  
   - “Low” = \(\{3,4,5\}\)  
   - “High” = \(\{6,7,8\}\)  
   - More balanced.  
3. **Threshold = 6.5**  
   - “Low” = \(\{3,4,5,6\}\)  
   - “High” = \(\{7,8\}\)  
   - Few high-quality wines.

---

## 3. Data Import & Preprocessing

Let’s assume you have the wine quality dataset in CSV form (`winequality-red.csv`). We’ll:
1. **Load** the data.
2. **Z-score** normalize features.
3. **Binarize** labels with a flexible threshold \(\mathrm{qualThreshold}\).

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

def create_wine_dataloaders(qualThreshold=5.0, test_size=0.2, batch_size=32):
    # 1) Load CSV
    df = pd.read_csv("winequality-red.csv", sep=';')
    
    # 2) Extract data
    data = df.drop('quality', axis=1).values
    labels = df['quality'].values
    
    # 3) Z-score normalization
    data_mean = np.mean(data, axis=0)
    data_std  = np.std(data, axis=0)
    data      = (data - data_mean) / data_std
    
    # 4) Binarize labels with flexible threshold
    # Label = 1 if quality >= qualThreshold
    boolQuality = (labels >= qualThreshold).astype(int)
    
    # Convert to tensors
    data_t   = torch.tensor(data, dtype=torch.float32)
    labels_t = torch.tensor(boolQuality, dtype=torch.long)
    
    # 5) Create train/test split
    N = len(labels_t)
    idx = np.random.permutation(N)
    test_size_abs = int(N * test_size)
    train_idx, test_idx = idx[test_size_abs:], idx[:test_size_abs]
    
    train_data = TensorDataset(data_t[train_idx], labels_t[train_idx])
    test_data  = TensorDataset(data_t[test_idx],  labels_t[test_idx])
    
    # 6) Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader
```

**Note**: The function `create_wine_dataloaders` returns **train_loader**, **test_loader** with a user-specified threshold (`qualThreshold`).

---

## 4. Model Architecture: Leaky ReLU + Adam

We define a 2-layer network (or any architecture you like), using **Leaky ReLU**. The challenge instructions:

- Activation: **LeakyReLU**
- Optimizer: **Adam** with **lr=0.001**
- Train for **500 epochs**.

```python
class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)  # raw logits
        return x

def train_wine_model(train_loader, test_loader, epochs=500):
    model = WineModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []
    
    for epoch in range(epochs):
        ###################
        # Training
        ###################
        model.train()
        batch_losses, batch_accs = [], []
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            _, preds = torch.max(logits, dim=1)
            batch_accs.append((preds == yb).float().mean().item())
        
        train_losses.append(np.mean(batch_losses))
        train_accs.append(np.mean(batch_accs))
        
        ###################
        # Testing
        ###################
        model.eval()
        batch_losses_test, batch_accs_test = [], []
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                logits_t = model(Xb_t)
                loss_t = criterion(logits_t, yb_t)
                batch_losses_test.append(loss_t.item())
                
                _, preds_t = torch.max(logits_t, dim=1)
                batch_accs_test.append((preds_t == yb_t).float().mean().item())
        
        test_losses.append(np.mean(batch_losses_test))
        test_accs.append(np.mean(batch_accs_test))
    
    return model, train_losses, test_losses, train_accs, test_accs
```

---

## 5. Checking Model & Dataloaders

Before the main experiment, it’s good to do a **quick test**:

```python
# Quick test with threshold=5.5
train_loader, test_loader = create_wine_dataloaders(qualThreshold=5.5)
model, train_losses, test_losses, train_accs, test_accs = train_wine_model(train_loader, test_loader, epochs=10)

print("Final Train Accuracy:", train_accs[-1])
print("Final Test Accuracy: ", test_accs[-1])
```

This should confirm the code runs without error.

---

## 6. The Experiment with Three Thresholds

The code snippet below:
1. Loops over thresholds: `[4.5, 5.5, 6.5]`
2. Creates data, trains the model for 500 epochs.
3. Plots:
   - **Loss** vs. epoch
   - **Accuracy** vs. epoch (train & test)
   - **Accuracy per class** (low vs. high quality) in the test set.

```python
thresholds = [4.5, 5.5, 6.5]

for thresh in thresholds:
    print(f"\n=== Running threshold = {thresh} ===")
    
    # 1) Create data
    train_loader, test_loader = create_wine_dataloaders(qualThreshold=thresh, test_size=0.2, batch_size=32)
    
    # 2) Count how many samples are low/high
    #    We'll do this on the entire dataset (train+test)
    #    Just to illustrate imbalance
    all_data, all_labels = [], []
    for ds in [train_loader.dataset, test_loader.dataset]:
        Xds, yds = ds[:]
        all_data.append(Xds)
        all_labels.append(yds)
    all_data   = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    
    n_low  = (all_labels == 0).sum().item()
    n_high = (all_labels == 1).sum().item()
    print(f"Total samples: {n_low+n_high}, Low= {n_low}, High= {n_high}")
    
    # 3) Train
    model, train_losses, test_losses, train_accs, test_accs = train_wine_model(train_loader, test_loader, epochs=500)
    
    # Evaluate final performance on test set
    # per-class accuracy
    model.eval()
    with torch.no_grad():
        test_acc_low, test_acc_high = [], []
        for Xb_test, yb_test in test_loader:
            logits_test = model(Xb_test)
            _, preds_test = torch.max(logits_test, 1)
            acc = (preds_test == yb_test)
            # separate by class
            acc_low  = acc[yb_test==0].float().mean().item() if (yb_test==0).any() else np.nan
            acc_high = acc[yb_test==1].float().mean().item() if (yb_test==1).any() else np.nan
            
            if not np.isnan(acc_low):  test_acc_low.append(acc_low)
            if not np.isnan(acc_high): test_acc_high.append(acc_high)
    
    # final average
    avg_low  = np.mean(test_acc_low) if len(test_acc_low)   > 0 else 0
    avg_high = np.mean(test_acc_high) if len(test_acc_high)> 0 else 0
    print(f"Avg Test Accuracy: {test_accs[-1]:.2f}, Low= {avg_low:.2f}, High= {avg_high:.2f}")
    
    # 4) Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    fig.suptitle(f"Threshold={thresh}, (Low={n_low}, High={n_high})")
    
    # (a) Loss curves
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(test_losses,  label='Test  Loss')
    axs[0].set_xlabel("Epoch"), axs[0].set_ylabel("Loss"), axs[0].legend()
    
    # (b) Accuracy curves
    axs[1].plot(train_accs, label='Train Acc')
    axs[1].plot(test_accs,  label='Test  Acc')
    axs[1].set_xlabel("Epoch"), axs[1].set_ylabel("Accuracy"), axs[1].legend()
    
    # (c) Per-class test accuracy (bar plot)
    axs[2].bar(["Low", "High"], [avg_low, avg_high], color=['blue','orange'])
    axs[2].set_ylim([0,1])
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Per-Class Test Accuracy")
    axs[2].text(0,  avg_low+0.02,  f"{avg_low:.2f}",  ha='center')
    axs[2].text(1,  avg_high+0.02, f"{avg_high:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()
```

**Note**: The final plots show:
- **Loss** vs. epoch (train & test)
- **Accuracy** vs. epoch (train & test)
- **Bar plot**: final test accuracy for low/high classes.

---

## 7. Typical Results

1. **Threshold=4.5**  
   - Very few “low” wines, many “high” wines.  
   - Model quickly “learns” to predict “high” → overall test accuracy might be ~90%, but **low** class accuracy is near **0%**.
2. **Threshold=5.5**  
   - More balanced split; both classes have a substantial number of samples.  
   - Model typically achieves moderate (~75–80%) accuracy on each class.
3. **Threshold=6.5**  
   - Many “low” wines, fewer “high” wines.  
   - The model might bias toward “low,” yield ~70–80% overall accuracy, but poor performance (often <50%) on the “high” class.

---

## 8. Key Takeaways

1. **Class Imbalance Leads to Bias**  
   - A model can exploit the imbalance by **predicting the majority class** most of the time, achieving deceptively high accuracy.
2. **Look at Per-Class Accuracy**  
   - Overall accuracy might hide the fact that minority classes are **completely misclassified**.
3. **Data Distribution**  
   - If you have control over dataset collection, strive for **balanced** categories.  
   - In real scenarios, consider **resampling** (oversampling minority, undersampling majority) or **data augmentation** techniques.
4. **Future Solutions**  
   - Weighted loss functions, specialized sampling, or other methods can mitigate imbalance.

---

## 9. Further Exploration

- **Try Weighted Cross-Entropy**: Give higher loss weight to minority class.  
- **Oversample** the minority class or **undersample** the majority.  
- **Confusion Matrix**: Visualize the confusion matrix to see how the model is biased.  
- **F1-Score / Precision / Recall**: Use additional metrics beyond accuracy, especially in imbalanced situations.

**End of Notes – "Data: CodeChallenge – Unbalanced Data"** 
```