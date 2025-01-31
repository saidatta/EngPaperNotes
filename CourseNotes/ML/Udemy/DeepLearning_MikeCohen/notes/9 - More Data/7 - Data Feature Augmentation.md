aliases: [Feature Augmentation, Data Augmentation, Additional Features, Statistical Significance]
tags: [Data, FeatureEngineering, QWERTY, PyTorch, Experiment, T-test]
## Overview
In this lecture, we explore how **feature augmentation**—adding **new features** (e.g., non-linear combinations of existing features)—impacts a network’s performance. We use the **QWERTY** dataset (originally 2D) and create a **third feature**: the Euclidean distance to the origin. We then compare model performance on:
- The **original 2D** dataset
- The **augmented 3D** dataset

By training each model multiple times and performing a **statistical t-test**, we determine whether the additional feature yields a **significant** performance boost.

---
## 1. QWERTY Dataset Recap

We have a **3-class** synthetic dataset in 2D:
1. Blue cluster near \([1,2]\)
2. Black cluster near \([4,4]\)
3. Red cluster near \([4,-2]\)

Typically, we represent each sample as \((x, y)\). But we suspect **distance to the origin** might be helpful in distinguishing the blue cluster (closer to \((0,0)\)) from the others.

---

## 2. Creating the Data (2D → 3D)

### 2.1 Generating 2D QWERTY

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

# 1) Number of samples per cluster
nPerClust = 300

# 2) Cluster centers
A_center = [1, 2]
B_center = [4, 4]
C_center = [4, -2]

# 3) Generate data
A_data = np.random.randn(nPerClust, 2) + A_center
B_data = np.random.randn(nPerClust, 2) + B_center
C_data = np.random.randn(nPerClust, 2) + C_center

# 4) Label them: 0,1,2
A_labels = np.zeros((nPerClust,))
B_labels = np.ones((nPerClust,))
C_labels = 2*np.ones((nPerClust,))

# 5) Concatenate
data2D = np.vstack((A_data,B_data,C_data))
labels  = np.concatenate((A_labels,B_labels,C_labels)).astype(int)

# Let's visualize
plt.figure(figsize=(5,5))
plt.scatter(data2D[:,0], data2D[:,1], c=labels, alpha=0.5)
plt.title("QWERTY data in 2D")
plt.axis('equal')
plt.show()
```

### 2.2 Computing Distance to Origin

We add a **3rd dimension**: \(\mathrm{distToOrigin} = \sqrt{x^2 + y^2}\)

```python
dist2orig = np.sqrt( data2D[:,0]**2 + data2D[:,1]**2 )

print("Distances shape:", dist2orig.shape)
print("Few example distances:", dist2orig[:6])
```

We create **augmented** data by concatenating this new feature:

```python
data3D = np.column_stack([data2D, dist2orig])
print("2D data shape:", data2D.shape)
print("3D data shape:", data3D.shape)
```

---

## 3. Preparing Train/Test Splits

We’ll create two PyTorch datasets:
1. **Original** (2D)
2. **Augmented** (3D)

Each split into train/test loaders.

```python
def make_dataLoaders(useExtraFeature=False, test_size=0.1):
    # X_ ? data2D or data3D
    X = data3D if useExtraFeature else data2D
    Y = labels
    
    # Convert to PyTorch
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_t, Y_t, test_size=test_size, random_state=42
    )
    
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset  = TensorDataset(X_test,  Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=True)
    
    return train_loader, test_loader
```

---

## 4. Defining the Model for 2D or 3D Inputs

We want a dynamic input layer size: **2** if `useExtraFeature=False`, **3** if `True`. We do so in the constructor:

```python
class QwertyModel(nn.Module):
    def __init__(self, useExtraFeature=False):
        super(QwertyModel, self).__init__()
        self.useExtraFeature = useExtraFeature
        
        # input dimension = 3 if we have the extra distance feature, else 2
        input_dim = 3 if useExtraFeature else 2
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 categories
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape might be [batch,3] if useExtraFeature
        # or [batch,2] if not
        # but let's see how we handle it carefully

        # If we stored x in 3D but want only 2D,
        # or x in 3D but only the first 2 columns matter, etc.
        # Actually, we can let the dataloader handle this if we want

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

Alternatively, you might store both `data3D` in your dataset but only feed the first 2 columns if `useExtraFeature=False`. It depends on your code style.

---

## 5. Training Function

```python
def trainQwertyModel(useExtraFeature=False, epochs=100):
    # Make data
    train_loader, test_loader = make_dataLoaders(useExtraFeature=useExtraFeature)
    
    # Build model
    model = QwertyModel(useExtraFeature=useExtraFeature)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_acc, test_acc, losses = [], [], []
    
    for epoch in range(epochs):
        # train
        model.train()
        batch_accs, batch_losses = [], []
        for Xb, Yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, Yb)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            preds = torch.argmax(logits, 1)
            acc   = (preds == Yb).float().mean().item()
            batch_accs.append(acc)
        
        train_acc.append(np.mean(batch_accs))
        losses.append(np.mean(batch_losses))
        
        # test
        model.eval()
        batch_accs_test = []
        with torch.no_grad():
            for Xb_t, Yb_t in test_loader:
                logits_t = model(Xb_t)
                preds_t  = torch.argmax(logits_t, 1)
                acc_t    = (preds_t == Yb_t).float().mean().item()
                batch_accs_test.append(acc_t)
        test_acc.append(np.mean(batch_accs_test))
    
    return train_acc, test_acc, losses, model
```

---

## 6. Running a Single Trial

```python
trainA, testA, lossesA, modelA = trainQwertyModel(useExtraFeature=False, epochs=100)
trainB, testB, lossesB, modelB = trainQwertyModel(useExtraFeature=True,  epochs=100)

print("Final accuracy (2D):", testA[-1])
print("Final accuracy (3D):", testB[-1])
```

### 6.1 Visualization

You can plot the **loss**, **train accuracy**, **test accuracy** as usual:

```python
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(lossesA, label='2D')
plt.plot(lossesB, label='3D')
plt.title("Loss over epochs")
plt.legend()

plt.subplot(1,3,2)
plt.plot(trainA, label='2D Train')
plt.plot(trainB, label='3D Train')
plt.title("Train Accuracy")
plt.legend()

plt.subplot(1,3,3)
plt.plot(testA, label='2D Test')
plt.plot(testB, label='3D Test')
plt.title("Test Accuracy")
plt.legend()

plt.show()
```

---

## 7. Multiple Trials + Statistical Test

To robustly check if the 3D version truly helps, we:
1. Repeat training **multiple times** (e.g., 10 runs) for each approach.
2. Compare final test accuracies via a **t-test**.

```python
nRuns = 10
acc_2D = []
acc_3D = []

for _ in range(nRuns):
    # 2D
    _, test_2d, _, _ = trainQwertyModel(useExtraFeature=False, epochs=80)
    acc_2D.append(test_2d[-1])
    
    # 3D
    _, test_3d, _, _ = trainQwertyModel(useExtraFeature=True,  epochs=80)
    acc_3D.append(test_3d[-1])

print("2D final accuracies:", acc_2D)
print("3D final accuracies:", acc_3D)

# t-test
tval, pval = ttest_ind(acc_2D, acc_3D)
print(f"T-val = {tval:.3f}, P-val = {pval:.3e}")
```

- If **p-value < 0.05**, the difference is “statistically significant” under standard assumptions.  
- If not, we conclude no meaningful difference was detected between 2D and 3D augmentation in this scenario.

---

## 8. Discussion & Key Points

1. **Feature Augmentation**  
   - We introduced a **non-linear** feature (\(\sqrt{x^2 + y^2}\)), hoping to help classification.  
   - Sometimes models can learn this relation on their own, or it might genuinely help if the network or data is small.
2. **No Guarantee** of Benefit  
   - The network’s existing non-linear layers might already replicate that transformation internally.  
   - If the model is already at a performance ceiling or your new feature is not informative, the improvement might be marginal or zero.
3. **Statistical Testing**  
   - Doing repeated runs and using a **t-test** helps confirm whether differences are real or just random fluctuations.  
   - If **p-value** is high (\(>0.05\)), you can’t claim a significant improvement from the new feature.

### 8.1 When Does Feature Augmentation Help?
- **Lower-dimensional data** or **complex transformations** not easily discovered by the model might yield improvements.  
- **Domain knowledge** can guide which derived features might be relevant (e.g., angle, energy in signal processing).

### 8.2 Avoid Redundant Linear Features
- Adding purely **linear** combinations (e.g., \(x+y\)) rarely helps, because the input layer already learns linear transformations.

---

## 9. Takeaways & Practical Tips

- **Feature Engineering**: In some tasks, carefully engineered non-linear features can significantly help (common in domain-specific problems).
- **Network’s Non-Linearity**: Large networks with enough capacity might already approximate such transformations (distance, etc.) internally.
- **Statistical Validation**: Repeated experiments + a **t-test** (or more robust stats) clarify if an approach truly yields consistent improvements.

---

## 10. Further Explorations
1. **Different Non-Linear Features**:  
   - Try angle to origin (\(\arctan2(y,x)\)), or polynomial expansions \(x^2, xy, y^2\).
2. **Bigger Networks**:  
   - If the network is larger, the benefit of augmented features might vanish because the model can discover them itself.
3. **Different Architecture**:  
   - Test if a deeper or wider network reaps bigger gains from feature augmentation.
4. **Confidence Intervals**:  
   - Instead of (or in addition to) a t-test, consider plotting confidence intervals or using non-parametric tests for small sample repeated runs.

---

**End of Notes – “Data: Data Feature Augmentation”**  
```