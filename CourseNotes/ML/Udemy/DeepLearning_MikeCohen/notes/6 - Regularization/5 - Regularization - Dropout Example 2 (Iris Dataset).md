## Table of Contents
1. [[Motivation]]
2. [[Dataset & Data Preparation]]
3. [[Model Architecture with Dropout]]
4. [[Training Function & Code Structure]]
5. [[Experiment: Varying Dropout Rates]]
6. [[Results & Observations]]
7. [[Key Takeaways]]

---

## 1. Motivation
In the previous example, we trained on a custom, non-linear dataset.  
Here, we replicate the **dropout** process on a **standard small dataset**: **Iris**.  
- Iris typically separates well with **low-complexity** models.  
- We suspect dropout **might not** help if the dataset + model architecture is not sufficiently large or complex.

---

## 2. Dataset & Data Preparation

```python
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1) Load iris
iris_data = load_iris()
X = iris_data.data         # shape: (150, 4)
y = iris_data.target       # 3 classes: 0,1,2

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# 3) Convert to PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 4) DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
```

**Notes**:
- **Iris** has \(150\) samples, **4** features, and **3** class labels.  
- We use a **batch size** of 16, typical for small sets.

---

## 3. Model Architecture with Dropout
We create a **2-layer** MLP (input: 4 features, hidden: 12 units, output: 3 classes). We incorporate **dropout** after each linear + ReLU block.

```python
```python
class IrisDropoutModel(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super().__init__()
        self.input  = nn.Linear(4, 12)  # 4 -> 12
        self.hidden = nn.Linear(12, 12) # 12 -> 12 (optional hidden)
        self.output = nn.Linear(12, 3)  # 12 -> 3 (3-class classification)
        self.dr = dropoutRate          # dropout probability

    def forward(self, x):
        # Input layer -> ReLU -> Dropout
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training)

        # Hidden layer -> ReLU -> Dropout
        x = F.relu(self.hidden(x))
        x = F.dropout(x, p=self.dr, training=self.training)

        # Output layer (3 classes => no activation yet, typically used w/ CrossEntropyLoss)
        x = self.output(x)
        return x
```

**Key Points**:
- We use `self.training` to ensure dropout is **only** active in `train()` mode.  
- The final layer is simply **linear** (no activation), suitable for **CrossEntropyLoss** in PyTorch.

---

## 4. Training Function & Code Structure

### 4.1. Create & Train Model
```python
```python
def create_model(dropout_rate):
    model = IrisDropoutModel(dropout_rate)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, loss_fn, optimizer

def train_model(model, loss_fn, optimizer, train_loader, test_loader, epochs=500):
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        ### 1) Training phase
        model.train()  # dropout = ON
        batch_acc = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)      # shape: (batch_size, 3)
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # compute training accuracy
            preds = torch.argmax(y_pred, axis=1)
            acc   = (preds == y_batch).float().mean().item()
            batch_acc.append(acc)

        train_accs.append(np.mean(batch_acc))

        ### 2) Evaluation phase
        model.eval()   # dropout = OFF
        with torch.no_grad():
            for X_testb, y_testb in test_loader:
                y_pred_test = model(X_testb)
                test_preds  = torch.argmax(y_pred_test, axis=1)
                test_acc    = (test_preds == y_testb).float().mean().item()
        test_accs.append(test_acc)

    return train_accs, test_accs
```

- We cycle through **train** mode and **eval** mode each epoch.
- Store accuracy across epochs in `train_accs` / `test_accs`.

---

## 5. Experiment: Varying Dropout Rates

### 5.1. Running the Experiment
We systematically vary dropout rate \(p\) from 0.0 to 1.0 in discrete steps, training for a fixed number of epochs.

```python
```python
dropout_rates = np.linspace(0, 1, 10)
final_train_accuracies = []
final_test_accuracies  = []

for dr in dropout_rates:
    model, loss_fn, optimizer = create_model(dr)
    train_accs, test_accs = train_model(model, loss_fn, optimizer, train_loader, test_loader, epochs=500)

    # Take the average of the last 50 epochs to smooth
    final_train_accuracies.append(np.mean(train_accs[-50:]))
    final_test_accuracies.append(np.mean(test_accs[-50:]))

# Plot the results
plt.figure(figsize=(6,4))
plt.plot(dropout_rates, final_train_accuracies, 'o-', label='Train Accuracy')
plt.plot(dropout_rates, final_test_accuracies,  'o-', label='Test Accuracy')
plt.xlabel("Dropout Probability p")
plt.ylabel("Average Accuracy (last 50 epochs)")
plt.title("Iris Dataset: Effect of Dropout Rate")
plt.legend()
plt.show()
```

### 5.2. Expected Runtime
- This experiment is relatively quick, since **Iris** is small.  
- Each training session is 500 epochs with a moderate batch size.

---

## 6. Results & Observations

### 6.1. Typical Outcome
- With **p=0.0** (no dropout), the model likely achieves near **perfect** or very high accuracy, both train and test.  
- As `p` increases, **train** and **test** accuracies **decrease**.  
- The best performance might remain at **p=0.0** or a small fraction.

### 6.2. Interpretation
1. **Small Dataset**: Iris only has 150 samples, so heavy dropout often doesn’t help.  
2. **Simple Model**: The model is not extremely large/deep. Dropout often benefits bigger networks or more extensive data.  
3. **Over-regularizing**: Large `p` might hamper the network’s ability to learn the limited data.

---

## 7. Key Takeaways
1. **Dropout** can sometimes degrade performance for:
   - **Small** datasets, or
   - **Shallow** / **low-parameter** models.
2. **Empirical Testing**: Always treat dropout rate as a hyperparameter:
   - Evaluate different `p` in `[0,1]`, track train/test accuracy.  
   - Choose a rate that balances generalization vs. underfitting.
3. **Next Steps**:
   - Try dropout on **larger** or **more complex** tasks, where it often leads to a beneficial increase in test accuracy.  
   - Explore **other** regularization forms (e.g., **L2** / **weight decay**), which can also improve performance in certain settings.

**Transition**  
In upcoming lectures, we’ll move on to **weight-based** regularization methods (L1/L2) to further explore ways of reducing overfitting in deep learning models.

---

**End of Notes**  
Through this example, you see how **dropout** in a small dataset (like Iris) might not yield the gains it does in larger or more complex scenarios. Instead, it can even reduce accuracy—reinforcing the importance of **context** and **hyperparameter** experimentation in deep learning.