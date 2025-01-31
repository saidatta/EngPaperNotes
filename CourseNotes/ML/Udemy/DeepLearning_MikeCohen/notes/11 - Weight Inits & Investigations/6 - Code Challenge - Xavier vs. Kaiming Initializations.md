## 1. Overview

In this challenge, we **compare** two popular weight initialization schemes—**Xavier** and **Kaiming**—by training a feedforward model on the **Wine Quality** dataset **multiple times**. Specifically:

1. Implement two identical network architectures, differing only in **weight initialization** (Xavier vs. Kaiming).  
2. Run each initialization scheme **10 times** with fresh random seeds.  
3. Compare **final performance** via T-tests to see if one method statistically outperforms the other on:
   - Training Loss
   - Training Accuracy
   - Test Accuracy

**Key Learning Points**:
- How to merge code for **custom weight init** (Xavier or Kaiming) into an existing PyTorch workflow.  
- How to gather **statistics** over multiple runs and assess significance (T-test).  
- Gaining **practical** experience in setting up reproducible deep learning experiments.

---

## 2. Dataset and Basic Model Setup

### 2.1. Wine Quality Data

We use the **Wine Quality** dataset (binary classification: good vs. not-good). The code below follows previous notes/examples where we:

1. Load the **wine-quality** dataset.
2. Convert the **quality** rating into a **binary** label.  
3. Create train/test splits, standardize features if necessary.

**Example** (simplified snippet, adjust as needed):

```python
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# 1) Load or import the wine dataset 
# (Depending on your local path or custom function)
# Suppose X, y are numpy arrays, y in {0,1}

# 2) Optional scaling
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    Xz, y, test_size=0.1, random_state=42
)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)
```
```

### 2.2. Model Definition

We define a **feedforward network** with a few layers:

```python
```python
class WineNet(nn.Module):
    def __init__(self, in_features, n_hidden=32, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
```

**Note**: We keep architecture **identical** for both initialization schemes (Xavier vs. Kaiming).

---

## 3. Defining the Training Routine

A basic function to:
1. Create an **optimizer** (e.g., `SGD` or `Adam`).  
2. Train over a certain number of **epochs**.  
3. Track **loss**, **train accuracy**, and **test accuracy**.

```python
```python
def train_model(net, 
                Xtrain, ytrain, 
                Xtest, ytest, 
                epochs=600, 
                lr=0.001):
    
    optimizer = optim.SGD(net.parameters(), lr=lr)  # or Adam
    loss_fn   = nn.CrossEntropyLoss()
    
    # History
    loss_history  = []
    train_acc_hist = []
    test_acc_hist  = []
    
    for epoch in range(epochs):
        net.train()
        # forward
        y_hat = net(Xtrain)
        loss  = loss_fn(y_hat, ytrain)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # compute train accuracy
        with torch.no_grad():
            preds_train = torch.argmax(y_hat, dim=1)
            train_acc = (preds_train == ytrain).float().mean().item()
        train_acc_hist.append(train_acc)
        
        # compute test accuracy
        net.eval()
        with torch.no_grad():
            y_hat_test = net(Xtest)
            preds_test = torch.argmax(y_hat_test, dim=1)
            test_acc = (preds_test == ytest).float().mean().item()
        test_acc_hist.append(test_acc)
    
    return loss_history, train_acc_hist, test_acc_hist
```
```

---

## 4. Custom Weight Initializations

We’ll create two functions to **apply** Xavier or Kaiming (for weights only):

```python
```python
import torch.nn.init as init

def xavier_init_weights(net):
    """Apply Xavier normal initialization to all linear layers' weights."""
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
    return net

def kaiming_init_weights(net):
    """Apply Kaiming uniform initialization (assuming ReLU) to all linear layers' weights."""
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param, nonlinearity='relu')
    return net
```
```

**Note**: We leave **bias** parameters at PyTorch’s default (often uniform in a small range). Alternatively, we could also re-init biases to zero or another small constant.

---

## 5. Single Run Example

We first **test** each scheme on a single run to ensure code correctness and gather initial performance curves.

```python
```python
# Single-run test
input_size = X_train_t.shape[1]

# 1) Xavier init
model_xavier = WineNet(in_features=input_size)
model_xavier = xavier_init_weights(model_xavier)

lx, trax_x, teax_x = train_model(model_xavier, X_train_t, y_train_t, X_test_t, y_test_t)

# 2) Kaiming init
model_kaiming = WineNet(in_features=input_size)
model_kaiming = kaiming_init_weights(model_kaiming)

lk, trax_k, teax_k = train_model(model_kaiming, X_train_t, y_train_t, X_test_t, y_test_t)

# Plot to compare
epochs_arr = range(len(lx))
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(epochs_arr, lx, label='Xavier')
plt.plot(epochs_arr, lk, label='Kaiming')
plt.title('Loss vs. Epoch')
plt.legend()

plt.subplot(1,3,2)
plt.plot(epochs_arr, trax_x, label='Xavier')
plt.plot(epochs_arr, trax_k, label='Kaiming')
plt.title('Train Accuracy')

plt.subplot(1,3,3)
plt.plot(epochs_arr, teax_x, label='Xavier')
plt.plot(epochs_arr, teax_k, label='Kaiming')
plt.title('Test Accuracy')

plt.tight_layout()
plt.show()
```
```

**Observation**:
- Results vary from run to run due to randomness in the data splits, initialization, etc.  
- The curves might hint one method is better in training or testing, but not conclusive.

---

## 6. Full Experiment: Multiple Runs + Statistical Tests

We repeat each scheme **10 times**. This yields 20 total runs (10 for Xavier, 10 for Kaiming). We then compare final results with a **T-test**.

```python
```python
n_runs = 10
results = {
    'xavier': {'loss': [], 'train_acc': [], 'test_acc': []},
    'kaiming': {'loss': [], 'train_acc': [], 'test_acc': []}
}

for run_i in range(n_runs):
    # Xavier
    mx = WineNet(input_size)
    mx = xavier_init_weights(mx)
    lx, trax_x, teax_x = train_model(mx, X_train_t, y_train_t, X_test_t, y_test_t)
    
    # Kaiming
    mk = WineNet(input_size)
    mk = kaiming_init_weights(mk)
    lk, trax_k, teax_k = train_model(mk, X_train_t, y_train_t, X_test_t, y_test_t)
    
    # We'll average the last 5 epochs for stable estimate
    results['xavier']['loss'].append(  np.mean(lx[-5:]) )
    results['xavier']['train_acc'].append( np.mean(trax_x[-5:]) )
    results['xavier']['test_acc'].append(  np.mean(teax_x[-5:]) )
    
    results['kaiming']['loss'].append(  np.mean(lk[-5:]) )
    results['kaiming']['train_acc'].append( np.mean(trax_k[-5:]) )
    results['kaiming']['test_acc'].append(  np.mean(teax_k[-5:]) )
    
    print(f"Run {run_i+1}/{n_runs} done.")
```
```

**Note**: This might take minutes depending on your system and number of epochs.

### 6.1. Visualizing Distribution of Final Metrics

We plot the final (averaged over last 5 epochs) **loss**, **train accuracy**, **test accuracy** as points for each of the 10 runs, side by side for Xavier vs. Kaiming.

```python
```python
metrics_list = ['loss','train_acc','test_acc']

plt.figure(figsize=(12,4))

for i, metric in enumerate(metrics_list):
    plt.subplot(1,3,i+1)
    x_vals = results['xavier'][metric]
    k_vals = results['kaiming'][metric]
    
    # simple x offsets for plotting
    x_xav = [0]*len(x_vals)
    x_kai = [1]*len(k_vals)
    
    plt.scatter(x_xav, x_vals, color='blue', alpha=0.7, label='Xavier')
    plt.scatter(x_kai, k_vals, color='red', alpha=0.7, label='Kaiming')
    
    # optional: also show means
    plt.plot([-0.1,0.1],[np.mean(x_vals)]*2, color='blue')
    plt.plot([0.9,1.1],[np.mean(k_vals)]*2, color='red')
    
    plt.xlim([-0.5,1.5])
    plt.title(metric)
    
    if i==0:
        plt.legend()

plt.tight_layout()
plt.show()
```
```

**Interpretation**:
- **Loss**: do we see lower final loss consistently for one scheme?  
- **Train Accuracy**: does one scheme systematically yield better training?  
- **Test Accuracy**: is there a generalization difference?

### 6.2. T-Test Comparison

We do a simple **independent samples** T-test (`ttest_ind`) for each metric:

```python
```python
from scipy.stats import ttest_ind

for metric in metrics_list:
    x_vals = results['xavier'][metric]
    k_vals = results['kaiming'][metric]
    
    t_val, p_val = ttest_ind(x_vals, k_vals, equal_var=False)  # Welch's T-test
    print(f"{metric.upper()}:\n Xavier mean={np.mean(x_vals):.4f}, Kaiming mean={np.mean(k_vals):.4f} "
          f"=> t={t_val:.3f}, p={p_val:.3f}")
    print("---")
```
```

We interpret:
- If **p < 0.05**, likely a significant difference in means.  
- The **sign** of **t** indicates which scheme has larger mean (positive t => first group has higher mean, negative => second group higher).

---

## 7. Interpreting Results and Possible Extensions

### 7.1. Observations

- **Train Loss / Accuracy**: One scheme may provide faster or deeper training convergence.  
- **Test Accuracy**: Even if training diverges, the final test accuracy might be **very similar** or slightly different.  
- Sometimes **Kaiming** might overfit more strongly (higher training accuracy, not necessarily better test accuracy).

### 7.2. Further Improvements / Exploration

1. **Try `Adam`** instead of `SGD`, reduce epochs.  
2. Increase the **number of runs** (20, 50, 100) for more robust stats, albeit longer runtime.  
3. Plot the entire **training curve** for each run to see variance in learning dynamics.  
4. Run on a **different** dataset (MNIST, CIFAR) for deeper architecture, see if results differ.

---

## 8. Summary

1. **Xavier vs. Kaiming**: Two widely respected weight init schemes.  
2. **Multiple runs** essential for robust conclusions. Single-run differences can be random.  
3. **Statistics**: A T-test helps confirm whether observed differences are meaningful or mere noise.  
4. **Wine Quality** experiment shows typical outcomes: Kaiming often yields **faster** or **deeper** training, but final generalization can be near-similar—sometimes slightly better or slightly worse, depending on the model/dataset.

---

## 9. Further Reading

1. **Glorot & Bengio (2010)**: *Understanding the difficulty of training deep feedforward neural networks*  
2. **He et al. (2015)**: *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*  
3. **PyTorch** `nn.init`: [Docs](https://pytorch.org/docs/stable/nn.init.html)  
4. **Deep Learning Book**, Goodfellow et al., sections on *Initialization and Optimization*.

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** (e.g. `WeightInit_XavierVsKaiming.md`).  
2. **Paste** the entire markdown (including frontmatter `---`).  
3. Add or modify any **internal links** (e.g., `[[Statistical Tests in Python]]`).  
4. Adjust headings/tags to fit your vault structure.

These notes walk you through a **practical** experimental design: comparing **Xavier** and **Kaiming** initialization on a **Wine Quality** classification task, collecting results over multiple runs, and performing **t-tests** to gauge **statistical significance** of any observed differences.