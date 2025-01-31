## 1. Overview

In the prior lectures, we learned:
- **Why** random weight initialization is essential (to break symmetry).  
- The risks of **too-small** or **too-large** variance: vanishing vs. exploding gradients.  

This **code challenge** explores how **different weight variances** at initialization affect a model’s **final** performance on **MNIST**. Specifically, we:

1. Define 25 **logarithmically spaced** values for \(\sigma\) (standard deviation) ranging from \(10^{-4}\) to \(10^1\).  
2. Initialize **all** trainable parameters to random normal values with those \(\sigma\).  
3. Train the model on MNIST for each \(\sigma\), collecting **final** accuracies.  
4. Visualize how performance depends on initial variance.

---

## 2. Key Objectives

1. Understand how to modify **PyTorch** weights before training.  
2. Observe **accuracy** vs. **weight variance**.  
3. Inspect **learned weight distributions** post-training for different initializations.

---

## 3. Dataset and Model Setup

### 3.1. Data Preparation

```python
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) MNIST transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2) Load MNIST train/test
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
```
```

### 3.2. Model Architecture

We use a **standard** feed-forward network. This architecture is similar to previous MNIST demos:

```python
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)        # final logits
        return x
```
```

### 3.3. Training Function

```python
```python
def train_model(net, train_loader, test_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc_hist = []
    test_acc_hist  = []
    
    for epoch in range(epochs):
        net.train()
        correct_train = 0
        total_train   = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = net(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            
            # measure train accuracy in this batch
            preds = y_hat.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train   += len(y_batch)
        
        train_acc = correct_train / total_train
        train_acc_hist.append(train_acc)
        
        # evaluate on test set
        net.eval()
        correct_test = 0
        total_test   = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                out = net(X_t)
                preds_t = out.argmax(dim=1)
                correct_test += (preds_t == y_t).sum().item()
                total_test   += len(y_t)
        test_acc = correct_test / total_test
        test_acc_hist.append(test_acc)
        
        # Print for monitoring
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
    
    return train_acc_hist, test_acc_hist
```
```

**Note**: We collect training/test accuracy per epoch. Later, we might **average** final epochs for a single final accuracy measure.

---

## 4. Experiment: Varying Weight Initialization Standard Deviation

### 4.1. Defining a Range of std Devs

We create **25** logarithmically spaced values from \(10^{-4}\) to \(10^{1}\):

```python
```python
num_points = 25
stdevs = np.logspace(-4, 1, num_points)  # e.g., [1e-4, 1.38e-4, ..., 10]
print(stdevs)
```
```

### 4.2. Main Loop: Building + Training Models

**Key Step**: For each \(\sigma\), we:
1. Instantiate a **fresh** network.  
2. Overwrite **all** parameters with normal random values \(\mathcal{N}(0, \sigma^2)\).  
3. Train for 10 epochs, collect accuracy.  
4. Extract **final** weights for histogram.

```python
```python
nhistBins = 80
final_acc = []
hist_params = []

for sd in stdevs:
    # 1) fresh network
    net = MNISTNet()
    
    # 2) Initialize all parameters to normal(0, sd)
    with torch.no_grad():
        for name, param in net.named_parameters():
            param.data = torch.randn_like(param.data) * sd  # mean=0, std=sd
    
    # 3) Train
    train_acc, test_acc = train_model(net, train_loader, test_loader, epochs=10, lr=0.001)
    
    # Take the last few epochs' test acc, or just last epoch
    final_acc_val = np.mean(test_acc[-3:])  # average of last 3
    final_acc.append(final_acc_val)
    
    # 4) Collect final weights for histogram
    all_params = np.array([])
    with torch.no_grad():
        for name, param in net.named_parameters():
            wvals = param.detach().flatten().cpu().numpy()
            all_params = np.concatenate((all_params, wvals))
    
    # binning
    histVals, xbinedge = np.histogram(all_params, bins=nhistBins, density=False)
    xbincenters = 0.5*(xbinedge[:-1]+xbinedge[1:])
    hist_params.append((sd, xbincenters, histVals))
    
    print(f"Finished SD={sd:.5f}, Final Acc={final_acc_val*100:.2f}%\n")
```
```

**Note**: This loop might take **minutes** to run (e.g., 6–10 minutes). We store:
- `final_acc` as a list of final accuracies.
- `hist_params` as a list of tuples `(sd, bin_centers, histogram_counts)` for each \(\sigma\).

---

## 5. Visualizing the Results

### 5.1. Accuracy vs. Standard Deviation

```python
```python
plt.figure(figsize=(7,5))
plt.plot(stdevs, np.array(final_acc)*100, 'o-', ms=5)
plt.xscale('log')
plt.ylim([0, 100])
plt.xlabel("Weight Init Std Dev (log scale)")
plt.ylabel("Final Test Accuracy (%)")
plt.title("Accuracy vs. Weight Initialization Std Dev")
plt.grid(True)
plt.show()
```
```

**Interpretation**:
- Often poor performance at **very low** std dev (\(10^{-4}\)), where weights are nearly zero → risk vanishing gradients.
- Also poor at **very high** std dev (e.g., \(\sigma=10\)), where weights can explode → risk exploding gradients.
- A **sweet spot** in between yields the **best** accuracy.

### 5.2. Weight Distribution Histograms

We can **overlay** the final weight distributions for each \(\sigma\). Because there are 25 lines, we might color-map them:

```python
```python
plt.figure(figsize=(8,6))

cmap = plt.cm.get_cmap('viridis', num_points)
for i, (sd, xbins, histcount) in enumerate(hist_params):
    # Normalize histcounts if desired
    plt.plot(xbins, histcount, 
             color=cmap(i), 
             label=f"SD={sd:.4f}" if i in [0, num_points-1] else None, 
             alpha=0.7)

plt.xlabel("Weight Value")
plt.ylabel("Histogram Count")
plt.title("Learned Weight Distributions (All Layers) for Different Inits")
plt.legend()
plt.show()
```
```

**Possible Observations**:
- **Small \(\sigma\)** → final weights cluster near 0 even after training.  
- **Large \(\sigma\)** → final weights spread wide, possibly saturating neurons or causing instability.  
- Mid-range \(\sigma\) → moderate spread, typically yields highest accuracy.

---

## 6. Interpretations and Insights

1. **Vanishing Gradients** (Low \(\sigma\)):  
   - If weights start extremely close to zero, the network can’t update effectively; partial derivatives vanish.  
   - Final performance remains ~random or suboptimal.

2. **Exploding Gradients** (High \(\sigma\)):  
   - Large initial weights can blow up activations, causing unstable training or plateauing at poor solutions.  
   - Final accuracy may degrade dramatically.

3. **Sweet Spot**:  
   - A moderate \(\sigma\) fosters stable gradient flows and good learning.  
   - **Xavier** or **Kaiming** initialization aim to *automatically* find suitable scale for each layer.

4. **Back Propagation** can’t fix *extremely* poor initial scales:  
   - If you start too far off, it’s difficult for gradient descent to converge to a good parameter regime within limited epochs.

---

## 7. Summary and Next Steps

**Key Lessons**:
1. **Initial variance** matters: too small or too big stalls or degrades learning.  
2. Modern networks use specialized initialization (Xavier, Kaiming) to systematically **balance** variance across layers.  
3. This challenge helped reinforce **manipulating** weight parameters **manually** before training and analyzing **final** distributions vs. **final** accuracy.

**Next**:
- We’ll implement **Xavier** and **Kaiming** initialization in code.  
- Compare them to naive random inits and see how these strategies automatically find a near-optimal variance range.

---

## 8. Further Reading

- **Glorot & Bengio (2010)**: *Understanding the difficulty of training deep feedforward neural networks* ([link](http://proceedings.mlr.press/v9/glorot10a.html))  
- **He et al. (2015)**: *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* ([link](https://arxiv.org/abs/1502.01852))  
- **PyTorch** `nn.init`: [Documentation](https://pytorch.org/docs/stable/nn.init.html) for built-in init functions  
- **Deep Learning Book** (Goodfellow, Bengio, Courville): Sections on *optimization* and *parameter initialization*.

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your vault (e.g., `WeightVarianceCodeChallenge.md`).  
2. **Paste** the entire markdown content (including frontmatter) above.  
3. (Optional) Add your own internal links, e.g., `[[Gradient Descent Challenges]]`, or rename headings for your indexing needs.  

This note provides a **complete** walkthrough of the **weight variance challenge** on MNIST, illustrating how **std dev** at initialization strongly influences **final accuracy** and **learned weight distribution**.