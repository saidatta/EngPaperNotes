Below is a **very detailed, Obsidian-friendly** set of notes on **why it’s important to keep batch sizes consistent** in deep learning, highlighting how tiny batches in the test set can lead to misleading or strange accuracy plots. This lecture uses a toy “qwerties doughnuts” dataset as an example.

## Table of Contents
1. [[Overview]]
2. [[Dataset & Setup]]
3. [[Unequal Batch Sizes: The Core Problem]]
4. [[Code Explanation & Visualization]]
5. [[Why Tiny Batches Cause Strange Accuracy "Bands"]]
6. [[Correcting the Issue]]
7. [[Key Takeaways]]

---
## 1. Overview
**Goal**: Demonstrate how having **inconsistent or extremely small** mini-batch sizes (especially in the test set) can produce **odd, misleading** accuracy results. We see how a single **very small** batch (e.g., batch size of 2) can quantize the accuracy into discrete “bands.”

**Key Idea**:  
- When a test batch has only a few samples (like 2), the accuracy in that batch can only be \(0\%, 50\%,\) or \(100\%\). Averaging that batch’s accuracy with others can create conspicuous “stripes” or “bands” in the accuracy plot.

---

## 2. Dataset & Setup
We use a **non-linear “qwerties doughnuts”** dataset (circular distributions). The code includes:
1. Generating the data.  
2. Building a **small MLP** classifier.  
3. Training with a **mini-batch** size (e.g., 16) for the **train** set.  
4. Testing with a **split** that yields an awkward test batch distribution.

*(Pseudocode illustration below; exact shapes or hyperparameters may vary in your final code.)*

```python
```python
# Suppose we have X_train, y_train, X_test, y_test from the "qwerties doughnuts" dataset
# We'll create a train_loader with batch_size=16, drop_last=True (normal for training)
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

# Problematic test_loader with awkward size
test_batchsize = len(X_test) - 2  # e.g. 40 total => batch of 38 + leftover batch of 2
test_loader  = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False)
```

---

## 3. Unequal Batch Sizes: The Core Problem
- **Train loader** has uniform mini-batches of size 16.  
- **Test loader** has 2 batches: one large (e.g., 38) and one tiny (2).  
- The **accuracy** for the 2-sample batch can only be 0%, 50%, or 100%, leading to weird averaged results over the test set.

---

## 4. Code Explanation & Visualization
During training, each epoch we do something like:

```python
```python
train_accs = []
test_accs  = []

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # forward, compute loss, backprop, step
        ...
    
    # Evaluate test accuracy
    model.eval()
    batch_accs_test = []
    for Xb_test, yb_test in test_loader:
        # predictions
        preds_test = model(Xb_test).argmax(dim=1)
        acc_test   = (preds_test == yb_test).float().mean().item()
        batch_accs_test.append(acc_test)
    
    # Possibly average across test batches
    epoch_test_acc = sum(batch_accs_test)/len(batch_accs_test)
    test_accs.append(epoch_test_acc)
```

- If the **test loader** has **two** batches (e.g., 38 samples + 2 samples), we effectively **average** the accuracy from a 38-sample batch with a 2-sample batch.  
- The 2-sample batch’s accuracy can only be \(\{0\%, 50\%, 100\%\}\). This results in discrete “bands” or steps in overall test accuracy each epoch.

---

## 5. Why Tiny Batches Cause Strange Accuracy "Bands"

1. **Small Number of Samples** → **Coarse Accuracy**  
   - Accuracy = \(\text{Correct}/\text{Total}\). With only 2 samples, you can get \(\frac{0}{2}=0\%\), \(\frac{1}{2}=50\%\), or \(\frac{2}{2}=100\%\).  
2. **Averaging**: 
   - The final test accuracy for that epoch is something like `0.5*(Acc_of_batch_1 + Acc_of_batch_2)`, meaning the second small batch can drastically shift the final accuracy.  
3. **Visualization**:  
   - The accuracy plot forms horizontal “stripes” or “bands” at discrete values instead of a smooth curve.

---

## 6. Correcting the Issue
- **Consistent Batch Size**: Typically, set the **test_loader** to have **1** batch (the entire test set) or at least similarly sized mini-batches as training.  
- **drop_last=False** for test**:** Usually keep `False` so you do not discard any test samples.  
- If your test set is large, it’s fine to break it into multiple smaller, but **consistent** batches to avoid memory issues—but avoid extremely small leftover batches.

### Example Fix:
```python
```python
test_loader = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset),  # single batch
    shuffle=False
)
```
Or specify a more standard batch size (like 16 or 32) that neatly divides your test set, if possible.

---

## 7. Key Takeaways
1. **Small leftover test batches** can produce **weird** accuracy banding due to the coarse granularity of correct/incorrect over few samples.  
2. **Ensure** consistent batch sizing (or at least avoid extremely tiny test batches) to get **smooth** and **reliable** accuracy measurements.  
3. Typically, **test** sets are processed in **one** batch (the entire test set) if memory allows, or at least in uniform mini-batches.  
4. This phenomenon is a good illustration that test accuracy depends not only on the model but also on **how** metrics are aggregated.

**Conclusion**:  
Mini-batch training is crucial for speed and some regularization effects, **but** leftover or very small test batches can skew accuracy results. Always confirm that your test batch sizes are **appropriate** for accurate, stable metric reporting.