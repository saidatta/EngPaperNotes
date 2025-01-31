## Table of Contents
1. [[Context & Motivation]]
2. [[What are Training and Evaluation Modes?]]
3. [[Why Toggle Between Training and Eval?]]
4. [[PyTorch Functions: train(), eval(), no_grad()]]
5. [[Illustrative Training Loop]]
6. [[Practical Considerations]]
7. [[Key Takeaways]]

---
## 1. Context & Motivation
When building deep learning models (especially in PyTorch), we often talk about **“switching”** a model between:
- **Training mode**: where gradient updates occur and certain regularization layers (e.g., dropout, batch normalization) behave differently.
- **Evaluation mode**: where we only perform **forward passes** (inference) without updating model weights.

**Why does this matter?**  
- Some layers have **different behaviors** in training vs. inference.  
- We want to **turn off** gradient calculations and certain forms of regularization when evaluating on dev or test sets.

---

## 2. What are Training and Evaluation Modes?
### 2.1. Training Mode
- **Gradients** are computed for backpropagation.  
- Regularization layers such as:
  - **Dropout**: Randomly zeroes node outputs (activations).
  - **Batch Normalization**: Uses batch statistics (mean, variance) from the current minibatch.
- This mode is **enabled by default** in a newly created PyTorch model.

### 2.2. Evaluation Mode
- **No** gradients are computed for backprop.  
- **Dropout** should be deactivated (i.e., no random dropout).  
- **Batch Normalization** uses **running averages** (tracked during training) instead of mini-batch statistics.

---

## 3. Why Toggle Between Training and Eval?
1. **Prevent Overfitting**: We apply dropout during training, but not during testing.  
2. **Realistic Inference**: At inference time, batch norm uses learned means/variances from training data rather than real-time mini-batch stats.  
3. **Compute Efficiency**: We can disable gradient tracking to save memory and processing time when simply evaluating the model.

> **Important**: If you do not switch to eval mode for dropout or batch norm layers, **your inference** could yield **inconsistent** or **incorrect** predictions compared to the intended design.

---

## 4. PyTorch Functions: train(), eval(), no_grad()

1. **`model.train()`**  
   - Puts the entire model in **training mode**.  
   - Ensures dropout is active, batch normalization uses batch statistics, etc.  
   - This is **the default** state for newly constructed `nn.Module` objects.

2. **`model.eval()`**  
   - Puts the entire model in **evaluation (inference) mode**.  
   - Dropout is deactivated, batch norm uses moving averages (accumulated during training).  
   - **Crucial** for correct dev/test performance when using certain regularizations.

3. **`with torch.no_grad():`**  
   - A **context manager** that deactivates gradient computations for all statements within its scope.  
   - Greatly reduces overhead when you do **not** need to compute gradients (e.g., test predictions).  
   - Use it whenever you **only** need forward passes (in dev or test phases).

### Example of Usage
```python
```python
# net is your model (e.g., nn.Module)

net.train()       # sets the model to training mode
# training code with gradient updates ...

net.eval()        # sets the model to evaluation mode
with torch.no_grad():
    # code here does not track gradients
    predictions = net(test_data)  # forward pass only
    # no dropout, batch norm uses running stats
```

---

## 5. Illustrative Training Loop
Below is a **pseudo-code** snippet showing how to integrate `train()`, `eval()`, and `no_grad()` in a typical **epoch-based** training loop.

```python
```python
num_epochs = 100

for epoch in range(num_epochs):
    ### 1) TRAINING PHASE ###
    net.train()  # activate dropout, batch norm uses batch stats
    
    for X_batch, y_batch in train_loader:
        # forward pass with gradient tracking
        optimizer.zero_grad()
        y_pred = net(X_batch)
        loss   = loss_fn(y_pred, y_batch)
        # backprop
        loss.backward()
        optimizer.step()
    
    ### 2) EVALUATION PHASE ###
    net.eval()   # deactivate dropout, batch norm uses running averages
    
    with torch.no_grad():  # no gradient computations
        # entire dev/test set forward pass
        dev_preds = net(dev_features)
        dev_loss  = loss_fn(dev_preds, dev_labels)
        
        # if you have a test set, similarly:
        # test_preds = net(test_features)
        # test_loss  = loss_fn(test_preds, test_labels)
        
    # optionally, print epoch metrics or track them
    print(f"Epoch {epoch+1}/{num_epochs}, Dev Loss: {dev_loss.item():.4f}")
    
    # loop ends, next epoch continues
```

### Key Observations
- We **always** re-enable training mode (`net.train()`) before processing training batches.  
- We **switch to** `net.eval()` plus `torch.no_grad()` for dev/test data once each epoch.  
- This is especially **crucial** if using **dropout** or **batch normalization**.

---

## 6. Practical Considerations
1. **Computational Efficiency**:
   - For small models, overhead of grad computations may be negligible.  
   - For **large** CNNs, transformers, or big MLPs, `with torch.no_grad()` can **significantly** reduce memory usage and speed up dev/test evaluation.

2. **Batch Normalization**:
   - Uses different **running mean/variance** in `eval()` vs. the actual mini-batch stats in `train()`.  
   - If you accidentally remain in `train()` mode during dev, your dev results can vary wildly from epoch to epoch.

3. **Dropout**:
   - If you **forget** to call `net.eval()`, dropout will keep happening at inference, causing **inconsistent** test results.

4. **Default Behavior**:
   - A newly initialized model is by default in `train()` mode. You only **explicitly** need `net.train()` if you previously set `net.eval()`.

5. **Fine-Tuning**:
   - Sometimes you might want partial freeze or custom toggling for certain layers. In that case, you can manually override some layers’ training/eval state if needed (advanced use case).

---

## 7. Key Takeaways
1. **Why Toggle**:  
   - Avoid computing gradients (speed, memory) and ensure correct dropout/batch norm behavior during dev/test.  
2. **How**:  
   - `model.train()`: training mode.  
   - `model.eval()`: inference mode (dropout + BN changed).  
   - `torch.no_grad()`: suppress gradient tracking.  
3. **When**:  
   - **Train**: before processing training data each epoch.  
   - **Eval**: before generating dev/test predictions.  
   - `torch.no_grad()`: around code blocks that only do forward passes.

**Next Steps**:  
- The upcoming lectures on **dropout** regularization will illustrate why you must call `net.eval()` for correct dev/test performance.  
- Always remember to switch back to **train** mode once you move on to the next training epoch.

---

**End of Notes**  
These detailed notes should help you systematically manage **train** vs. **eval** modes in PyTorch. Proper usage ensures correct behavior of **regularization layers** (e.g., dropout, batch norm) and **saves** computational resources when generating test or dev set predictions.