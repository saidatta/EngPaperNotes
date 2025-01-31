Below is a set of **Obsidian-style**, highly detailed notes aimed at a PhD-level engineer for the lecture:

**“Autoencoders: CodeChallenge – How many units?”** 

These notes capture the reasoning, code, and visual outputs for a parametric experiment exploring different numbers of encoder/decoder units and bottleneck (latent) units in a feedforward autoencoder. The dataset used is EMNIST (Extended MNIST), although the ideas easily extend to other datasets.

## 1. Motivation & Overview

1. **Goal**: Investigate the impact of autoencoder size on reconstruction performance.
   - Vary the number of **encoder/decoder units** in a hidden layer (e.g., from 10 to 500).
   - Vary the number of **bottleneck (latent) units** (e.g., from 5 to 100).
2. **Dataset**: EMNIST (Extended MNIST).
   - Typically 28×28 grayscale images of letters/numbers.
   - For simplicity, we flatten each 28×28 image to a 784-dimensional vector.
3. **Primary Metric**: Mean Squared Error (MSE) over reconstruction. We track the **average of the final three losses** in training to gauge each network’s performance.
4. **Key Trade-Off**:
   - More (and larger) hidden layers → better reconstruction but heavier computational costs and less compression.
   - Fewer units → faster, cheaper, but potentially higher loss (poorer reconstructions).

---

## 2. Problem Statement

### 2.1 Parametric Sweeps

We perform a grid search over two main hyperparameters:

1. **Encoder/Decoder Units** (\(n_\text{enc}\)): from 10 to 500 in 12 linearly spaced steps.
2. **Bottleneck Units** (\(n_\text{bottle}\)): from 5 to 100 in 8 linearly spaced steps.

This yields \(12 \times 8 = 96\) unique autoencoder configurations.

### 2.2 Training Protocol

- **Data**: 
  - 20,000 images (subset of EMNIST).
  - Each image scaled to \([0,1]\).
  - We do **mini-batch** training in a custom loop (without PyTorch’s `DataLoader`), ensuring we see each sample exactly once per epoch.
- **Epochs**:  
  - For each configuration, we train for a small number of epochs (e.g., 3).
  - Each epoch iterates over all 20,000 images in mini-batches (e.g., 32 samples per batch).
- **Loss**: MSE between input image \( \mathbf{x} \) and reconstructed output \(\hat{\mathbf{x}}\).
- **Status Logging** (Optional Challenge):
  - Print a single-line progress report using Python’s `sys.stdout.write` to minimize clutter in the notebook output.

---

## 3. Implementation Outline

1. **Data Loading & Normalization**:
   - Load EMNIST data (20k samples).
   - Reshape to \((N,784)\) and normalize to \([0,1]\).
2. **Model Definition**:
   - **Input dimension**: 784
   - **Hidden dimension** (encoder/decoder): \(n_\text{enc}\) (varies).
   - **Latent dimension**: \(n_\text{bottle}\) (varies).
3. **Training Function**: 
   - Accepts \((n_\text{enc}, n_\text{bottle})\) and trains the model for a fixed number of epochs.
   - Inside each epoch:
     - Shuffle indices.
     - Create mini-batches.
     - Compute forward pass & MSE loss.
     - Perform backpropagation.
   - Return the **final few epochs’ average loss** or store the full training curve.
4. **Parametric Loop**:
   - Double `for` loop over all combinations of \(n_\text{enc}\) and \(n_\text{bottle}\).
   - Track the final performance in a 2D results matrix.
5. **Visualization**:
   - Generate a color map (heatmap) where rows = \(n_\text{enc}\), columns = \(n_\text{bottle}\).
   - Optionally, plot line graphs to see how loss changes with increasing units.

---

## 4. Code Snippets

Below is a representative skeleton in PyTorch. Adapt as needed.

```yaml
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
```
```

### 4.1 Data Loading & Normalization

```yaml
```python
# Suppose you have loaded X_emnist as a NumPy array of shape (N, 28, 28)
# We'll just call it X_emnist for demonstration.

N = 20000  # number of samples to use
X = X_emnist[:N].reshape(N, 784).astype(np.float32) / 255.0

X_tensor = torch.tensor(X)  # shape: (N, 784)
```
```

### 4.2 Defining the Autoencoder

```yaml
```python
class AENet(nn.Module):
    def __init__(self, n_enc=250, n_bottle=50):
        super(AENet, self).__init__()
        
        # Layers
        self.enc1 = nn.Linear(784, n_enc)
        self.enc2 = nn.Linear(n_enc, n_bottle)
        self.dec1 = nn.Linear(n_bottle, n_enc)
        self.dec2 = nn.Linear(n_enc, 784)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        # Decoder
        x = self.relu(self.dec1(x))
        x = self.sigmoid(self.dec2(x))
        return x

def create_model(n_enc, n_bottle):
    return AENet(n_enc, n_bottle)
```
```

### 4.3 Training Function

- **Note**: We do our own mini-batch loop (no `DataLoader`).

```yaml
```python
def train_autoencoder(
    X, n_enc, n_bottle, 
    epochs=3, batch_size=32, 
    lr=0.001
):
    """
    Train an autoencoder with specified hidden (n_enc) and 
    bottleneck (n_bottle) dimensions.
    
    Returns:
    - final_losses: average of the last few (e.g., 3) epoch losses
    - model: trained AENet instance
    """
    
    model = create_model(n_enc, n_bottle)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    N = X.shape[0]
    n_batches = N // batch_size
    
    losses_each_epoch = []
    
    for ep in range(epochs):
        # Shuffle indices for the entire dataset
        idx_perm = np.random.permutation(N)
        
        epoch_loss = 0.0
        
        for b_i in range(n_batches):
            # Batching
            start = b_i * batch_size
            end = start + batch_size
            batch_indices = idx_perm[start:end]
            
            x_batch = X[batch_indices]
            
            # Forward
            x_hat = model(x_batch)
            loss = criterion(x_hat, x_batch)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / n_batches
        losses_each_epoch.append(avg_epoch_loss)
    
    # Return the average of the final 3 epochs
    final_losses = np.mean(losses_each_epoch[-3:])
    return final_losses, model
```
```

### 4.4 Running the Experiment

```yaml
```python
# Define sweeps
enc_units_list = np.linspace(10, 500, 12, dtype=int)      # 12 linearly spaced steps
bottle_units_list = np.linspace(5, 100, 8, dtype=int)     # 8 linearly spaced steps

results = np.zeros((len(enc_units_list), len(bottle_units_list)))

total_expts = len(enc_units_list) * len(bottle_units_list)
exp_count = 0

for i, n_enc in enumerate(enc_units_list):
    for j, n_bottle in enumerate(bottle_units_list):
        exp_count += 1
        
        # Train model
        final_loss, _ = train_autoencoder(
            X_tensor, 
            n_enc=n_enc, 
            n_bottle=n_bottle, 
            epochs=3,
            batch_size=32
        )
        
        results[i, j] = final_loss
        
        # Single-line status update:
        message = f"Finished experiment {exp_count} of {total_expts}\r"
        sys.stdout.write(message)
        sys.stdout.flush()

print("\nDone with all experiments!")
```
```

**Explanation**:
1. **enc_units_list**: 12 distinct values from 10 to 500.  
2. **bottle_units_list**: 8 distinct values from 5 to 100.  
3. **results**: 2D array storing final losses.  
4. **Progress Logging**:  
   - `\r` carriage return moves the cursor back to the start of the line.  
   - `sys.stdout.flush()` ensures the output is updated immediately.

---

## 5. Results & Visualization

### 5.1 Heatmap

```yaml
```python
plt.figure(figsize=(8,6))
plt.imshow(results, cmap='hot', aspect='auto', 
           origin='lower',
           extent=(bottle_units_list[0], 
                   bottle_units_list[-1],
                   enc_units_list[0],
                   enc_units_list[-1]))
plt.colorbar(label='Final Loss')
plt.xlabel("Bottleneck (n_bottle)")
plt.ylabel("Encoder/Decoder Units (n_enc)")
plt.title("Parametric AE Performance (EMNIST)")

plt.show()
```
```

- **Interpretation**:
  - Lighter colors = **lower** final loss (better reconstruction).
  - Darker = **higher** loss.
  - Typically, top-left corner (small \(n_\text{enc}, n_\text{bottle}\)) has high loss.
  - Bottom-right corner (large \(n_\text{enc}, n_\text{bottle}\)) has low loss (but more parameters, more cost).

### 5.2 Line Plots (Optional)

You can also plot each line separately for a given bottleneck size:

```yaml
```python
plt.figure(figsize=(9,6))

for col_idx, nb in enumerate(bottle_units_list):
    plt.plot(enc_units_list, results[:, col_idx], label=f'{nb} Bottleneck')

plt.xlabel('Encoder/Decoder Units')
plt.ylabel('Final Loss')
plt.title('Loss vs. Encoder Size for Various Bottleneck Sizes')
plt.legend()
plt.show()
```
```

- Each curve corresponds to a different \(n_\text{bottle}\), showing how adding encoder/decoder units gradually lowers the error.

---

## 6. Observations & Discussion

1. **General Trend**: As \(n_\text{enc}\) and \(n_\text{bottle}\) increase, the reconstruction error decreases. This is unsurprising—more capacity means less compression or more "room" to encode features.
2. **Trade-off**: 
   - If you want strong data compression, you choose fewer bottleneck units, but you pay the price in higher MSE.  
   - If your primary goal is reconstruction fidelity, you pick larger \(\text{bottle}\) dimension or larger hidden layers.
3. **Computation Time**: 
   - With 96 total models, some configurations can become very large (500 → 784 fully-connected layers). Each pass can be slower.  
   - Reducing epochs or mini-batch size can mitigate run time but might degrade training accuracy.

---

## 7. Additional Considerations

1. **CNN Autoencoders**:
   - For image data, convolutional autoencoders typically provide superior performance with fewer parameters. 
   - The dimensionality constraints become more intuitive (channels & spatial dimensions).
2. **Loss Function Variations**:
   - MSE vs. BCE (`nn.BCELoss`) or other robust reconstruction losses might alter the final performance landscape.
3. **Epoch Settings**:
   - We used 3 epochs over the entire dataset. If each epoch is a full pass (with mini-batching), total training steps are still moderate.
   - The 10,000-epoch approach from simpler demos is not directly comparable; often those “epochs” were effectively single mini-batch updates.
4. **Real-World Considerations**:
   - If the dataset were larger or more complex (e.g., color images), you’d likely want more flexible architectures and fewer brute-force sweeps.

---

## 8. Summary

- **Key Takeaway**: There’s no single “correct” number of units in the encoder/decoder or bottleneck; it depends on:
  1. Desired compression.
  2. Desired reconstruction quality.
  3. Available compute/time.
- **Experiment**: We systematically measured final MSE across 96 configurations. As expected, networks with **more** hidden and latent units reconstruct better.
- **Practical Skills**:
  - Creating mini-batches **manually** (without `DataLoader`).
  - Printing a **single-line** progress report with `sys.stdout.write`.
  - Plotting a **heatmap** of final results for easy inspection.

**This exercise** helps highlight the fundamental cost-performance trade-off in autoencoder design. When you need higher fidelity for your application, aim for bigger networks. If your focus is maximum compression or feature extraction, you can tolerate fewer latent units and potentially reduced reconstruction quality.

---

**End of Notes**.