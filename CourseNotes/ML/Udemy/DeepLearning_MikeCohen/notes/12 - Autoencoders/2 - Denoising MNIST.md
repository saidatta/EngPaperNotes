## 1. Overview and Motivation

- **Objective**: Use an autoencoder to reconstruct MNIST images from noisy inputs.
- **Key Concept**: 
  - Autoencoders learn to map \( \mathbf{x} \) (the original input) to \(\hat{\mathbf{x}}\) (the reconstruction) via a **bottleneck** (latent layer).
  - In denoising applications, we deliberately corrupt the input \(\mathbf{x}\) to get \(\mathbf{x}_{\text{noisy}}\) but still ask the network to reconstruct the **original clean** image.
- **Highlight**:  
  - Even a simple, fully-connected (feedforward) autoencoder can remove a decent amount of noise from images.
  - Convolutional autoencoders typically perform even better, but we focus on a basic feedforward version here.

---

## 2. Data and Preprocessing

### 2.1 MNIST Data
- Consists of \(28 \times 28\) grayscale images of handwritten digits, labeled 0–9.
- Each image can be flattened to a 784-dimensional vector for a simple feedforward network.

### 2.2 Normalizing MNIST
- Range the pixel values from \( [0,255] \) down to \([0,1]\) by dividing by 255.
  ```python
  # Suppose X is the MNIST data as a NumPy array
  X = X / 255.0  
  ```

### 2.3 (Optional) Data Loaders
- One can use PyTorch’s `DataLoader` to create mini-batches.
- **This example** demonstrates how we can skip the full DataLoader pipeline and directly sample mini-batches randomly during training.

---

## 3. Architecture of the Autoencoder

### 3.1 Network Layout

\[
\text{Input} \; (784) 
\;\longrightarrow\; 
\underbrace{250}_\text{encoder} 
\;\longrightarrow\; 
\underbrace{50}_\text{bottleneck} 
\;\longrightarrow\; 
\underbrace{250}_\text{decoder} 
\;\longrightarrow\;
\text{Output} \; (784)
\]

- **Input layer (784 units)**: one neuron per pixel of a flattened MNIST image.
- **Encoder layers**:
  - First hidden layer: 250 units, activation ReLU.
  - Bottleneck (latent) layer: 50 units, activation ReLU.
- **Decoder layers**:
  - Mirror the encoder: 250 hidden units, activation ReLU.
  - Output layer: 784 units, **sigmoid** activation to produce values in \([0,1]\).

**Why Sigmoid at the Final Layer?**
- Because the original images are normalized between 0 and 1, using sigmoid ensures the reconstructions \(\hat{\mathbf{x}}\) also lie within \([0,1]\).

### 3.2 Loss Function

\[
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \bigl\| \mathbf{x}^{(n)} - \hat{\mathbf{x}}^{(n)} \bigr\|^2
\]
- In PyTorch, we can use `nn.MSELoss()` for this.
- Conceptually, the autoencoder’s goal is to **minimize reconstruction error** between input \( \mathbf{x} \) and output \(\hat{\mathbf{x}}\).

### 3.3 Comparison to Classification Models
- **Classification**: The network output is typically a predicted class label (or a probability distribution over classes).
- **Autoencoder**: The network output is the **reconstructed input** itself.

---

## 4. PyTorch Implementation

Below is a **minimal** PyTorch script illustrating how to build and train the autoencoder for denoising.

```python
```yaml
# In Obsidian, store code in a fenced code block:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```
```

### 4.1 Define the Autoencoder

```python
class AENet(nn.Module):
    def __init__(self):
        super(AENet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Linear(784, 250)
        self.enc2 = nn.Linear(250, 50)
        
        # Decoder
        self.dec1 = nn.Linear(50, 250)
        self.dec2 = nn.Linear(250, 784)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encode
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        
        # Decode
        x = self.relu(self.dec1(x))
        x = self.sigmoid(self.dec2(x))
        
        return x

def create_model():
    # Instantiate the autoencoder
    model = AENet()
    return model
```

### 4.2 Training Function

- **Note**: We are not using a DataLoader or explicit train/test split here.  
- **Mini-batch sampling**: Each epoch randomly selects a subset (e.g., 32 samples) from the dataset.

```python
def train_autoencoder(X, epochs=10000, batch_size=32, lr=0.001):
    """
    Trains the autoencoder with random mini-batches each epoch.
    
    Parameters:
    -----------
    X : torch.Tensor
        Tensor of shape (N, 784) representing the flattened MNIST images.
    epochs : int
        Number of training epochs.
    batch_size : int
        Size of randomly sampled mini-batch each epoch.
    lr : float
        Learning rate for the Adam optimizer.
        
    Returns:
    --------
    losses : list
        A list of loss values tracked over training epochs.
    model : nn.Module
        The trained PyTorch autoencoder model.
    """
    
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    N = X.shape[0]
    
    for ep in range(epochs):
        # Random mini-batch sampling
        idx = np.random.choice(N, batch_size, replace=False)
        x_batch = X[idx, :]  # shape: (batch_size, 784)
        
        # Forward pass
        x_hat = model(x_batch)
        
        # Compute loss: MSE between input and reconstruction
        loss = criterion(x_hat, x_batch)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    return losses, model
```

### 4.3 Quick Sanity Check (Untrained Model)

- **Goal**: Confirm the network’s output shape matches the input shape before we do a full training run.

```python
# Suppose X is a torch.Tensor of shape (N, 784) for some subset of MNIST data.
model_test = create_model()
x_test_batch = X[:5, :]     # First 5 images
with torch.no_grad():
    y_test_batch = model_test(x_test_batch)
print("Input shape:", x_test_batch.shape)
print("Output shape:", y_test_batch.shape)

# => Expect (5, 784) for both
```

- Visualizing the random output:
  ```python
  # Convert to numpy for plotting
  fig, axs = plt.subplots(2, 5, figsize=(10,4))

  for i in range(5):
      # Original
      axs[0, i].imshow(x_test_batch[i].reshape(28,28), cmap='gray')
      axs[0, i].set_title("Original")
      axs[0, i].axis('off')
      
      # Untrained Output
      axs[1, i].imshow(y_test_batch[i].reshape(28,28), cmap='gray')
      axs[1, i].set_title("Untrained")
      axs[1, i].axis('off')

  plt.tight_layout()
  plt.show()
  ```
- **Result**: Random noise or near-random patterns in the bottom row, as the model is untrained.

---

## 5. Training and Results

### 5.1 Run the Training

```python
X_tensor = torch.tensor(X, dtype=torch.float32)  # Where X is your normalized MNIST data
losses, autoenc_model = train_autoencoder(X_tensor, epochs=10000, batch_size=32, lr=0.001)
```

- **Note on Epochs**: 10,000 might seem large, but each epoch uses only **one** mini-batch of size 32 (randomly sampled). In classification, we typically run fewer epochs but loop over all mini-batches per epoch.

### 5.2 Loss Curve

```python
plt.figure(figsize=(8,5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()
```

- Expect the loss to drop quickly at first and then gradually improve.

### 5.3 Reconstructing Clean Images

```python
# Let's see how the trained model reconstructs the first 5 images:
with torch.no_grad():
    y_hat_clean = autoenc_model(X_tensor[:5, :])

fig, axs = plt.subplots(2, 5, figsize=(10,4))
for i in range(5):
    # Original
    axs[0, i].imshow(X_tensor[i].reshape(28,28), cmap='gray')
    axs[0, i].set_title("Orig")
    axs[0, i].axis('off')
    
    # Reconstructed
    axs[1, i].imshow(y_hat_clean[i].reshape(28,28), cmap='gray')
    axs[1, i].set_title("Reconst")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
```

- **Observation**: The reconstructed images should look like “slightly fuzzier” versions of the original digits.

---

## 6. Denoising Experiment

### 6.1 Create Noisy Images

```python
# Let's pick 10 images from X_tensor
num_imgs = 10
noise_factor = 1/4.0

X_clean = X_tensor[:num_imgs, :]
# Add uniform noise [0,1], scaled down by 0.25
noise = torch.rand_like(X_clean) * noise_factor

X_noisy = X_clean + noise
# Clip values above 1.0
X_noisy = torch.clamp(X_noisy, 0.0, 1.0)
```

### 6.2 Push Noisy Inputs Through the Autoencoder

```python
with torch.no_grad():
    Y_hat_denoised = autoenc_model(X_noisy)
```

### 6.3 Visual Comparison

```python
fig, axs = plt.subplots(3, num_imgs, figsize=(2*num_imgs, 6))

for i in range(num_imgs):
    # Original Clean
    axs[0, i].imshow(X_clean[i].reshape(28,28), cmap='gray')
    axs[0, i].set_title("Clean")
    axs[0, i].axis('off')
    
    # Noisy
    axs[1, i].imshow(X_noisy[i].reshape(28,28), cmap='gray')
    axs[1, i].set_title("Noisy")
    axs[1, i].axis('off')
    
    # Denoised
    axs[2, i].imshow(Y_hat_denoised[i].reshape(28,28), cmap='gray')
    axs[2, i].set_title("Denoised")
    axs[2, i].axis('off')

plt.tight_layout()
plt.show()
```

**Analysis**:
- The denoised outputs typically **remove much of the background noise**.
- Some digits become slightly thicker or fuzzier. The model may also introduce small distortions (e.g., incorrectly smoothing edges).

---

## 7. Observations and Discussion

1. **Surprisingly Simple**  
   - The autoencoder code is shorter than many classification examples: no explicit labels, no accuracy metric, fewer lines overall.
2. **Latent Space**  
   - The network compresses 784 inputs to 50. Despite this, reconstructions can be quite faithful, suggesting the model captures essential digit features.
3. **Noise Removal**  
   - Even a basic feedforward autoencoder can remove moderate uniform noise effectively. With more data or a deeper architecture, results might improve.
4. **Potential Improvements**  
   - **Use the full 60K–70K MNIST set** and iterate over all samples each epoch for better training.  
   - **Convolutional Autoencoder**: Typically outperforms fully-connected layers for image tasks.  
   - **Change the noise model**: Try Gaussian noise, salt-and-pepper noise, or heavier uniform noise.  
   - **Experiment with different losses** (e.g., `nn.BCELoss` if your data is in [0,1]) or add regularization.

---

## 8. Additional Explorations

1. **Adjust Noise Strength**  
   - Increase or decrease `noise_factor` to see how robust the trained model is.
2. **Change Loss Function**  
   - Compare `nn.MSELoss()` vs. `nn.BCELoss()` for reconstruction.
3. **Vary Latent Dimensionality**  
   - If bottleneck = 2 or 10 or 100, how do reconstructions change?
4. **Training Regime**  
   - Use PyTorch DataLoaders to do a proper train/test split. Observe difference in reconstruction quality on unseen data.
5. **Reduce Epochs**  
   - Inspect how quickly the model converges. Plot partial results vs. number of epochs.

---

## 9. Summary

- **Concept**: A feedforward autoencoder can be trained to remove noise from MNIST images by reconstructing the original clean image.
- **Implementation**: Very similar to standard PyTorch feedforward networks—only difference is:
  1. Output dimension = Input dimension.
  2. Loss compares \(\hat{\mathbf{x}}\) to \(\mathbf{x}\).
  3. No explicit “label” vectors. 
- **Performance**: The denoising capability can be observed even with a small 50-dimensional latent space. 
- **Implication**: Autoencoders are powerful for data compression, feature extraction, and data cleaning tasks in deep learning.

---  

**End of Notes**