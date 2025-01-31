aliases: [FFN, FNN, FCN, Fully Connected Networks, Feedforward Networks]
tags: [Deep Learning, Feedforward, ANN, FullyConnected, MNIST]
## Overview
In this set of notes, we will dive deeper into **Feedforward Networks (FFNs)**, also commonly referred to as:
- **Fully Connected Networks (FCNs)**
- **Fully Feedforward Connected Networks (FFCNs)**
- **Artificial Neural Networks (ANNs)** (in the context where each layer is “fully connected” to the next)

We’ll clarify the terminology, discuss the general structure of these networks, and understand how feedforward networks serve as the foundation for many advanced deep learning architectures. We will also briefly introduce the famous **MNIST dataset** and motivate why it’s commonly used as a starting example in training neural networks.

---

## 1. Terminology & Clarifications

### 1.1 "Fully Connected Network" vs. "Truly Fully Connected"
- A **Fully Connected Network** in deep learning is a network where **every node (unit) in layer \(n\) connects to every node in layer \(n+1\)**.
  - Example: If layer \(n\) has 4 units and layer \(n+1\) has 3 units, then each of the 4 units projects to all 3 units, resulting in \(4 \times 3 = 12\) connections.

- However, from a strict graph-theory or neuroscience perspective, **"fully connected"** might imply that **every node is connected to every other node** in the entire network (including lateral and backward connections).  
  - This is **not** usually what we mean in deep learning.  
  - Hence, many prefer the term **"feedforward fully connected"** to be more precise.

### 1.2 "Feedforward Network" vs. Backpropagation
- The term **Feedforward** refers to how **activations** (signals) flow *forward* from **input → hidden layers → output**.
- Despite the name “feedforward,” these networks still rely on **backpropagation** for learning. However, backpropagation is part of the *training algorithm*, not the forward pass of the data flow.

### 1.3 Alternative Names
You might see these acronyms used interchangeably:
- **ANN:** Artificial Neural Network  
- **FNN:** Feedforward Neural Network  
- **FCN:** Fully Connected Network  
- **FFCN:** Fully Feedforward Connected Network  

Ultimately, they typically mean the same conceptual architecture (for standard MLPs). The distinctions usually come from context or personal naming preferences.

---

## 2. Why Focus on Feedforward Networks?

1. **Fundamental Building Block:**  
   FFNs are the **foundational** architecture for many deep learning models, such as:
   - **CNNs** (Convolutional Neural Networks)
   - **RNNs** (Recurrent Neural Networks)
   - **GANs** (Generative Adversarial Networks)
   - **VAEs** (Variational Autoencoders)
   - and more…

2. **Core Concepts:**  
   If you thoroughly understand FFNs:
   - How layers connect
   - How forward passes work
   - How backpropagation updates weights
   - Common activation functions, cost functions  
   Then it becomes easier to grasp more advanced (and specialized) architectures.

3. **Simplicity & Versatility:**  
   - FFNs can be applied to a variety of tasks like classification, regression, reconstruction, etc.  
   - A baseline FFN on simpler datasets (e.g., **MNIST**) is often the first step in learning deep learning.

---

## 3. Structure of a Basic Feedforward Network

### 3.1 Layer-by-Layer Connections
Consider a network with:
- Input layer: \(\mathbf{x}\) (dimensionality = \(d\))
- Hidden layers: \(\mathbf{h}^{(1)}, \mathbf{h}^{(2)}, \dots\)
- Output layer: \(\mathbf{y}\)

**Forward Pass** for a single hidden layer:
\[
\mathbf{z}^{(1)} = W^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \quad \Rightarrow \quad \mathbf{h}^{(1)} = \sigma(\mathbf{z}^{(1)})
\]
\[
\mathbf{z}^{(2)} = W^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)} \quad \Rightarrow \quad \mathbf{y} = \sigma(\mathbf{z}^{(2)})
\]

- \(W^{(l)}\): Weight matrix for layer \(l\).
- \(\mathbf{b}^{(l)}\): Bias vector for layer \(l\).
- \(\sigma(\cdot)\): Activation function (e.g., ReLU, sigmoid, tanh).

### 3.2 Activation Functions
Common choices:
- **Sigmoid** \(\sigma(x) = \frac{1}{1 + e^{-x}}\)
- **Tanh** \( \tanh(x) \)
- **ReLU** \( \mathrm{ReLU}(x) = \max(0, x) \)
- **LeakyReLU**, **ELU**, etc.

### 3.3 Backpropagation
- **Objective / Loss Function**: \(\mathcal{L}(\mathbf{y}, \mathbf{t})\), where \(\mathbf{t}\) is the target label.
- Compute gradients \(\frac{\partial \mathcal{L}}{\partial W^{(l)}}\) and \(\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}\).
- Update parameters iteratively (e.g., with Stochastic Gradient Descent or Adam).

---

## 4. The MNIST Dataset

### 4.1 Description
- **MNIST** stands for **Modified National Institute of Standards and Technology**.
- It contains \(28 \times 28\) grayscale images of handwritten digits (0 through 9).
- Each image is labeled with the correct digit (0–9).
- **Training set size**: 60,000 images  
- **Test set size**: 10,000 images  

### 4.2 Why MNIST?
- **Simplicity**: Images are small and relatively easy to classify.
- **Benchmark**: Commonly used for quick experiments and proof-of-concept demos.
- **Fast Training**: Even basic networks can achieve high accuracy (>90%) with minimal tuning.

---

## 5. Example: Implementing a Simple FFN on MNIST (PyTorch)

Below is a comprehensive example of how to implement and train a simple feedforward network (2-layer MLP) on MNIST using **PyTorch**. The code includes comments and step-by-step explanations.

```python
# %% [markdown]
# ## Simple FFN on MNIST - PyTorch Example

# %% [code]
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 2. Data Loading & Transformations
# Download MNIST dataset and create DataLoader objects
transform = transforms.Compose([
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean & std of MNIST
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 3. Define the FFN model
class SimpleFFN(nn.Module):
    def __init__(self):
        super(SimpleFFN, self).__init__()
        # Fully connected layers:
        self.fc1 = nn.Linear(28*28, 128)  # from 784 inputs to 128 hidden units
        self.fc2 = nn.Linear(128, 10)     # from 128 hidden units to 10 outputs (digits)
        
        # We can use ReLU for the hidden layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        # Flatten the image into a single vector of size 784
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleFFN()

# 4. Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Training Loop
for epoch in range(num_epochs):
    model.train()  # set model to training mode
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 3. Update
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

# 6. Evaluation
model.eval()  # set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # turn off gradient calculations
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy of the model on the 10,000 test images: {accuracy:.2f}%")
```

### 5.1 Explanation
1. **DataLoader** prepares batches of MNIST images and labels.
2. **SimpleFFN** class has:
   - `fc1` for the hidden layer (784 → 128)
   - `fc2` for the output layer (128 → 10)
   - `ReLU` activation after `fc1`
3. We use **CrossEntropyLoss** for multi-class classification.
4. **Adam optimizer** updates the weights.
5. We **train** for a few epochs, printing out the loss periodically.
6. Finally, we measure **test accuracy**.

Typically, even this simple network can reach \( \sim 97\% \) accuracy on MNIST with minimal tuning.

---

## 6. Visualizations

### 6.1 Basic Network Diagram
Here’s a simple ASCII diagram of our 2-layer FFN for MNIST:

```
   Input Layer (784)    ->   Hidden Layer (128)    ->   Output Layer (10)
[ x1 x2 x3 ... x784 ]        [ h1 ... h128 ]                [ y1 ... y10 ]
           |                         |                          |
        W^(1), b^(1)              W^(2), b^(2)              (predicted digit)
```

- Each arrow set represents a matrix multiplication plus bias, followed by a non-linear activation function (ReLU for the hidden layer).

### 6.2 Confusion Matrix
To better understand performance, you could compute and visualize a **confusion matrix** for the final predictions vs. true labels. For MNIST, it might look like this:

|        | Pred 0 | Pred 1 | Pred 2 | ... | Pred 9 |
|--------|--------|--------|--------|-----|--------|
| True 0 |   980  |    0   |   0    | ... |    1   |
| True 1 |   0    |   1130 |   2    | ... |    0   |
| True 2 |   1    |    5   |  1002  | ... |    2   |
| ...    |   ...  |  ...   |   ...  | ... |   ...  |
| True 9 |   0    |    1   |   1    | ... |   992  |

(Exact numbers depend on your training result.)

---

## 7. Key Takeaways

1. **Terminology Pitfalls**: “Fully Connected” in deep learning typically means *layer-to-layer* connections rather than literal full connectivity among all neurons in the network.
2. **Feedforward Flow**: Signals move forward, but weight updates happen via backpropagation.
3. **Foundational Architecture**: Mastering FFNs builds intuition for advanced models (CNNs, RNNs, GANs, VAEs, etc.).
4. **MNIST as a Playground**: It’s small, standard, and easy to get started with. Great for testing basic neural network ideas.

---

## 8. Further Reading & References

- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. \[[Link](https://www.deeplearningbook.org)\]
- Official [**PyTorch Tutorials**](https://pytorch.org/tutorials/) for more detailed examples.
- [**MNIST Dataset**](http://yann.lecun.com/exdb/mnist/) by Yann LeCun.

---

## 9. Next Steps
In the upcoming sections, we will:
- Explore more **architectural optimizations** and regularization techniques (dropout, batch normalization, etc.).
- Delve into **convolutional neural networks (CNNs)** and compare them with FFNs.
- Expand on advanced topics like RNNs, GANs, and VAEs—each of which can be viewed as a feedforward network with special constraints or innovations.

> **Recommended Note Linking**:  
> - See [[Activation Functions]] for deeper insights into activation types.  
> - Refer to [[Backpropagation-Details]] for more mathematical depth on computing gradients.  
> - Check [[AdvancedOptimizers]] for details on Adam, RMSProp, etc.

---

**End of Notes: “Feedforward Networks (FFNs) – Concepts”** 
```