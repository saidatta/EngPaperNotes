
alias: [Hyperparameters, Meta Parameters, Parameters vs. Meta]
tags: [Deep Learning, Lecture Notes, Hyperparameters, Neural Networks]

In deep learning, **parameters** are the values that the network learns automatically through the training process (e.g., weights and biases), while **meta parameters** (sometimes called **hyperparameters**) are the values that **you**, as the model designer, choose before or during training. This note clarifies the difference, explores common meta parameters, and shows why handling meta parameters is one of the most challenging (yet enjoyable!) aspects of deep learning.

---
## 1. Overview

1. **Parameters**: These are quantities *learned* by the network during training (e.g., weights and biases).
2. **Meta Parameters**: These are quantities that *you specify* and are *not* automatically learned (e.g., number of layers, learning rate, batch size, etc.).

**Key Takeaway**:  
- **Parameters** are internal to the model and adjusted by gradient descent.  
- **Meta parameters** are external design choices that shape *how* the model learns.

---

## 2. Distinction Between Parameters and Meta Parameters

### 2.1 Parameters

- **Definition**: Values adjusted by the optimization (learning) process.  
  - Examples:  
    - **Weights** (\(W\)) between neurons  
    - **Biases** (\(b\)) added to neuron outputs  

During backpropagation, **parameters** are updated automatically using gradients:
\[
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
\]
Where \(\eta\) is the learning rate, \(\mathcal{L}\) is the loss function, and the partial derivatives \(\frac{\partial \mathcal{L}}{\partial W}\) and \(\frac{\partial \mathcal{L}}{\partial b}\) are obtained from backpropagation.

### 2.2 Meta Parameters

- **Definition**: Model features that *you* set. They are not automatically learned by the network, though in some advanced techniques (e.g., AutoML), meta parameters can also be tuned algorithmically.
- **Examples**:
  1. **Model Architecture**  
     - Number of hidden layers  
     - Number of units (or neurons) per layer  
     - Activation functions per layer  
  2. **Training Configuration**  
     - Learning rate (\(\eta\))  
     - Optimization function (e.g., SGD, Adam, RMSProp)  
     - Batch size  
     - Number of epochs  
  3. **Regularization / Normalization**  
     - Dropout rates  
     - Weight decay (L2 regularization factor)  
     - Batch normalization or layer normalization  
  4. **Data Preprocessing**  
     - Data normalization  
     - Data augmentation strategies  
  5. **Cross Validation Strategy**  
     - Train-validation split sizes  
     - K-fold parameters  
  6. **Initialization Schemes**  
     - He initialization  
     - Xavier/Glorot initialization  

**Key Insight**: Since there are many such meta parameters, we cannot possibly explore *all* combinations. This is what makes designing deep learning models challenging—but also creative and fun.

---

## 3. Why Meta Parameters Are Important

1. **They Affect Model Performance**  
   - The wrong learning rate can prevent convergence or cause divergence.  
   - Poor architecture choice might underfit or overfit.  

2. **They Control Complexity**  
   - More layers and larger layer sizes \(\rightarrow\) higher model complexity.  
   - Regularization meta parameters (e.g., dropout) reduce overfitting.

3. **They Influence Training Speed**  
   - Optimizer choice and batch size can drastically change how quickly models learn.

4. **They Are Key to the Model's Generalization**  
   - Validation set size and early stopping policies determine how well the network generalizes.

Despite their importance, **there is no guaranteed way to know the best meta parameters** upfront. We often rely on domain knowledge, best practices, and empirical exploration.

---

## 4. Common Meta Parameters in Practice

Below is a **non-exhaustive** list of meta parameters commonly used and what they control:

| Meta Parameter            | Typical Values / Options              | Purpose                                                                       |
|---------------------------|---------------------------------------|-------------------------------------------------------------------------------|
| **Number of Layers**      | 1 to 100+ (depending on network type) | More layers can capture more complex patterns but are harder to train.        |
| **Units per Layer**       | 8,16,32,...1024 (powers of 2 often)   | Controls capacity of each layer. More units can increase representation power.|
| **Activation Function**   | ReLU, Leaky ReLU, Sigmoid, Tanh, etc. | Shapes how neurons learn complex functions.                                   |
| **Learning Rate** (\(\eta\))       | 1e-4, 1e-3, 1e-2, etc.            | Controls step size in gradient updates.                                       |
| **Optimizer**             | SGD, Adam, RMSProp, etc.              | Each has different characteristics in terms of speed & convergence.           |
| **Batch Size**            | 16, 32, 64, 128...                    | The number of samples processed before parameter update.                      |
| **Number of Epochs**      | 10, 50, 100, ...                      | How many times the model sees the entire dataset.                             |
| **Regularization**        | L1/L2, Dropout rate, Weight decay     | Prevents overfitting by penalizing complexity.                                |
| **Initialization**        | Xavier, He, Zeros, etc.               | Determines the starting point of parameter values.                            |
| **Data Normalization**    | Standardize (mean=0, std=1), MinMax   | Improves training stability and convergence.                                  |
| **Cross Validation Size** | e.g., 10% hold-out, k-fold cross-val  | Monitors overfitting and tunes model selection.                               |

---

## 5. Illustrative Example in Code (PyTorch)

Below is a minimal PyTorch example showing how you might set meta parameters and watch how your network parameters get learned automatically.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# 1. Set meta parameters
# -------------------------
num_layers = 2                # Number of hidden layers
hidden_units = 64            # Units per hidden layer
activation_function = nn.ReLU # Activation function
learning_rate = 0.001        # Learning rate
batch_size = 32              # Batch size
num_epochs = 10              # Number of epochs
dropout_rate = 0.5           # Dropout probability

# -------------------------
# 2. Create a sample dataset
# -------------------------
# Let's just make a random dataset for demonstration
input_dim = 20
output_dim = 2
X = torch.randn(1000, input_dim)
y = torch.randint(0, output_dim, (1000,))

# Create a DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# 3. Define the model architecture
# -------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, output_dim, dropout_rate=0.0, activation_fn=nn.ReLU):
        super(SimpleNN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Create hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_units))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_units
        
        # Final output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = SimpleNN(
    input_dim=input_dim,
    hidden_units=hidden_units,
    num_layers=num_layers,
    output_dim=output_dim,
    dropout_rate=dropout_rate,
    activation_fn=activation_function
)

# -------------------------
# 4. Define loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()       # Another meta parameter choice
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# 5. Train the model
# -------------------------
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# The trained parameters (weights, biases) have been updated automatically
# while we manually specified the meta parameters above.
```

**Notes**:
- **Parameters**:  
  - `model.parameters()` → includes all learnable weights and biases in the network.  
- **Meta Parameters**:  
  - `num_layers`, `hidden_units`, `learning_rate`, `dropout_rate`, etc.  

---

## 6. Visualizing Parameters vs. Meta Parameters

Here’s a conceptual diagram:

```
          ┌──────────────────────────────────┐
          │        Deep Neural Network      │
          │        (Learned Parameters)     │
          │    W1, b1, W2, b2, ... , Wn, bn │
          └──────────────────────────────────┘
                           ^
                           |
             ┌─────────────────────────────────┐
             │     Meta Parameters (User)     │
             │ - # Hidden Layers             │
             │ - # Units per Layer           │
             │ - Learning Rate               │
             │ - Optimizer                   │
             │ - Batch Size                  │
             │ ...                           │
             └─────────────────────────────────┘
```

- The **upper box** represents everything the network learns automatically (`W1, b1, W2, b2, …`).
- The **lower box** includes the meta parameters that you must choose to guide the learning process.

---

## 7. Challenges and Strategies

1. **Combinatorial Explosion**  
   - Each meta parameter can take many possible values → The search space grows exponentially.
2. **Empirical Tuning**  
   - Use best practices, domain knowledge, or published research to guess a good range of values.
3. **Systematic Search Methods**  
   - **Grid Search**: Exhaustive, but quickly becomes expensive.  
   - **Random Search**: Often a more efficient alternative.  
   - **Bayesian Optimization** (e.g., Hyperopt, Optuna): More sophisticated, tries to learn from past trials.
4. **Iterative Refinement**  
   - Start with a known good baseline (e.g., from literature or common default values).  
   - Run experiments to refine each meta parameter step by step.

---

## 8. Practical Advice

- **Start Simple**: Don’t jump to huge architectures or exotic optimizers right away.  
- **Log Everything**: Keep detailed notes on which meta parameters you tried and their results (e.g., using tools like TensorBoard, Weights & Biases, or a simple spreadsheet).  
- **Use Validation**: Always monitor validation performance and possibly test multiple splits or folds.  
- **Learn from Others**: Look at open-source implementations, papers, or code examples.  

Remember: **No single configuration will always work best**. Success often comes from systematic experimentation and accumulated intuition.

---

## 9. Conclusion

- **Parameters** are the internal, learned values (weights, biases).  
- **Meta Parameters (Hyperparameters)** are the *design choices* you make for your model’s architecture, training process, and data handling.
- The large number of meta parameters is **both a challenge** (because you cannot explore them all) and **an opportunity** (because you can creatively tailor models to suit your problem).

> **Quote from the Lecture**:  
> “What really makes deep learning hard is all of the meta parameters. It does make deep learning time-consuming and complicated, but it’s also part of the fun and empirical discovery.”

Embrace the iterative process of tuning meta parameters. Over time, you’ll develop a strong intuition for which configurations are likely to work well for a given task.

---

## 10. Further Reading & References

- **Goodfellow, Bengio, Courville**, *Deep Learning* (MIT Press) – Chapter on hyperparameter tuning.  
- **Learning Rate Schedules**: Research cyclical learning rates, warm restarts, etc.  
- **Regularization Methods**: Dive into dropout, batch normalization, and data augmentation strategies.  
- **Advanced Tuning**: Explore Bayesian optimization or evolutionary algorithms for automated hyperparameter search.

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: Mike’s “Meta Parameters: Concepts”  

```
Remember to link these notes with other relevant notes on Neural Network architectures, Loss functions, and Optimization in your Obsidian vault!
```
