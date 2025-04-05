## Simple Regression
Regression is a statistical procedure used for predicting data values. In this context, we'll focus on a simple regression where we predict one continuous variable from another.

The formula for a simple regression is: 

```math
y_i = β_0 + β_1*x_i + ε_i
```

In this formula, `y_i` is the dependent variable we are trying to predict. `x_i` is the independent variable we are using to make the prediction. `β_0` is the y-intercept, and `β_1` is the slope of the line (the weight or scale for `x_i`). `ε_i` is the error or residual term, representing the difference between the values we predict (`y_i`) and the values we observe in real data.

This is similar to the equation of a line (`y = mx + b`) where `m` is the slope, `b` is the y-intercept, and `y` is the dependent variable.
![[Pasted image 20250203075619.png]]
## Mapping to Perceptron
The equation for a perceptron can be mapped to the simple regression equation:
```math
y = b + Σ(wx)
```

Here, `y` is the output of the perceptron, `b` is the bias, `w` is the weight, and `x` is the input. The summation (`Σ`) represents a weighted sum of all inputs.

The bias `b` in the perceptron corresponds to the intercept `β_0` in the simple regression. The weight `w` corresponds to the slope `β_1`.

The activation function `σ` in the perceptron does not have an equivalent in the simple regression equation, because it is a non-linear function and the simple regression is a purely linear model.

The error or residual term `ε` in the simple regression is not explicitly mentioned in the perceptron equation, but it is implicitly present and is used in the computation of the loss function.

## ANN for Regression: Python Implementation with PyTorch

We will implement a regression-like model in PyTorch. Our model will be a bit more sophisticated than a simple regression because it will include a non-linearity.

Our goal is to predict the y values (dependent variable) based on the x values (independent variable). We will generate a dataset that has a strong positive correlation between the x and y values, but also includes some noise.

The architecture of our model will look something like this:

```
X --(W1)--> σ --(W2)--> ŷ
B1 ------> /   B2 ------> /
```

Here, `X` is the input, `W1` and `W2` are the weights, `B1` and `B2` are the biases, `σ` is the activation function (introducing non-linearity), and `ŷ` is the predicted output. 

We will use the Mean Squared Error (MSE) as our loss function, which is suitable for continuous numerical predictions. The loss for each data point will be the squared difference between the predicted (`ŷ`) and actual (`y`) values.

```math
MSE = Σ(y - ŷ)^2
```

The weights of the model will be trained using gradient descent.

### Real-life Example

We'll create a dataset where the x values represent the hours studied by a student and the y values represent the test scores of the student. The relationship between the hours studied and the test scores is positive but not perfect due to other factors (noise) such as the student's inherent aptitude, quality of study material, etc.

Our model's job is to learn this relationship and make predictions about a student's test score based on the hours they studied. Here is a Python code using PyTorch to create this model:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create a dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Convert to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model and define the loss and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_train_tensor)
    
    # Compute Loss
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_train_tensor)

# Plot the data
plt.scatter(x_train, y_train, label='True data')
plt.scatter(x_train, y_pred_tensor.numpy(), label='Predictions')
plt.legend()
plt.show()
```

This script creates and trains a model to predict a student's test score based on the number of hours they studied. After training, the script evaluates the model and plots the true data and the model's predictions.

Keep in mind that this is a simple example. In real-world scenarios, datasets are usually much larger and models might be more complex. But the principles remain the same: you define a model, specify a loss function, and optimize the model's parameters by minimizing the loss function.


------
## Introduction

Deep learning allows us to create complex models to understand data better. In this lesson, we'll be exploring how to implement an artificial neural network (ANN) for regression using PyTorch, and how it relates to traditional statistical models like regression.

## Implementing ANN for Regression

### Libraries and Data Generation

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(0)
x = np.random.rand(30, 1)
y = x + np.random.rand(30, 1) / 2
```

Here we import the necessary libraries and generate random data for `x` and `y`. We then plot this data to visualize the linear relationship.

### Building the Model

```python
# Define model
model = nn.Sequential(
    nn.Linear(1, 1),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(1, 1)  # output layer
)
```

In PyTorch, we define our model using the `Sequential` function. This allows us to string together the layers of our model, including the input layer, a ReLU (Rectified Linear Unit) activation function, and the output layer. The inputs to the `nn.Linear()` function are the number of inputs and the number of outputs.

### Meta Parameters and Training

```python
# Define meta parameters
learning_rate = 0.05
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)
    # Compute loss
    loss = loss_fn(y_pred, y)
    losses.append(loss)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

We define some meta parameters for our model, including the learning rate, the loss function (mean squared error), and the optimizer (stochastic gradient descent).

The training loop includes a forward pass (inputting the data and getting the model's output), computing the loss, and backpropagation (adjusting the model's parameters based on the loss).

### Model Evaluation

```python
# Run forward pass
predictions = model(x)

# Compute test loss
test_loss = torch.mean((predictions - y)**2)
print('Test loss:', test_loss.item())

# Plot loss over time
plt.plot(range(epochs), losses)
plt.show()

# Plot predictions vs actual data
plt.scatter(x, y)
plt.plot(x, predictions.detach(), color='red')
plt.show()

# Compute correlation
correlation = np.corrcoef(predictions.detach().numpy().flatten(), y.flatten())[0,1]
print('Correlation:', correlation)
```

After training, we evaluate our model by running a forward pass and computing the test loss. We also plot the loss over time during training, and compare the model's predictions with the actual data. Lastly, we compute the correlation between the predictions and the actual data.

## Traditional Statistical Models vs Deep Learning Models

Traditional statistical models, like regression, ANOVA, or general linear models, often work better on smaller or simpler datasets or where linear solutions are optimal. These models are usually better mathematically characterized and offer guaranteed optimal solutions. They are also often more interpretable compared to deep learning models.

However, as deep learning continues to evolve, these models could potentially supersede traditional statistical models in the future. This ongoing discussion is an important aspect of understanding the evolving landscape of machine learning and statistics.

## Note 9: The Code

The main script includes several steps:

1. **Importing Libraries**: Essential Python libraries are imported, including `torch` which is used for creating and managing the deep learning model.

    ```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    ```

2. **Creating the Dataset**: The dataset is created by generating 30 random numbers for X and Y is equal to X plus some scaled random numbers. This created a linear relationship between X and Y.

    ```python
    np.random.seed(0)
    x = np.random.rand(30, 1)
    y = 2*x + 1 + 0.1*np.random.randn(30, 1)
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    ```

3. **Building the Model**: A simple model is built using PyTorch's neural network (`nn`) module. The model contains an input layer, a non-linear activation function (ReLU), and an output layer.

    ```python
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(1, 1),
    )
    ```

4. **Setting Meta-Parameters**: Learning rate and loss function are defined. Mean Squared Error (MSE) loss function is used here. An optimizer is also specified, which is Stochastic Gradient Descent in this case.

    ```python
    learning_rate = 0.05
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ```

5. **Training the Model**: The model is trained for 500 epochs. The training loop includes the forward pass, computing the loss, and backpropagation.

    ```python
    epochs = 500
    for t in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

6. **Testing and Visualizing the Model**: After training, the model's performance is evaluated using the test data. The loss is calculated, and the actual data and the model's predictions are plotted for visual comparison.

    ```python
    test_loss = (model(x) - y).pow(2).mean().item()
    print(f"Test loss: {test_loss}")
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(epochs), losses.detach().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.scatter(x.detach().numpy(), y.detach().numpy(), label='Data')
    plt.plot(x.detach().numpy(), model(x).detach().numpy(), 'r-', label='Predicted')
    plt.legend()
    plt.show()
    ```

## Note 10: Traditional Models vs Deep Learning

Despite the remarkable potential and achievements of deep learning, traditional statistical models like regression, ANOVA, general linear models, etc. still hold significant value. They tend to work better on smaller or simpler datasets, or when linear solutions are optimal. 

These models also tend to be mathematically better characterized and often have guaranteed optimal solutions, unlike gradient descent. Also, traditional models tend to be more interpretable. 
