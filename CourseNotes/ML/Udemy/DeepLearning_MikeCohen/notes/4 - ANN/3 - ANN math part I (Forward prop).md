## Overview

In this note, we delve into the mathematics underpinning Artificial Neural Networks (ANNs), particularly focusing on the elements such as biases, weights, and activation functions that contribute to the functioning of ANNs.

## Key Concepts

### Forward Propagation
Forward Propagation is the process by which input data are transformed into an output in a neural network. This mechanism involves the use of perceptrons, which are fundamental building blocks of deep learning.

#### Perceptron
![Perceptron2](https://miro.medium.com/v2/resize:fit:1400/1*ts5LSdtkfSsMYS7M0X84Tw.gif)
![Perceptron](https://upload.wikimedia.org/wikipedia/commons/8/8a/Perceptron_example.svg)

A **perceptron** takes a number of inputs (for example, two features of the real world), applies weights to them, and adds a bias term. The result is a linear weighted sum of the inputs, which is then passed through a nonlinear function known as the activation function.

The output (`y`) of a model is expressed mathematically as:

`y = σ(x1*w1 + x2*w2 + ... + xn*wn + b)`

The term `x1*w1 + x2*w2 + ... + xn*wn + b` is a dot product between a vector containing the inputs and a vector containing the weights, and it represents the linear part of the computation in a perceptron. `σ` is the activation function, which introduces non-linearity.

### Activation Function
An activation function maps an input (the output of the linear part of the model) onto an output (the final output of the model). There are various types of activation functions used in deep learning, the most common being sigmoid, hyperbolic tangent (tanh), and Rectified Linear Unit (ReLU).

![Activation Functions](https://miro.medium.com/max/1200/1*4ZEDRpFuCIpUjNgjDdT2Lg.png)

### Classification in ANNs
ANNs classify inputs by plotting them in a feature space and dividing that space with a decision boundary or separating hyperplane. This boundary is determined by the weights and the bias of the model. The location of the decision boundary doesn't change with the application of the nonlinear activation function; instead, the numerical values in the feature space are transformed.

In a 2D feature space, the decision boundary is a line where the output of the model is equal to 0 (or 0.5 when converted into probabilities). Any data points on one side of the line belong to one class or category, while data points on the other side belong to another class or category.

## Learning the Weights
Learning the appropriate weights to create the optimal decision boundary is a crucial part of training an ANN. This is achieved through a process called backpropagation, which is essentially gradient descent applied to neural networks. 

## Summary

This note provided a detailed view of the math underpinning ANNs, including the concept of forward propagation, the role of the activation function, the process of classifying inputs, and the learning of weights through backpropagation. It's important to note that while we focused on a single perceptron, deep learning involves repeating and linking many such perceptrons together. 

## Up Next

In the following sections, we'll explore errors, losses, and cost functions, followed by a deep dive into the concept of backpropagation. 

--- 

## References and Further Reading
- [A Gentle Introduction to Neural Networks](https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc)
- [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-949


### Sigmoid activation example: IRIS classification (flowers)
Sure, let's create a simple binary classifier using a single layer Perceptron with a sigmoid activation function. We'll use the popular Iris dataset from scikit-learn, which is a multi-class classification problem, but for the sake of simplicity, we'll convert it into a binary classification problem.

Please note that the example provided here is extremely simplified and is not meant to be a representation of good machine learning practices - it's just an illustrative example. 

First, let's import the necessary libraries:

```python
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

Next, let's load the Iris dataset and convert it to a binary problem:

```python
# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to a binary problem by keeping only the first two classes
X = X[y < 2]
y = y[y < 2]
```

Now, let's split the dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We also standardize our features using StandardScaler for optimal performance
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Now, let's define the sigmoid function and its derivative:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

Now, let's create a simple Perceptron:

```python
class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.zeros(1)

    def forward(self, x):
        self.output = sigmoid(np.dot(x, self.weights) + self.bias)
        return self.output

    def compute_gradients(self, x, y):
        self.d_weights = (self.output - y) * sigmoid_derivative(np.dot(x, self.weights) + self.bias) * x
        self.d_bias = (self.output - y) * sigmoid_derivative(np.dot(x, self.weights) + self.bias)

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias
```

We can now train our Perceptron on the Iris dataset:

```python
p = Perceptron(n_inputs=4)  # We have 4 features in the Iris dataset
learning_rate = 0.1
n_epochs = 1000

# Training loop
for epoch in range(n_epochs):
    for i in range(len(X_train)):
        p.forward(X_train[i])
        p.compute_gradients(X_train[i], y_train[i])
        p.update_parameters(learning_rate)
```

Now, we can test our Perceptron on the test set:

```python
n_correct = 0
for i in range(len(X_test)):
    output = p.forward(X_test[i])
    if (output >= 0.5 and y_test[i] == 1) or (output < 0.5 and y_test[i] == 0):
        n_correct += 1

print(f"Test accuracy: {n_correct / len(X_test)}")
```
-----
### Explanation and future

The above code provides a simple example of a binary classifier using a single layer Perceptron with a sigmoid activation function. This model could be used in any real-life scenario where you need to distinguish between two classes based on given features.

The Iris dataset used in the example is a popular dataset in the field of machine learning. It contains measurements of 150 iris flowers from three different species. In the context of the dataset, a real-life use case might be a botanist wanting to automate the process of classifying iris flowers based on the lengths and widths of their petals and sepals.

However, the binary classification problem can be generalized to a wide range of other real-life scenarios. Here are a few examples:

1. **Email Spam Detection**: Classify emails as 'spam' or 'not spam' based on features like the email text, subject, sender, etc.

2. **Customer Churn Prediction**: Predict whether a customer will churn (i.e., leave the company) or not based on features like their usage of a company's product or service, demographic information, past purchase behavior, etc.

3. **Loan Default Prediction**: Predict whether a borrower will default on a loan based on features like their credit history, income level, loan amount, etc.

4. **Medical Diagnosis**: Diagnose whether a patient has a specific disease or not based on features like their symptoms, medical history, results from medical tests, etc.

In each of these cases, the model could be trained on a historical dataset where the true class is known, allowing it to learn how to classify new, unseen instances.