## Basic Architecture of ANNs

ANNs are composed of several components:
- **Inputs** (x1, x2, etc.): These are the features or data that are fed into the network. 
- **Weights** (w1, w2, etc.): These parameters determine the influence of each input on the output. They are learned during the training process.
- **Computational node**: This node computes the weighted sum of the inputs. This weighted sum, a linear operation, then gets passed through a nonlinear function.
- **Nonlinear function**: This function (also known as the activation function) introduces non-linearity into the network, allowing it to learn complex patterns.
- **Output** (y-hat): This is the prediction of the model. 
- **Bias term**: This is often set to one and implicitly included in the model, but it does have its own weight. The bias term allows the activation function to be shifted to the left or right.

```
Basic Perceptron Model:
x1, x2 (inputs) --[W1, W2 (weights)]--> Σ (computational node) --[f (nonlinear function)]--> y-hat (output)
```

## Feature Space and Separating Hyperplanes
The **feature space** is a geometric representation of the data where each feature is an axis and each observation is a coordinate. 

When considering binary classification (e.g., pass or fail), we seek to find a **separating hyperplane** in the feature space. This hyperplane is a boundary that divides the feature space into two categories. In a 2D space, this hyperplane is a line; in a 3D space, it's a plane; in higher dimensions, it's a hyperplane.

```
Feature Space:
2D space with x-axis representing hours studied and y-axis representing hours slept. Each data point (student) is a coordinate on this plane.

Separating Hyperplane:
A line that best separates the passing students from the failing students.
```

## Types of ANN Predictions

ANNs can make different types of predictions:

1. **Discrete/Categorical/Binary/Boolean Predictions**: These are distinct categories or options. For example, pass or fail. 

2. **Numeric/Continuous Predictions**: These are output values on a continuous scale. For example, an exam score.

The type of output that the model produces influences the choice of activation function and the overall architecture of the ANN.

```
Discrete Prediction Example:
Pass or fail prediction based on hours studied and hours slept.

Continuous Prediction Example:
Predicting the actual exam score based on hours studied and hours slept. Requires an additional dimension to represent the continuous exam score.
```

## Key Concepts and Definitions

- **Perceptron**: The simplest type of artificial neural network, consisting of a single neuron or node.
- **Feature Space**: A geometric representation of the data where each feature is an axis and each observation is a coordinate.
- **Separating Hyperplane**: A boundary that divides the feature space into two categories.
- **Gradient Descent**: The process by which the ANN adjusts its weights during training to minimize the difference between its predictions and the actual outcomes.
- **Activation Function**: A function that introduces non-linearity into the ANN, allowing it to learn complex patterns.
- **Discrete/Categorical/Binary/Boolean Predictions**: Predictions that fall into distinct categories or options.
- **Numeric/Continuous Predictions**: Predictions that are output values on a continuous scale.

## Diagrams and Figures

1. **Diagram of a Basic Perceptron Model**: Shows the inputs, weights, computational