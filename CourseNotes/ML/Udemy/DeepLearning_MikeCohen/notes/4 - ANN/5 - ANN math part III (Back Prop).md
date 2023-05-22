## Summary
In this chapter, we extend our understanding from the basic Perceptron model to a deep network. We learn about the process of forward propagation and back propagation, as well as the concept of the chain rule in calculus. Additionally, we learn how back propagation is essentially gradient descent applied in a different context.

---

## The Shortening
- Deep learning networks can become quite complex, so for simplicity, we replace the entire Perceptron diagram (inputs, weights, linear summation, nonlinear activation function) with a single circle, referred to as a unit or a node.

![Perceptron Diagram](obsidian://open?vault=ML&file=Udemy%2FDeepLearning_MikeCohen%2Fmedia%2FShortening.png)

---

## From Perceptron to Deep Network

- Each node in a deep learning network is essentially a standalone Perceptron.
- Inputs are received, a weighted sum is computed, passed through a non-linearity, and then forwarded to the next node in the network.
- The nodes in the initial layer receive input from the data, while nodes in subsequent layers receive inputs as outputs from previous layers.
- Each node operates independently, unaware of the larger network it is part of.

![Deep Network Diagram](DeepNetworkDiagram.png.md)

---

## Forward Propagation and Back Propagation

- Forward Propagation: This is the process where the input data is propagated forward in the network. The inputs are processed by each layer and the output is generated.
- Back Propagation: The process of updating the weights of the network based on the error of the output. The error is computed at the output and is propagated backward in the network to adjust the weights.

---

## Back Propagation is Gradient Descent Super-Charged

- Back propagation is an extension of gradient descent, where the weights are updated by subtracting the gradient of the loss function times the learning rate from the current weights.
- The gradient of the loss function with respect to the weights is computed using the chain rule of calculus.
- For backpropagation, we consider each node as a function and calculate its derivative. Then using chain rule, we calculate the total derivative.

- Backpropagation is essentially a clever application of the chain rule from calculus in order to compute gradients efficiently, making it possible to update the weights of a neural network through gradient descent.

The gradient descent algorithm is as follows:

1.  Initialize weights randomly.
2.  Feed forward inputs through the network to get the output.
3.  Compute the error of the output.
4.  Backpropagate the error through the network, calculating the gradient of the error with respect to each weight.
5.  Update the weights by subtracting the gradient times the learning rate from the current weights.

The weight update rule is:

mathematicaCopy code

`Δw = -η * ∂E/∂w`

where:

-   `Δw` is the change in weights.
-   `η` is the learning rate.
-   `∂E/∂w` is the gradient of the error with respect to the weights.

This gradient is computed using the chain rule:

cssCopy code

`∂E/∂w = ∂E/∂a * ∂a/∂z * ∂z/∂w`

where `a` is the activation of the neuron, `z` is the weighted input to the neuron, and `w` is the weight.

The backpropagation algorithm is what allows us to compute this gradient efficiently.

### Real-Life Example

Let's return to our weather prediction model. We have inputs (temperature), weights, a sigmoid activation function, and a mean squared error loss function. We've made a prediction and computed an error, and now we want to adjust our weights to reduce the error.

First, we calculate the gradient of the error with respect to the weights using the chain rule, as described above.

Next, we update our weights by subtracting the gradient of the error with respect to the weights, multiplied by a small learning rate, from the current weights. This causes the weights to change in the direction that reduces the error.

For example, if the current temperature is 15 degrees, the weight is 0.6, the bias is 0.2, and the actual outcome (rain = 1, no rain = 0) is 1, the model might predict a 0.7 chance of rain. If this is incorrect (it didn't rain), we would compute an error and use backpropagation to compute the gradient of the error with respect to the weight. We then update the weight by subtracting the gradient times a learning rate (say, 0.01) from the current weight. This would give us a new weight that should result in a lower error the next time the model predicts whether it will rain given a temperature of 15 degrees.

![Back Propagation Equation](BackPropEquation.png.md)

---

### Backpropagation and Chain Rule

Backpropagation is the method used to update the weights in an Artificial Neural Network (ANN). It is derived from the chain rule from calculus, which is used to compute the derivative of composite functions.

Let's look at a simple example. Suppose we have a feedforward network with a single hidden layer and we are using the mean squared error (MSE) as our loss function and a sigmoid as our activation function.

Given an input `x`, the output `y` is calculated as follows:

```
z = w*x + b          [1]
a = sigmoid(z)       [2]
y_hat = a            [3]
E = 1/2 * (y - y_hat)^2     [4]
```

where:
- `z` is the weighted input,
- `a` is the activation of the neuron,
- `y_hat` is the predicted output, and
- `E` is the error.

The goal is to find `dE/dw` - the derivative of the error `E` with respect to the weight `w`. We do this through the chain rule:

```
dE/dw = dE/da * da/dz * dz/dw
```

Since:
- `dE/da` = `-(y - y_hat)` (from equation [4]),
- `da/dz` = `sigmoid(z) * (1 - sigmoid(z))` (derivative of the sigmoid function), and
- `dz/dw` = `x` (from equation [1]).

Substituting these values gives us:

```
dE/dw = -(y - y_hat) * sigmoid(z) * (1 - sigmoid(z)) * x
```

This is the value we use to update the weights in the network during training.

### Real-Life Example

Consider a simple weather prediction model that uses temperature (`x`) to predict whether it will rain (`y`). The model uses a single-layer feedforward network with a sigmoid activation function.

Given a temperature `x`, it calculates a weighted sum `z = w*x + b`, applies a sigmoid function to get `a = sigmoid(z)`, and produces a prediction `y_hat = a`. If the actual outcome `y` (1 if it rains, 0 if it doesn't) differs from `y_hat`, there's an error `E = 1/2 * (y - y_hat)^2`.

To improve the model, we need to adjust `w` and `b` to reduce the error. We use backpropagation and the chain rule to calculate the rate at which the error changes with respect to changes in `w` and `b`, and then update `w` and `b` to reduce the error.
![Chain Rule Application](ChainRuleApplication.png)

---

## References

- Cohen, M. (2023). Deep understanding of deep learning. Udemy.
