
## Supervised Learning

-   Associated with the success of machine learning
-   Task: taking a collection of input data to make predictions
-   Examples:

Input |Prediction

Pixel values of an image | Name of an object in an image
Measured medical data | State of a patient's health
Robotic sensor data | The location of obstacles

## Input Data Structure

-   Common mathematical structures: vector (1D), matrix (2D), tensor (higher dimensions)
	-   Machine learning problems are often high-dimensional
-   Input: feature vector, output: scalar, vector, or tensor
-   Function describing the relation between input and output: y=f(x)

## Features

-   Components describing the inputs to learning systems
-   Often measured data in machine learning
-   Attributes and features are used interchangeably

## Parameterized Approximation Function

-   Define a parameterized function to approximate the desired input-output relation: y = f’(x; w)
-   A model is an approximation of a system to predict behavior

## Learning Algorithm

-   Define a function to describe the goal of learning: loss function (L)
-   Use gradient descent to minimize the loss function
    -   Update rule: w_i ← w_i - α∇w L (Eq. 3)
    -   Gradient: ∇w = (∂/∂w_1, ..., ∂/∂w_n) (Eq. 4)
-   Goal: find values of w that best predict unseen data

## Supervised Learning Prediction

-   Regression: predict continuous output variable
-   Classification: predict discrete values
-   Output variable (y) in classification is called a label

## Probabilistic Framework

-   Describe the true underlying world model with a probability density function: p(Y=y∣x) (Eq. 5)
-   Model a density function: p(Y=y∣x;w) (Eq. 6)

## Bayesian Modeling

-   Probabilistic models and data used to estimate model parameters and make predictions
-   Maximum a posteriori choice (MAP) and maximum likelihood estimate (MLE) are common forms of machine learning
    -   MAP: w = argmax p(w∣y,x) (Eq. 8)

## Causal Models and Bayesian Networks

-   Formulate multivariate probabilistic models
-   Inference: derive predictions
-   Bayesian networks model entities that build the causal structure of the problem
-   Provide strong predictive strength and explanatory ability