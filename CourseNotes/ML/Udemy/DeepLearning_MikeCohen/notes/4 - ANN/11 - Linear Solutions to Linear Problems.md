## Key Points:
1. The video introduces the concept of how deep learning models can be used to solve linear problems effectively with one minor change.
2. The change involves removing the non-linear activation functions from the hidden layers of the model, keeping the non-linearity at the output layer.
3. The linear model performs remarkably well, often reaching 100% accuracy, in comparison to a non-linear model, which can exhibit chaotic results.

### Linear Solutions to Linear Problems:
The video presents a small code challenge using the model from previous videos. The challenge involves making one small change to the model, which dramatically improves the performance on the Cordy's classification problem. The change is to remove the non-linear activation functions (ReLU) from the hidden layers, leaving only the output layer's non-linear activation function (sigmoid). 

### Impact of Removing Non-linear Activation Functions:
After implementing this change and running the model, the results were remarkable. Despite some instances of lower accuracy, the model generally achieved a near-perfect 99.5% accuracy or even 100% accuracy. It was observed that the number of models with greater than 70% accuracy was exactly 100%. This performance was far more consistent than the previous models that included the non-linear activation functions.

### Non-linear vs Linear Models:
The video goes on to explain why this model behaved as it did. The Cordy's data set is a simple separation problem that is linearly solvable, meaning a straight line can be drawn in the feature space. As such, a linear model is likely to outperform a non-linear model because adding non-linearities forces the model to search for more complex solutions.

### Implications for Problem Solving:
This experiment reinforces the principle that not all problems need complex solutions. While deep learning and non-linear models are great for complex, non-linear problems, simpler linear problems are best suited to linear solutions. The video concludes by emphasizing that every problem should be approached with an open, analytical, creative, and critical mindset to determine the most appropriate solution.

### Code Snippet:
The code changes involve commenting out the ReLU activation functions in the hidden layers while leaving the sigmoid activation function in the output layer. The model then consisted of a linear layer followed by a linear output layer.

```python
# Existing code: model with non-linear activation function (ReLU)
# Commented out code: Removing non-linear activation functions in hidden layers

# model = nn.Sequential(
#     nn.Linear(2,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,2),
#     nn.Sigmoid()
# )

# Modified code: model without non-linear activation function (ReLU) in hidden layers
model = nn.Sequential(
    nn.Linear(2,64),
    #nn.ReLU(),
    nn.Linear(64,64),
    #nn.ReLU(),
    nn.Linear(64,2),
    nn.Sigmoid()
)
```
To run the modified model, either run each cell one by one, or use the "Run All" command.

## Upcoming:
In the next video, the mathematical difference between linear and non-linear models will be explored, along with the concept that multi-level linear models do not actually exist.

## References:
- Deep Understanding of Deep Learning by Mike Cohen on Udemy.
- [Linear vs. Nonlinear Models](https://www.sciencedirect.com/topics/engineering/linear-models)
- [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
- [Deep Learning

 Models for Linear Problems](https://towardsdatascience.com/can-deep-learning-solve-every-problem-5e1867132f88)
