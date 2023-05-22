## Expectation vs Reality

In a perceptron model, we feed data into the model and it computes a weighted linear sum of those inputs, which then passes through a nonlinear activation function to provide a prediction. This prediction may not always align with the reality, meaning there can be an error in the prediction. 

##### Examples
1. If a model predicts that a student will pass an exam with a 98% probability, but the student fails, then there's a significant error in the prediction.
		i. The error can be calculated as the difference between the model's prediction (denoted as Y hat) and the actual outcome (denoted as Y). We need to quantify this error for the model to learn from it. 
| Student | Actual Outcome (Y) | Model's Prediction (Y hat) | Error (Y - Y hat) |
|---------|-------------------|----------------------------|-------------------|
| John    | 1 (Passed)        | 0.95                       | 1 - 0.95 = 0.05   |
| Emily   | 0 (Failed)        | 0.98                       | 0 - 0.98 = -0.98  |
| Alex    | 1 (Passed)        | 0.70                       | 1 - 0.70 = 0.30   |
| Maria   | 1 (Passed)        | 0.88                       | 1 - 0.88 = 0.12   |
| Noah    | 0 (Failed)        | 0.60                       | 0 - 0.60 = -0.60  |

In this table, the Actual Outcome (Y) is either 1 (Passed) or 0 (Failed). The Model's Prediction (Y hat) is a probability between 0 and 1, where values closer to 1 indicate a higher likelihood of passing according to the model. The Error (Y - Y hat) is the difference between the actual outcome and the model's prediction. Positive errors indicate that the model underestimated the likelihood of passing, while negative errors indicate that the model overestimated the likelihood of passing.

In this scenario, the model's predictions were significantly off for Emily and Noah, leading to high absolute errors. These errors provide a valuable signal for the model to learn from and improve its future predictions.

2. Consider a simple perceptron model trained to classify whether an email is spam (1) or not spam (0). Let's say the model predicts that an email is spam with a 90% probability, i.e., Y hat = 0.9. However, in reality, the email is not spam, i.e., Y = 0. So, the error in the prediction is Y - Y hat = 0 - 0.9 = -0.9.

## Loss Functions

Errors are used to generate losses, which is done using loss functions. There are many loss functions in deep learning, but most of them are variants of two commonly used ones: Mean Squared Error (MSE) and Cross-entropy.

- **Mean Squared Error (MSE):** Used for continuous data when the model outputs a numerical prediction. The MSE loss function is straightforward: it's the square of the difference between the model's prediction and the target. The squaring ensures that all values are positive and it links the MSE to other methods in machine learning and statistics like regression, ANOVA, and variance.

  MSE Formula: 
  ```
  MSE = 1/2 * (Y - Y hat)^2
  ```
**Example:**
MSE = 1/2 * (Y - Y hat)^2
MSE = 1/2 * (0 - 0.9)^2 = 0.405

- **Cross-Entropy:** Also known as logistic error function, it's used for categorical data when the model outputs a probability. The loss function here is cross entropy, and it's the same formula as the one used in the math section of the course.

  Cross-Entropy Formula:
  ```
  Cross-Entropy = -Y log(Y hat) - (1 - Y) log(1 - Y hat)
  ```
**Example:**
- This is more commonly used for classification problems. Using the same example:
Cross-Entropy = -Y log(Y hat) - (1 - Y) log(1 - Y hat)
Cross-Entropy = -(0) log(0.9) - (1 - 0) log(1 - 0.9) = 2.3

## From Loss to Cost

Loss functions work per sample. You plug in the data from one sample, get one output, and compute the loss for that sample. The cost function, on the other hand, is simply the average of the losses for many different samples. 

If you have N samples, you compute the loss for each individual sample, average all of those losses together, and call that the cost. 

Cost Function Formula:
```
Cost = 1/N * Σ Losses
```
**Example**:
Imagine we have three samples with the following Cross-Entropy losses: 0.5, 0.7, and 1.2. The cost function would then be the average of these losses:

```makefileCopy 
Cost = 1/N * Σ Losses Cost = 1/3 * (0.5 + 0.7 + 1.2) = 0.8
```


## The Goal of DL Optimization

The entire goal of deep learning is to find the set of weights that minimizes the cost function. This is the optimization criterion used in the training. Training based on individual losses can be time-consuming and can lead to overfitting, so we often train models using batches or groups of samples to average the losses.

The Optimization Goal:
```
Find W such that J(W) is minimized
Where, J(W) is the cost function
```
**Example:**
In the context of our spam classifier, the goal of deep learning is to adjust the model's weights during training in such a way that the cost function (average loss over all email samples) is minimized. For example, if the initial weights result in a cost of 0.8, the optimization algorithm will adjust the weights to try and get this cost lower, say to 0.5 or 0.3, etc.

## Is Anything Lost in Cost?

Averaging over too many losses can decrease sensitivity, particularly when there is a lot of sampling variability. So, in practice, models are often trained using batches of samples. This introduces you to the concept of batch training.

**Example:**
If our email dataset is very large, we may choose to train our model on smaller batches of emails at a time, say 100 emails per batch. If the variability in the spam characteristics is high between batches, this could impact the sensitivity of the model to accurately classify spam. This is a trade-off between computational efficiency (training on batches is faster) and model accuracy (training on individual samples is more accurate but slower).