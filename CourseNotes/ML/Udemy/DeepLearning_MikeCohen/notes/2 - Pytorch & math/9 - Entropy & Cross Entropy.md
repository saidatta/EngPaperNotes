
- **Entropy**: A measure of the uncertainty, randomness, or disorder in a set of data. It quantifies the average amount of information needed to predict the outcome of a random event.
- Formula: H(P) = - Σ [P(x) * log₂(P(x))]
  - H(P): Entropy of probability distribution P
  - P(x): Probability of event x occurring
  - log₂(P(x)): Logarithm base 2 of the probability of x
- Units: bits (base 2 logarithm)
- Entropy increases with the number of equally likely events (higher uncertainty)
- Binary entropy: A specific case of entropy when there are only two possible events (e.g., binary classification)
  - Formula: H(P) = - [p * log₂(p) + (1-p) * log₂(1-p)]
  - p: Probability of the first event
  - 1-p: Probability of the second event

- **Cross Entropy**: A measure of the dissimilarity between two probability distributions P and Q over the same set of events. It can be used to assess the performance of a prediction model.
- Formula: H(P, Q) = - Σ [P(x) * log₂(Q(x))]
  - H(P, Q): Cross entropy between probability distributions P and Q
  - P(x): Probability of event x occurring according to distribution P
  - Q(x): Probability of event x occurring according to distribution Q
- Not symmetric: Cross entropy is not symmetric, meaning that H(P, Q) ≠ H(Q, P).
- Binary Cross Entropy: A specific case of cross entropy when there are only two possible events (e.g., binary classification)
  - Formula: H(P, Q) = - [p * log₂(q) + (1-p) * log₂(1-q)]
  - p: Probability of the first event according to distribution P
  - q: Probability of the first event according to distribution Q
  - 1-p: Probability of the second event according to distribution P
  - 1-q: Probability of the second event according to distribution Q

Torch Implementation:
1. Import torch and torch.nn.functional as F
   - torch: PyTorch library for deep learning
   - torch.nn.functional: A submodule of PyTorch containing various functions for neural networks
2. Use F.binary_cross_entropy function to compute binary cross entropy between two probability distributions
3. Inputs order: Q (predicted probabilities), P (actual probabilities)
   - Q: Model-predicted probabilities
   - P: Ground truth probabilities (e.g., class labels)
4. Convert Python lists to PyTorch tensors for compatibility with the F.binary_cross_entropy function
5. Note: PyTorch binary cross entropy function is sensitive to the order of inputs (Q should be first, then P)
```python 
import torch 
import torch.nn.functional as F

# Ground truth probabilities (actual probabilities)

P = [1, 0]

# Model-predicted probabilities

Q = [0.1, 0.9]

# Convert Python lists to PyTorch tensors for compatibility

P_tensor = torch.tensor(P, dtype=torch.float32) Q_tensor = torch.tensor(Q, dtype=torch.float32)

# Compute binary cross entropy between two probability distributions using Torch

binary_cross_entropy = F.binary_cross_entropy(Q_tensor, P_tensor)

# Print the result

print("Binary Cross Entropy:", binary_cross_entropy.item()) 
```

#### Example 
Here's a real-life example that demonstrates the usage of entropy and cross-entropy in the field of natural language processing (NLP) for sentiment analysis:

Let's consider a sentiment analysis task where we want to classify movie reviews as positive or negative. We have a dataset of movie reviews labeled as positive or negative, and we want to train a deep learning model to predict the sentiment of new, unseen reviews.

To train the model, we represent each review as a numerical feature vector using techniques like word embeddings or bag-of-words representation. Each feature vector represents the occurrence or frequency of words in the review.

Once we have the feature vectors, we pass them through the model, which outputs a probability distribution over the two classes: positive and negative. The model predicts the probability of the review belonging to each class.

Now, let's say we have a movie review that the model predicts with the following probabilities: [0.8, 0.2], where index 0 represents the positive sentiment class and index 1 represents the negative sentiment class.

To evaluate the model's performance, we calculate the cross-entropy loss between the predicted probabilities and the true labels. The true label for this review is positive. We can represent the true label as a one-hot encoded vector: [1, 0], where index 0 corresponds to the positive class.

The cross-entropy loss is computed using the formula:

```bash
cross_entropy = -sum(true_label * log(predicted_prob))
```


Substituting the values, we have:

```scss
cross_entropy = -(1 * log(0.8) + 0 * log(0.2)) = -log(0.8) ≈ 0.223
```

The lower the cross-entropy value, the better the model's prediction aligns with the true label. In this case, the cross-entropy loss indicates that the model's predicted probability distribution is relatively close to the true label.

Entropy, on the other hand, measures the uncertainty or surprise associated with the predicted probability distribution. In this example, the entropy of the predicted probabilities [0.8, 0.2] can be calculated as:

```scss
entropy = -(0.8 * log(0.8) + 0.2 * log(0.2)) ≈ 0.500
```

A higher entropy value indicates higher uncertainty or lack of information in the predicted probabilities. In this case, the entropy value suggests that the model's prediction is relatively uncertain.

By calculating cross-entropy and entropy, we can assess the model's performance in predicting sentiment and understand the level of uncertainty in its predictions. These metrics play a crucial role in evaluating and improving the performance of deep learning models in sentiment analysis tasks.
