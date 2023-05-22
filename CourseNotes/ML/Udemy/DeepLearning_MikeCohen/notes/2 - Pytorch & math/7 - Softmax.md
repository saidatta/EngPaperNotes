
### Softmax Function
- Used to transform arbitrary numbers into probability distributions
- Ensures that the output values sum to 1
- Formula: 
    - `S(x_i) = e^(x_i) / Σ(e^(x_j))`
    - S(x_i): Softmax of x_i
    - x_i: i-th element of input vector
    - Σ: Summation over all elements
- Can be applied element-wise to a vector
##### Example
- Given x = {1, 2, 3}:
1.  Calculate the exponent of each element in x:
    
    -   e^(1) ≈ 2.7183
    -   e^(2) ≈ 7.3891
    -   e^(3) ≈ 20.0855
2.  Calculate the sum of these exponentiated values:
    
    -   Σ(e^(x_j)) = 2.7183 + 7.3891 + 20.0855 ≈ 30.1929
3.  Compute the softmax output for each element in x using the formula:
    
    -   S(x_1) = e^(1) / Σ(e^(x_j)) = 2.7183 / 30.1929 ≈ 0.0900
    -   S(x_2) = e^(2) / Σ(e^(x_j)) = 7.3891 / 30.1929 ≈ 0.2447
    -   S(x_3) = e^(3) / Σ(e^(x_j)) = 20.0855 / 30.1929 ≈ 0.6652

The softmax output for the given input vector x = {1, 2, 3} is approximately {0.0900, 0.2447, 0.6652}. As expected, the output values are non-negative and sum to 1.

#### Softmax Function Properties
1. Non-negative outputs
2. Outputs sum to 1
3. Continuous and differentiable

#### Uses of Softmax
- Multi-class classification problems
- Output layer in deep learning networks
###### Some usage examples
1.  **Text Classification**: Suppose you're developing a model that classifies news articles into different categories like sports, politics, entertainment, or technology. In this case, the softmax function can be used to convert the output of the model into probabilities for each class, making it easy to choose the most likely category.
    
2.  **Handwritten Digit Recognition**: In this application, a model is trained to recognize handwritten digits (0-9). The softmax function can be used to compute the probability distribution over the 10 possible digits, allowing the model to output the digit with the highest probability.
    
3.  **Natural Language Processing**: Softmax is frequently used in NLP tasks like Named Entity Recognition (NER) and Part-of-Speech (POS) tagging, where the goal is to classify words or phrases into specific categories. The softmax function helps generate probabilities for each possible class, making it easier to predict the correct category.
    
4.  **Image Classification**: In image classification problems, a model is trained to categorize images into one of several possible classes (e.g., animals, plants, objects). The softmax function can be used at the output layer to produce a probability distribution over these classes, allowing the model to select the most likely class for a given input image.
    
5.  **Reinforcement Learning**: In some reinforcement learning scenarios, an agent needs to select actions based on the predicted rewards or Q-values. The softmax function can be applied to these Q-values, generating a probability distribution over actions, and helping the agent balance exploration and exploitation.

#### Softmax Function Graphs
- Nonlinear increase with larger numbers
- Y-axis scaling differences
    - More numbers in input range lead to smaller output values
    - Softmax output needs to sum to 1

#### Computing Softmax with NumPy
```python
import numpy as np
z = [1, 2, 3]
numerator = np.exp(z)
denominator = np.sum(numerator)
sigma = numerator / denominator
```

#### Computing Softmax with PyTorch
```python
import torch
import torch.nn as nn

z = [1, 2, 3]
softmax = nn.Softmax(dim=0)
z_tensor = torch.tensor(z)
sigma = softmax(z_tensor)
```

#### Graphs and Visualization
- Linear transformation in log space
- Use logarithmic scaling on the y-axis

#### Key Takeaways
- Softmax function is essential for multi-class classification
- Can be easily computed using NumPy and PyTorch
- Provides a linear transformation in log space