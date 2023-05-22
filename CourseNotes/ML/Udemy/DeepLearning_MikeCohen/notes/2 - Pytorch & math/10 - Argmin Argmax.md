
## Chapter: Min/Max, Argmin/Argmax

### Minimum and Maximum
Given a set of numbers, the minimum value is the smallest number, and the maximum value is the largest number.

### Argmin and Argmax
The Argmin and Argmax functions return the index or location of the minimum and maximum values, respectively.

- Argmin: Returns the index of the smallest value
- Argmax: Returns the index of the largest value

In deep learning, Argmax is often used to identify the most probable class or label from the output of a neural network model, such as in image classification tasks.

### NumPy Implementation
Using NumPy, you can find the min, max, argmin, and argmax of a vector or matrix.

```python
import numpy as np

vector = np.array([10, 40, 30, -3, 5])
min_value = np.min(vector)
max_value = np.max(vector)
argmin_value = np.argmin(vector)
argmax_value = np.argmax(vector)
```

For matrices, you can also find the min and max values across rows or columns by specifying the `axis` parameter.

```python
matrix = np.array([[0, 1, 6], [5, 7, 2]])
min_value_axis0 = np.min(matrix, axis=0)
min_value_axis1 = np.min(matrix, axis=1)
```

### PyTorch Implementation
The same min, max, argmin, and argmax functions can be implemented in PyTorch using tensors.

```python
import torch

tensor = torch.tensor([10, 40, 30, -3, 5])
min_value = torch.min(tensor)
max_value = torch.max(tensor)
argmin_value = torch.argmin(tensor)
argmax_value = torch.argmax(tensor)
```

For matrices, you can use `torch.min` and `torch.max` with the `dim` parameter to find the min and max values across rows or columns.

```python
matrix = torch.tensor([[0, 1, 6], [5, 7, 2]])
min_value_dim0 = torch.min(matrix, dim=0)
min_value_dim1 = torch.min(matrix, dim=1)
```

Note that the output organization in PyTorch is slightly different from NumPy, as it returns an object with two attributes: `values` and `indices`. You can access them with `min_value_dim0.values` and `min_value_dim0.indices`.

#### **Example of ArgMax**

- Suppose we have a convolutional neural network (CNN) trained on a dataset of various animals, including cats, dogs, and birds. After training, the model is capable of predicting the probabilities for each class given an input image.
- Let's say we have an input image of a cat, and the model produces the following output probabilities: [0.1, 0.7, 0.2], where index 0 represents the cat class, index 1 represents the dog class, and index 2 represents the bird class.
- To determine the predicted class, we apply the Argmax function to find the index with the highest probability. In this case, the Argmax function will return index 1, indicating that the model predicts the input image as a dog.
- 
- Here's an example code snippet to illustrate this:

pythonCopy code

```python
import numpy as np  # Output probabilities from the model 
output_probabilities = np.array([0.1, 0.7, 0.2])  # Applying Argmax to get the predicted class
predicted_class = np.argmax(output_probabilities)  # Mapping the predicted class to its label 
labels = ['cat', 'dog', 'bird'] 
predicted_label = labels[predicted_class]  print(f"Predicted class {predicted_label}")`
```

The output will be:

`Predicted class: dog`

In this example, the Argmax function helped identify the index with the highest probability, allowing us to determine the predicted class of the input image.