In this chapter, we learned about creating and working with a multi-layer artificial neural network (ANN) in deep learning. The key topics covered in the chapter include the concept of hidden layers, input and output layers, coding specifics, and the importance of accuracy. Here are the detailed notes:

## Basic Components of Multilayer ANN

- A deep learning network comprises of three main components:
  - **Input Layer**: Directly connected to the outside world. Inputs to this layer are the actual data, for example, data from a database, experimental data, or pixel intensity values from an image.
  - **Output Layer**: Provides feedback into the outside world. This is where numerical predictions or probabilities of being in different categories are generated.
  - **Hidden Layers**: They reside between the input and output layers and are not connected to the real world. They don't get inputs from actual data and don't provide outputs that we interpret as relevant for real data.

```sh
graph LR
A[Input Layer] -- Data From Real World --> B[Hidden Layers]
B -- Outputs of Previous Layers --> C[Output Layer]
C -- Predictions/Probabilities --> Outside World
```
## Building a Multilayer ANN
- We can extend the code from a single-layer network to build a multi-layer network by adding more layers.
- In PyTorch, we create pairs of layers: a linear layer followed by a non-linear activation function. This pattern repeats for each hidden layer.
- We use the Rectified Linear Unit (ReLU) activation function for the input and hidden layers, while the output layer uses the sigmoid function.
- The parameters inside the linear layer functions represent the number of input features and output features respectively. For example, `nn.Linear(2, 16)` means that the input has 2 features, and the output from this layer (which becomes the input for the next layer) has 16 features.
- It's important to ensure that the number of output features in a layer matches the number of input features in the next layer to allow for correct matrix multiplication.
- The last output layer needs to have one output because we want to get one number out of the model. This output is interpreted as the probability of belonging to a specific category.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16) # Input layer
        self.fc2 = nn.Linear(16, 16) # Hidden layer
        self.fc3 = nn.Linear(16, 1) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Apply ReLU to input layer
        x = torch.relu(self.fc2(x)) # Apply ReLU to hidden layer
        x = torch.sigmoid(self.fc3(x)) # Apply Sigmoid to output layer
        return x
```

## Training the Multilayer ANN

- The process of training a multilayer ANN is the same as training a single-layer network. The primary difference is in the function that builds the model, as it now includes more layers.
- If the output of the model is the raw output passed through a sigmoid function, the decision boundary for classification should be 0.5. This is because the sigmoid function produces output in the range of [0,1], and values above 0.5 are closer to 1, while those below 0.5 are closer to 0.
- On the other hand, if the raw output of the model is used (negative to positive values), the decision boundary should be 0, as values above 0 would be closer to positive,