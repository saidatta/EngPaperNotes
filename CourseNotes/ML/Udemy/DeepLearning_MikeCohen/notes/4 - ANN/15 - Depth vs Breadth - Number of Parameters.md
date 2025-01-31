Below is an extensive set of Obsidian notes in Markdown format covering the lecture “ANN: Depth vs. Breadth: Number of Parameters.” These notes explain the key concepts (depth, breadth, and how they affect the total number of parameters), show manual and programmatic methods to count nodes and trainable parameters, include complete code examples in PyTorch, and provide visualizations and discussion. You can copy and paste the following into your Obsidian vault.


> **"In this video, I'm going to tell you about the two dimensions of complexity of deep learning models—depth versus breadth—and their implications for the total number of parameters."**  
>  
> **Overview:**  
> - **Depth:** Refers to the number of layers (typically the number of hidden layers) in a network.  
> - **Breadth (or Width):** Refers to the number of nodes in a layer.  
>  
> Despite seeming counterintuitive, deeper networks (with more layers but fewer nodes per layer) can sometimes have fewer total trainable parameters than shallower but wider networks.

---

## Table of Contents

1. [Key Concepts & Terminology](#key-concepts--terminology)
2. [Manual Counting of Parameters: An Example](#manual-counting-of-parameters-an-example)
3. [PyTorch Model Definitions](#pytorch-model-definitions)
    - [Wide Network Example](#wide-network-example)
    - [Deep Network Example](#deep-network-example)
4. [Counting Parameters in PyTorch](#counting-parameters-in-pytorch)
    - [Using `named_parameters()`](#using-named_parameters)
    - [Counting Biases to Infer Number of Nodes](#counting-biases-to-infer-number-of-nodes)
5. [Model Summary with `torchsummary`](#model-summary-with-torchsummary)
6. [Discussion & Insights](#discussion--insights)
7. [References](#references)

---

## Key Concepts & Terminology

- **Depth:**  
  The number of layers (or hidden layers) in a network. For example, a network with three hidden layers is considered deeper than one with a single hidden layer.

- **Breadth (or Width):**  
  The number of nodes (neurons) in a given layer. A network with 15 nodes in its hidden layer is said to be broader than one with 4 nodes.

- **Trainable Parameters:**  
  Include all the weights connecting nodes between layers **and** the bias terms (each node typically has one bias).  
  - **Example Calculation (for a fully connected layer):**  
    If a layer has \(m\) inputs and \(n\) outputs, then:
    - **Weights:** \(m \times n\)
    - **Biases:** \(n\)
    - **Total Parameters:** \(m \times n + n\)

- **Implications:**  
  A wide (shallow) network might have many nodes per layer and thus more parameters per layer, whereas a deep network with more layers but fewer nodes per layer can be surprisingly compact in terms of trainable parameters while still capturing complex, abstract representations.

---

## Manual Counting of Parameters: An Example

Consider two networks with the same input and output dimensions but different hidden architectures:

### Wide Network
- **Architecture:**  
  - **Input:** 2 features  
  - **Hidden Layer:** 4 nodes  
  - **Output:** 3 nodes

- **Parameter Count Calculation:**
  - **Input-to-Hidden:**  
    - Weights: \(2 \times 4 = 8\)  
    - Biases: \(4\)  
    - **Subtotal:** \(8 + 4 = 12\)
  - **Hidden-to-Output:**  
    - Weights: \(4 \times 3 = 12\)  
    - Biases: \(3\)  
    - **Subtotal:** \(12 + 3 = 15\)
  - **Total Parameters:** \(12 + 15 = 27\)

### Deep Network
- **Architecture:**  
  - **Input:** 2 features  
  - **Hidden Layer 1:** 2 nodes  
  - **Hidden Layer 2:** 2 nodes  
  - **Output:** 3 nodes

- **Parameter Count Calculation:**
  - **Input-to-Hidden1:**  
    - Weights: \(2 \times 2 = 4\)  
    - Biases: \(2\)  
    - **Subtotal:** \(4 + 2 = 6\)
  - **Hidden1-to-Hidden2:**  
    - Weights: \(2 \times 2 = 4\)  
    - Biases: \(2\)  
    - **Subtotal:** \(4 + 2 = 6\)
  - **Hidden2-to-Output:**  
    - Weights: \(2 \times 3 = 6\)  
    - Biases: \(3\)  
    - **Subtotal:** \(6 + 3 = 9\)
  - **Total Parameters:** \(6 + 6 + 9 = 21\)

> **Observation:**  
> The deeper network (with two hidden layers and fewer nodes per layer) has **21 parameters**, while the wide network (with one hidden layer but more nodes) has **27 parameters**.

---

## PyTorch Model Definitions

Below, we define both the wide and deep networks in PyTorch. For illustration, these models are kept simple (without non-linear activation functions for the purpose of counting parameters, though in practice you would include activations).

### Wide Network Example

```python
import torch
import torch.nn as nn

class WideNetwork(nn.Module):
    def __init__(self):
        super(WideNetwork, self).__init__()
        # Architecture: 2 inputs -> 4 hidden nodes -> 3 outputs
        self.fc1 = nn.Linear(2, 4, bias=True)   # Input to Hidden
        self.fc2 = nn.Linear(4, 3, bias=True)   # Hidden to Output

    def forward(self, x):
        # In practice, you would apply activation functions here (e.g., ReLU)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the wide network and print it
wide_net = WideNetwork()
print("Wide Network Architecture:")
print(wide_net)
```

### Deep Network Example

```python
class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()
        # Architecture: 2 inputs -> 2 hidden nodes -> 2 hidden nodes -> 3 outputs
        self.fc1 = nn.Linear(2, 2, bias=True)   # Input to Hidden1
        self.fc2 = nn.Linear(2, 2, bias=True)   # Hidden1 to Hidden2
        self.fc3 = nn.Linear(2, 3, bias=True)   # Hidden2 to Output

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Instantiate the deep network and print it
deep_net = DeepNetwork()
print("Deep Network Architecture:")
print(deep_net)
```

---

## Counting Parameters in PyTorch

### Using `named_parameters()`

You can loop through the named parameters of a model to sum up the total number of trainable parameters. In this example, we show how to count parameters for both networks.

```python
def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params} parameters")
    print(f"Total trainable parameters: {total_params}")
    return total_params

print("Parameters in Wide Network:")
wide_total = count_parameters(wide_net)

print("\nParameters in Deep Network:")
deep_total = count_parameters(deep_net)
```

*Expected Output:*  
- **Wide Network:** Should print a total of 27 trainable parameters.  
- **Deep Network:** Should print a total of 21 trainable parameters.

### Counting Biases to Infer Number of Nodes

Because each node (except inputs) typically has one bias parameter, you can count bias parameters to know the total number of nodes in the network.

```python
def count_bias_nodes(model):
    node_count = 0
    for name, param in model.named_parameters():
        if "bias" in name and param.requires_grad:
            # The number of elements in the bias corresponds to the number of nodes in that layer
            node_count += param.numel()
    print(f"Total number of nodes (via biases): {node_count}")
    return node_count

print("\nCounting nodes in Wide Network:")
wide_nodes = count_bias_nodes(wide_net)

print("Counting nodes in Deep Network:")
deep_nodes = count_bias_nodes(deep_net)
```

*Expected Outcome:*  
- The wide network should show a total node count that reflects its hidden layer and output nodes.  
- The deep network will count the nodes in both hidden layers and the output.

---

## Model Summary with `torchsummary`

For a concise summary of the model architecture (including per-layer parameter counts), you can use the `torchsummary` package.

> **Note:** Install the package if needed using `pip install torchsummary`.

```python
from torchsummary import summary

# For the wide network, the input size is 2 (features)
print("\nWide Network Summary:")
summary(wide_net, input_size=(2,))

print("\nDeep Network Summary:")
summary(deep_net, input_size=(2,))
```

The summary displays each layer, the output shape at that layer, and the number of parameters. Although these models are small (and their memory footprint may be near 0 MB), the technique scales to larger networks (such as convolutional neural networks).

---

## Discussion & Insights

- **Depth vs. Breadth:**  
  - **Breadth** (wider layers) increases the number of nodes per layer and typically the number of parameters (if you have many neurons in one layer, there are more connections to the next layer).  
  - **Depth** (more layers) increases the level of abstraction but may not necessarily increase the total number of parameters if the layers are kept narrow.
  
- **Trade-offs:**  
  - A **deeper** network (even with fewer parameters) might learn more abstract representations.  
  - A **wider** network might have more capacity (more parameters) but may require more data to train effectively and is prone to overfitting if not regularized.

- **Practical Implication:**  
  Understanding these two dimensions can help you design more efficient architectures that balance computational cost with model capacity. In our examples, the deep network had fewer parameters (21) than the wide network (27) even though it had more layers—illustrating that “more depth” does not automatically imply more parameters.

---

## References

- **Deep Learning Books:**  
  - Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press.
- **PyTorch Documentation:**  
  - [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)  
  - [torchsummary](https://github.com/sksq96/pytorch-summary)
- **Lecture Reference:**  
  - Mike X Cohen, “ANN: Depth vs. Breadth: Number of Parameters”

---

*End of Note*

These notes not only cover the theoretical background and manual parameter counting but also provide practical code examples to inspect and understand how depth and breadth affect the overall number of trainable parameters in a deep learning model. Happy exploring and deep learning!