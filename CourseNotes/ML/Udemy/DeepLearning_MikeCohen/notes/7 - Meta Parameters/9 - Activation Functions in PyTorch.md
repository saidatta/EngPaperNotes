aliases: [Activation Functions in PyTorch, PyTorch Activations, Non-Linear Functions]
tags: [Deep Learning, Meta Parameters, Neural Networks, PyTorch]

In this lesson, we **explore** different **activation functions** by **visualizing** their outputs over a range of inputs (rather than training a model). This helps reinforce how each activation behaves—especially around key regions—and shows how PyTorch implements them.

---

## 1. Imports & Basic Setup

We only need:
- **PyTorch** (for activation functions and tensor ops),
- **Matplotlib** (for plotting).

```python
import torch
import matplotlib.pyplot as plt

# Optional: Increase the default font size for better readability
plt.rcParams.update({'font.size': 18})

print("Libraries imported!")
```

---

## 2. Simple Visualizations of Common Activations

We'll create an array of inputs \(x\) from \([-3, 3]\) and plot:
- **ReLU**  
- **Sigmoid**  
- **Tanh**  
- (Feel free to add others!)

### 2.1 Generating Input Range

```python
# Create a 1D tensor from -3 to 3, e.g. 100 points
X = torch.linspace(-3, 3, steps=100)
```

### 2.2 Helper Function: Torch Functional API

One way to call activation functions is via **functional** APIs in the `torch` module (e.g., `torch.relu`, `torch.sigmoid`).

```python
def get_activation_output(act_fun_name, x):
    """
    Dynamically call e.g. torch.relu(x), torch.sigmoid(x), etc.
    using getattr() to map string -> function.
    """
    act_fun = getattr(torch, act_fun_name)  # e.g., torch.relu
    return act_fun(x)  # apply it to x
```

### 2.3 Plot Activations

```python
activation_functions = ["relu", "sigmoid", "tanh"]

plt.figure(figsize=(8,6))

for act_fun in activation_functions:
    y = get_activation_output(act_fun, X)
    plt.plot(X.numpy(), y.numpy(), label=act_fun)

# Optional: add grid lines for x=0, y=0
plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.axvline(0, color='gray', linewidth=1, linestyle='--')

plt.title("Common Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.show()
```

**Observations**:

- **ReLU** is 0 for negative \(x\) and grows linearly for \(x>0\).  
- **Sigmoid** squashes values into \((0,1)\), nearly 0 for large negative inputs, nearly 1 for large positives.  
- **Tanh** is zero-centered \((-1,1)\), saturating near \(\pm 1\).

---

## 3. Comparing Other PyTorch Activations

PyTorch provides **nn**-module-based activations (e.g., `nn.ReLU`) as well as “functional” ones in `torch.nn.functional`. Here we see how to access them with `torch.nn`.

### 3.1 Alternative or Extended Activations

- **ReLU6**: Similar to ReLU, but caps outputs at 6.  
- **LeakyReLU**: Allows a small negative slope, avoiding “dead ReLUs.”  
- **Hardshrink**, **ELU**, etc.: More specialized or less commonly used.

```python
def get_nn_activation(act_fun_name):
    """
    Return the activation function class from torch.nn module,
    e.g. torch.nn.ReLU(), torch.nn.ReLU6(), torch.nn.LeakyReLU(), etc.
    """
    act_class = getattr(torch.nn, act_fun_name)  # e.g. nn.ReLU6
    return act_class()  # instantiate, e.g. nn.ReLU6()
```

```python
X = torch.linspace(-3, 3, steps=100)

# We'll compare these four
act_funs = ["ReLU6", "LeakyReLU", "Hardshrink", "ReLU"]  # or add more

plt.figure(figsize=(8,6))

for af in act_funs:
    func = get_nn_activation(af)  # e.g. nn.ReLU6()
    Y = func(X)
    plt.plot(X.numpy(), Y.detach().numpy(), label=af)

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.axvline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Various torch.nn Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.show()
```

- **ReLU6** saturates at \(y=6\) (but we only plotted -3 to 3).  
- **LeakyReLU** shows a line for negative \(x\) with small slope (e.g. slope=0.01 or 0.1).  
- **Hardshrink** zeroes out values in a small region around 0 (threshold default=0.5).

---

## 4. Extended Range Example: ReLU6

To see **ReLU6**’s **upper bound**, we can plot a bigger input range:

```python
X_big = torch.linspace(-3, 9, steps=100)
relu6 = torch.nn.ReLU6()

Y_big = relu6(X_big)

plt.figure(figsize=(8,6))
plt.plot(X_big.numpy(), Y_big.detach().numpy(), label="ReLU6")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("ReLU6 over [-3, 9]")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.show()
```

**Note** the saturation:  
- For \(x \le 0\), output = 0.  
- For \(0 < x < 6\), output = \(x\).  
- For \(x \ge 6\), output = 6 (capped).

---

## 5. Torch vs. Torch.NN

### 5.1 Functional vs. Module

1. **Functional** (`torch.relu(x)`, `torch.sigmoid(x)`): Takes **input** directly, returns the transformed tensor.  
2. **Module** (`nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`): Returns **an object** (layer) that can be “called” on input.

**Example**:

```python
X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 1) Functional
y1 = torch.relu(X)

# 2) nn Module
relu_module = torch.nn.ReLU()
y2 = relu_module(X)

print("Functional:", y1)
print("Module:    ", y2)
assert torch.allclose(y1, y2), "They should be identical"
```

**Output**: Both forms produce the **same** result. The `nn.Module` version is often used in `forward()` methods of neural network classes, while the functional one is handy for quick computations.

---

## 6. Official List of Activations in PyTorch

For a **full** list, see [PyTorch Docs: Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations). Examples include:
- **ReLU, ReLU6, LeakyReLU, ELU, SELU, CELU, GELU, ...**  
- **Tanh, Sigmoid, Softplus, Softsign**, etc.  

**In practice**, most networks rely heavily on:
- **ReLU** (or a variant) for hidden layers, and
- **Sigmoid** or **Softmax** for output layers in classification tasks.

---

## 7. Further Exploration

### 7.1 Visualize Weighted Combinations

Below is a simple demonstration of a **linear combination** of two inputs passed through an activation function:

```python
# Example: x1 in [-2, 2], x2 fixed, w1, w2 are weights
import numpy as np

x1 = torch.linspace(-2, 2, steps=100)
x2 = 1.2  # fixed for demonstration
w1, w2 = 0.7, -0.4
b = 0.1

# Linear part
z = w1*x1 + w2*x2 + b

# Non-linear activation
relu_out = torch.relu(z)
sigmoid_out = torch.sigmoid(z)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x1.numpy(), z.numpy(), 'r', label="z = w1*x1 + w2*x2 + b")
plt.title("Linear Output")
plt.xlabel("x1")
plt.ylabel("z")
plt.legend()

plt.subplot(1,2,2)
plt.plot(x1.numpy(), relu_out.numpy(), 'b', label="ReLU(z)")
plt.plot(x1.numpy(), sigmoid_out.numpy(), 'g', label="Sigmoid(z)")
plt.title("Non-linear Outputs")
plt.xlabel("x1")
plt.ylabel("activation")
plt.legend()

plt.tight_layout()
plt.show()
```

Try **changing**:
- \((w1, w2, b)\) values  
- Different **activation**  
- Ranges of **x1**/**x2**  

**Observe** how the shape of the output changes.

---

## 8. Conclusion

- **Visualization** of activation functions highlights the **range** and **non-linearity** each provides.
- PyTorch offers **functional** (`torch.*`) and **module**-based (`nn.*`) activations.
- Although there are many activations, **ReLU** (and variants) + **Sigmoid** (for binary output) or **Softmax** (for multi-class) remain the **most common** in practice.

### Next Steps
- In following lessons, we’ll **train** neural networks with different activations and compare **empirical performance**.

---

## 9. References

1. **PyTorch Docs: Activations**: [Link](https://pytorch.org/docs/stable/nn.html#non-linear-activations)  
2. **Deep Learning Book** (Goodfellow et al.): *Chapter 6* for activation functions.  
3. **Swish, GELU**: Modern variants used in Transformers and other architectures.

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: Activation Functions PyTorch”  
```