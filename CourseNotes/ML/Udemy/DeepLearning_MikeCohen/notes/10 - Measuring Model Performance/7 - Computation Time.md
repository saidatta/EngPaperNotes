title: "Measuring Model Performance: Computation Time"
tags: [deep-learning, python, performance, timing]
## 1. Overview

Deep Learning models can take **substantial** time to train, making it essential to **measure** how long various parts of your code take to run. In this lecture, we focus on **two methods** of tracking **Computation Time**:

1. **Exact Method**: Using Python’s built-in `time` module for **precise** (sub-millisecond) timing.  
2. **Rough Method**: Using **Google Colab’s** built-in timing features (or a manual stopwatch, if you prefer).

Although this is **not** exclusive to deep learning (no PyTorch-specific code is required), it is particularly **useful** in deep learning contexts where training can be **time-consuming**.

---

## 2. Exact Computation Time Tracking with `time.process_time`

### 2.1. Overview

- The Python standard library includes the `time` module.  
- You can capture **start** time and **end** time around the operations you want to measure.  
- This method offers **sub-millisecond** precision, which is ideal for short or medium-length computations.

### 2.2. Example Code

Below is an example in **Google Colab** (or any Python environment) that trains a toy **MNIST** model. We wrap the **training** loop with `time.process_time()` calls to compute how long each epoch (and the entire experiment) takes.

```python
```python
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1) Preparing the MNIST dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # typical MNIST normalization
])

train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# --- 2) Defining a simple network ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # EXAMPLE: Timer *inside* the function
    # We will record the epoch time for each epoch
    for epoch in range(epochs):
        # Capture the start time of this epoch
        start_time_epoch = time.process_time()
        
        # Training loop
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate test accuracy
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                out = model(X_t)
                preds = out.argmax(dim=1)
                correct += (preds == y_t).sum().item()
                total   += len(y_t)
        test_acc = correct / total

        # End-of-epoch time
        end_time_epoch = time.process_time()
        elapsed_epoch  = end_time_epoch - start_time_epoch

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Test Acc: {test_acc:.3f} | "
              f"Epoch Time: {elapsed_epoch:.3f} sec")

# Instantiate model
model = SimpleNet()

# Timer *outside* the function to measure total experiment time
start_expt_time = time.process_time()
train_model(model, train_loader, test_loader, epochs=5, lr=0.001)
end_expt_time = time.process_time()
elapsed_expt  = end_expt_time - start_expt_time
print(f"\nTotal Experiment Time: {elapsed_expt:.2f} seconds")
```
```

#### Explanation
1. **`time.process_time()`** captures the processor time in **seconds**.  
2. We store the **start** time at the beginning of each epoch and compare it to the **end** time after training + evaluation.  
3. **`elapsed_epoch`** is thus how many seconds that epoch took.  
4. We also track the **total** experiment time outside the function.

---

## 3. Rough Computation Time Tracking in Google Colab

### 3.1. Colab’s Bottom Bar Timer

In **Google Colab**, at the **bottom-left corner**, there is a small green/grey icon or button:
- Clicking on it reveals a small bar that displays the **execution time** of the last cell, e.g., “12s” or “2m 14s.”
- This estimate is **coarse** compared to `time.process_time()`, but convenient.

**Why Use It?**  
- It’s quick for **long-running** cells (e.g., if a model training block can take minutes or hours).  
- You can say “This cell took ~13 minutes” by reading off the timer in the bottom bar.

### 3.2. Cell Execution Timestamps in Colab

Recent versions of Colab also display an **execution time** directly next to the **checkmark** on completed cells:
- A small gray text: “**3s**” or “**1m 20s**” near the top-right corner of each executed cell output.
- This feature gives you a **rough** measure at a glance.

---

## 4. Tips and Best Practices

1. **Combine Both Methods**  
   - Use `time.process_time()` for **short** operations or **precise** sub-second measurements.  
   - Use Colab’s bar timer (or a stopwatch) for **long** operations that span **minutes** or **hours**.

2. **Multiple Timers**  
   - It’s fine to track multiple intervals (e.g., **each epoch** vs. **overall experiment**).  
   - Be careful with variable names (e.g. `start_time_epoch` vs. `start_time_expt`) to avoid overwriting.

3. **When Using Jupyter/Local IDE**  
   - You won’t have the Colab-specific timer bar, but you can still use:
     ```python
     start = time.time()
     # code
     end   = time.time()
     print("Elapsed time:", end - start, "seconds")
     ```
   - Alternatively, you might install Jupyter notebook extensions for timing or use the built-in `%time` or `%timeit` magics.

4. **GPU vs. CPU Timing**  
   - If you’re using a GPU, note that **some operations** are asynchronous. You often need to synchronize (e.g., `torch.cuda.synchronize()`) before capturing time for an accurate measurement.

5. **Large-Scale Jobs**  
   - For **multi-hour** or **multi-day** training jobs, consider using a dedicated logging system (e.g., TensorBoard, Weights & Biases) which also tracks time stamps.

---

## 5. Visual Illustration

### 5.1. Epoch Timing Printout
A typical console output might look like:

```
Epoch 1/5 | Test Acc: 0.920 | Epoch Time: 6.482 sec
Epoch 2/5 | Test Acc: 0.939 | Epoch Time: 5.993 sec
Epoch 3/5 | Test Acc: 0.947 | Epoch Time: 6.101 sec
Epoch 4/5 | Test Acc: 0.951 | Epoch Time: 6.079 sec
Epoch 5/5 | Test Acc: 0.953 | Epoch Time: 6.043 sec

Total Experiment Time: 30.65 seconds
```

### 5.2. Colab Bottom Bar
- You’ll see “**12s**” or “**1m 50s**” in the bar at the bottom-left once the cell completes.  
- A small gray text next to the “checkmark” might read “**12s**.”

---

## 6. Summary

- **Exact Timing** with Python’s `time.process_time()` or `time.time()` gives **precise** sub-second measurements.  
- **Rough Timing** with Google Colab’s built-in bar or cell execution readouts is quick for **long** or **iterative** tasks.  
- **Multiple Timers** can be used in the same script (e.g., for each epoch vs. entire run).  
- Keeping track of **computation time** is crucial to **optimize** your deep learning pipelines and **plan** your experiments effectively.

---

## 7. Further Reading

- Python Docs on `time` module: [https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)  
- Jupyter Notebook Timing Magic: [`%time`, `%timeit`](https://ipython.readthedocs.io/en/stable/interactive/magics.html)  
- TensorBoard for logging training time, steps, GPU usage: [https://www.tensorflow.org/tensorboard/](https://www.tensorflow.org/tensorboard/)  
- Weights & Biases experiment tracking: [https://docs.wandb.ai/](https://docs.wandb.ai/)

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your vault (e.g., `Measuring_Computation_Time.md`).  
2. **Paste** the entire markdown above (including frontmatter `---`).  
3. Add your own internal links or references, and adjust any headings as needed.  

These notes provide a **thorough** introduction to timing computations in Python, focusing on **deep learning** contexts (where epochs can be lengthy).