aliases: [MNIST, Feedforward Networks, Deep Learning, Handwritten Digit Recognition]
tags: [Deep Learning, FFN, MNIST, PyTorch, Data Exploration]
## Overview
In this lecture, we explore the **MNIST dataset**, a canonical collection of handwritten digit images. MNIST consists of thousands of labeled digits (0 to 9), historically challenging for classical computer vision, yet now considered almost "solved" by modern deep learning. Nonetheless, MNIST remains an excellent **learning resource** for:
- Getting hands-on experience with neural network training
- Understanding data preprocessing
- Exploring model evaluation and performance metrics

We’ll examine the dataset structure, visualize and interpret sample images, and observe variability among examples of the same digit. By the end of these notes, you’ll be comfortable loading, reshaping, and analyzing MNIST data. We’ll also set the stage for training feedforward neural networks on this dataset in subsequent lectures.

---

## 1. Introduction to MNIST

### 1.1 Historical Background & Significance
- **MNIST** (Modified National Institute of Standards and Technology) is a dataset of handwritten digits from 0 to 9.
- Each image is \(28 \times 28\) pixels (784 total pixels), typically grayscale.
- Origin: widely used since the 1990s as a benchmark in computer vision and machine learning.
- Classical approaches could achieve around **95%** accuracy, which was impressive historically but insufficient for real-world deployment.  
- **Deep Learning** dramatically improved these results, reaching **>99%** accuracy with feedforward networks, **>99.7%** with convolutional neural networks (CNNs), etc.

### 1.2 Why Still Use MNIST?
1. **Educational Value**: MNIST is simpler than many modern datasets but still illustrates the key steps in data handling, model building, and evaluation.
2. **Fast to Train**: Models are typically small and train quickly on MNIST, making it great for demonstrations and prototyping.
3. **"Rite of Passage"**: Familiarizing oneself with MNIST is often the first milestone in deep learning courses and tutorials.

---

## 2. The MNIST Dataset Structure

### 2.1 Data Composition
- The original MNIST dataset includes **70,000 images** (60k for training, 10k for testing).
- Each digit (0–9) appears in varying styles from different writers.
- Some distributions provide **28×28** images stored in a flattened format of size \(1 \times 784\). A label column (digit category) is often appended.

### 2.2 Example Images

Here is a typical row of MNIST digits:

| Example Digit | Handwriting Example | Possible Variation |
|---------------|---------------------|--------------------|
| 0             | ![](https://via.placeholder.com/28) | Some 0s are more oval or more circular |
| 1             | ![](https://via.placeholder.com/28) | Some 1s have a base line or angled top |
| 7             | ![](https://via.placeholder.com/28) | Some 7s include a crossbar, some do not |

Because of handwriting style differences, the same digit can look quite different across samples.  

---

## 3. Basic Exploration in Python

In the snippet below, we assume you have access to a partial MNIST dataset (e.g., **20,000** images instead of the full 70k). The data is stored in CSV-like files with **785 columns**:
1. The **first column** is the **label** (the digit).
2. The **remaining 784 columns** are pixel intensities (values typically in \([0,255]\)).

> **Note**: The code examples use standard libraries: **NumPy**, **Matplotlib**. No deep learning framework is required for this part.

### 3.1 Setup and Imports

```python
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

# Example: Suppose your partial MNIST dataset is available at these paths
train_path = "/content/sample_data/mnist_train_small.csv"
test_path  = "/content/sample_data/mnist_test_small.csv"

# Load them as NumPy arrays
train_data = np.loadtxt(train_path, delimiter=',')
test_data  = np.loadtxt(test_path, delimiter=',')
```

### 3.2 Data Shapes & Label Extraction

```python
print("Train Data Shape:", train_data.shape)
print("Test Data Shape: ", test_data.shape)
```

- Suppose `train_data.shape` returns `(20000, 785)`.
  - **20,000** = number of images
  - **785** = 1 label + 784 pixel columns

We can separate the label from the pixels:

```python
# Separate labels and pixels
train_labels = train_data[:, 0].astype(int)      # First column
train_pixels = train_data[:, 1:].astype(float)   # Remaining columns

test_labels = test_data[:, 0].astype(int)
test_pixels = test_data[:, 1:].astype(float)

print("Train Pixels:", train_pixels.shape)  # (20000, 784)
print("Train Labels:", train_labels.shape)  # (20000,)
```

---

## 4. Visualizing MNIST

### 4.1 Displaying Images in 2D

Each **28×28** image is flattened into a **784-length** vector. To visualize them as actual images:

```python
num_images_to_show = 12
rand_indices = np.random.choice(train_pixels.shape[0], num_images_to_show, replace=False)

plt.figure(figsize=(10, 6))

for i, idx in enumerate(rand_indices):
    plt.subplot(3, 4, i + 1)
    # Reshape the row of 784 pixels into 28x28
    img = train_pixels[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {train_labels[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

- **Result**: A grid of randomly selected MNIST images, each labeled with its corresponding digit.
- Notice **variations** in handwriting style even for the same digit.

### 4.2 Flattened Pixel Values

Internally, the model sees a **1D array** of length 784. For example:

```python
sample_idx = rand_indices[0]
print("Flattened Pixel Values:\n", train_pixels[sample_idx])
print("Label:", train_labels[sample_idx])
```

You might visualize it as a plot of pixel intensity vs. pixel index:

```python
plt.figure()
plt.plot(train_pixels[sample_idx])
plt.title(f"Flattened pixel values of MNIST digit: {train_labels[sample_idx]}")
plt.xlabel("Pixel Index [0-783]")
plt.ylabel("Intensity [0-255]")
plt.show()
```

> **Human Perspective**: Hard to interpret.  
> **Neural Network Perspective**: Learns patterns from these 784 features.

---

## 5. Variability Among the Same Digit

### 5.1 Filtering a Specific Digit
Let’s look at **digit "7"**. We filter out all the indices from `train_labels` that correspond to the digit 7.

```python
digit_of_interest = 7
indices_for_seven = np.where(train_labels == digit_of_interest)[0]
print(f"Number of '{digit_of_interest}'s in the dataset:", len(indices_for_seven))

# Plot first few sevens
num_sevens_to_show = 8
plt.figure(figsize=(8,4))
for i in range(num_sevens_to_show):
    idx = indices_for_seven[i]
    img = train_pixels[idx].reshape(28, 28)
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle(f"Examples of the digit '{digit_of_interest}'")
plt.show()
```

Even among these 8 samples, you’ll see considerable variations in how people write the digit 7.

### 5.2 Pairwise Correlations

We can quantify similarity by calculating **correlation coefficients** between all pairs of "7" images:

```python
# Extract all 7 images
all_sevens_pixels = train_pixels[indices_for_seven]

# Compute correlation matrix
# Flatten each 28x28 image to 784, which it already is
# We'll get a matrix of shape (nSevens, nSevens)
corr_matrix = np.corrcoef(all_sevens_pixels)

plt.figure(figsize=(6,5))
plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix among '7' digits")
plt.show()

# Plot histogram of correlations
all_corr_values = corr_matrix[np.triu_indices(len(all_sevens_pixels), k=1)]
plt.figure()
plt.hist(all_corr_values, bins=50, color='steelblue')
plt.title("Distribution of Pairwise Correlations for '7' Images")
plt.xlabel("Correlation")
plt.ylabel("Count")
plt.show()
```

- The correlation matrix will often have a **moderate peak** around ~0.4 to 0.5, indicating shared structure but also diversity.
- Models must learn robust, invariant features rather than memorize exact pixel patterns.

### 5.3 Average Image

We can also compute the **mean image** of all 7s:

```python
mean_seven = all_sevens_pixels.mean(axis=0).reshape(28, 28)

plt.figure()
plt.imshow(mean_seven, cmap='gray')
plt.title("Average '7' Digit (Mean Over All Samples)")
plt.axis('off')
plt.show()
```

- You might observe a faint "7" shape when the pixel intensities are averaged, revealing consistent stroke patterns.

---

## 6. Key Takeaways

1. **Data Layout**: MNIST images are \(28 \times 28\), often stored as flattened 784-dimensional vectors plus 1 label column.
2. **Human vs. Machine View**:  
   - Humans see a 2D image;  
   - A feedforward neural network sees a 784-length feature vector.  
3. **Variability**: Even the same digit can look quite different in handwriting. Neural networks must learn **invariant** representations to achieve high accuracy.
4. **Correlations**: Pairwise correlation among images of the same digit often reveals moderate similarity, underscoring the need for feature abstraction beyond naive pixel matching.
5. **Why MNIST Remains Useful**: Despite near-saturated performance, it serves as an excellent *learning tool*—easy to process, easy to visualize, fast to train.

---

## 7. Next Steps

- **Feedforward Model Training**: Next, we’ll apply fully connected (feedforward) neural networks to classify MNIST digits.  
- **Building a Pipeline**: We’ll see how to build a training loop, define a loss function (cross-entropy), optimize with backpropagation, and measure accuracy.
- **Scaling Up**: Once you have a feedforward baseline, you can try advanced techniques like **convolutional layers** or **data augmentation**.

---

## 8. References & Further Reading

- [Yann LeCun’s MNIST Page](http://yann.lecun.com/exdb/mnist/) – The original MNIST repository.
- **Goodfellow, Bengio, & Courville** (2016). *Deep Learning*. MIT Press.  
- [Wikipedia: MNIST Database](https://en.wikipedia.org/wiki/MNIST_database) – Historical context and performance benchmarks on MNIST.

---

```markdown
**End of Notes – "Feedforward Networks: MNIST Concepts"**
```