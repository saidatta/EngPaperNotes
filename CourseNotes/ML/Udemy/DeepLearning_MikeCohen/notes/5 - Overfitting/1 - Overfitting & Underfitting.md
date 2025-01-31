## Table of Contents
1. [[What is Overfitting?]]
2. [[What is Underfitting?]]
3. [[Comparing Overfitting & Underfitting]]
4. [[Detecting Overfitting and Underfitting]]
5. [[Strategies to Avoid Overfitting]]
    - [[Cross-Validation]]
    - [[Regularization]]
6. [[Researcher Overfitting]]
    - [[Researcher Degrees of Freedom]]
    - [[Avoiding Researcher Overfitting]]
7. [[Code Examples]]
    - [[Polynomial Regression Example (Scikit-Learn)]]
    - [[Simple Deep Learning Example (Keras)]]
8. [[Key Takeaways]]

---

## 1. What is Overfitting?
Overfitting occurs when a model **learns not only the underlying patterns** in the training data but also **the noise or random fluctuations**. 

- **Symptoms of overfitting**:
  - Perfect or near-perfect performance on training data.
  - Poor performance on **unseen test data**.
  - The model effectively “memorizes” the training examples instead of learning generalizable patterns.

**Visual Example**:
- Imagine a set of points that roughly follow a straight line.
- A **simple linear model** (few parameters) might capture the trend well.
- A **high-degree polynomial** (many parameters) might pass exactly through every data point in the training set, but fail to predict well on new data.

---

## 2. What is Underfitting?
Underfitting occurs when a model **fails to learn the underlying structure** in the data. It is often **too simple** to capture the complex patterns that may exist.

- **Symptoms of underfitting**:
  - Poor performance on both training and test data.
  - The model is too constrained to capture the real trend or complexity.

**Visual Example**:
- If data clearly follows a curved or more complex relationship,
- A **very simple linear model** might be insufficient, missing the true pattern and providing a large error on both training and test sets.

---

## 3. Comparing Overfitting & Underfitting

| Aspect                  | Overfitting                             | Underfitting                                |
|-------------------------|-----------------------------------------|---------------------------------------------|
| **Sensitivity to noise** | Highly sensitive; starts modeling noise | Less sensitive; does not capture all patterns |
| **Detecting subtle effects** | More likely to detect subtle (but potentially spurious) patterns | Less likely to detect subtle real effects   |
| **Generalizability**      | Reduced – the model does not generalize well to new data | Also reduced – the model is too simplistic to capture the real pattern |
| **Data requirements**     | Over-parameterized models are harder to fit with limited data | Simpler models can often do reasonably well with smaller datasets, but might miss important patterns |

---

## 4. Detecting Overfitting and Underfitting
1. **Visualization** (feasible in low dimensions):
   - Plot the **training data** and the **model predictions** to see how closely the model follows the noise or misses the pattern.
2. **Statistical procedures** (needed for high dimensions):
   - **Cross-validation**, **information criteria** (AIC, BIC), or other model comparison metrics.
3. **Deep learning perspective**:
   - Often rely on **training vs. validation (or test) loss/accuracy** plots.
   - Monitor these during training:
     - If **training loss** keeps dropping but **validation loss** stops improving (or worsens), this is a hallmark of overfitting.

---

## 5. Strategies to Avoid Overfitting
There are two main high-level strategies introduced in the lecture:

### 5.1. Cross-Validation
- **Definition**: Partition the dataset into multiple folds. Iteratively use some folds for training and the remaining fold(s) for validation.
- **Purpose**: 
  - Ensures that every data point is used for both training and validation across different folds.
  - Helps determine if the model can consistently perform well on unseen folds.

### 5.2. Regularization
- **Definition**: Techniques that add constraints or penalties to model parameters to prevent the model from fitting noise.
- **Examples**:
  - *L1/L2 regularization*
  - *Dropout* (in neural networks)
  - *Early stopping*

---

## 6. Researcher Overfitting
### 6.1. Researcher Degrees of Freedom
- **Concept**: The idea that a researcher (data analyst) has many choices in:
  - Data cleaning, organization, and selection
  - Model architectures and hyperparameters
- **Implication**: 
  - Repeatedly tweaking models and data for the best possible fit on a single dataset can lead to a **lack of generalizability**.
  - It’s a form of overfitting at the **research-design level**, not just the model level.

### 6.2. Avoiding Researcher Overfitting
1. **Decide on model architecture in advance**:
   - Choose a standard approach (e.g., known architectures like ResNet for images).
   - Make only minor necessary adjustments.
   - *Feasible for well-studied problems (e.g., image recognition).*
2. **Hold out a final test set**:
   - Do not touch or look at this test set until the very end.
   - Use only your training (and perhaps a validation) set to develop and tweak the model.
   - Evaluate final performance on the held-out set to ensure real-world generalizability.

---

## 7. Code Examples

### 7.1. Polynomial Regression Example (Scikit-Learn)
Below is a **minimal** example in Python demonstrating **overfitting** vs. **underfitting** using polynomial features. This code creates synthetic data, then fits both a **simple linear model** and a **high-degree polynomial model**, illustrating how the polynomial model may overfit.

```python
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic data (roughly linear + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true function: y = 4 + 3x + Gaussian noise

# 2. Fit a simple linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 3. Fit a polynomial regression (degree 10)
poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# 4. Plot the results
X_new = np.linspace(0, 2, 100).reshape(100, 1)
y_lin_pred = lin_reg.predict(X_new)
X_new_poly = poly_features.transform(X_new)
y_poly_pred = poly_reg.predict(X_new_poly)

plt.scatter(X, y, c='blue', label='Training data')
plt.plot(X_new, y_lin_pred, 'g--', label='Linear Model')
plt.plot(X_new, y_poly_pred, 'r-', label='Polynomial Model (Deg=10)')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Underfitting vs Overfitting Example")
plt.show()

# 5. Evaluate error
y_pred_linear = lin_reg.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)

y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"Linear Model MSE on training data: {mse_linear:.4f}")
print(f"Polynomial Model MSE on training data: {mse_poly:.4f}")
```
1. Notice how the **Polynomial Model** might achieve a *very low error on training data*.
2. In practice, test the models on **new data** (or use cross-validation) to confirm that the polynomial model does not overfit.

### 7.2. Simple Deep Learning Example (Keras)
Below is a **simplified** example showing how overfitting can manifest in a deep neural network.

```python
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data (simple function)
np.random.seed(42)
X = np.random.rand(1000, 10)
y = (np.sum(X, axis=1) > 5.0).astype(int)  # binary classification

# Split into train and test
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Build a simple dense model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    validation_split=0.2,
                    verbose=0)  # silent training

# Plot training vs validation accuracy
import matplotlib.pyplot as plt

train_acc = history.history['accuracy']
val_acc   = history.history['val_accuracy']
epochs    = range(1, len(train_acc)+1)

plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
```

- If you see **training accuracy** continues to rise while **validation accuracy** plateaus or declines, that’s a classic **overfitting** pattern.
- **Mitigation**: 
  - Add a dropout layer, 
  - Use L2 regularization, 
  - Reduce network complexity, 
  - Use early stopping, etc.

---

## 8. Key Takeaways
1. **Overfitting** can be dangerous, but it’s nuanced:
   - It can catch subtle patterns, but may capture noise.
   - Reduces model generalizability to new data.
2. **Underfitting** misses the complexity of the data:
   - Leads to poor performance on both training and test data.
3. **Balance** between overfitting and underfitting is critical.
4. **Researcher overfitting** (over-tweaking) is also a real issue:
   - Using many degrees of freedom in data preparation and model design can overfit results to one particular dataset.
5. **Strategies**:
   - **Cross-validation** and **regularization** are standard defenses.
   - **Hold-out test sets** and final evaluations are crucial for real-world validation.

---

> **Tip**: In Obsidian, you can create individual notes for each topic above, linked by double brackets. For instance:  
> - [[What is Overfitting?]] -> A dedicated note explaining overfitting in detail.  
> - [[Cross-Validation]] -> A dedicated note about types of cross-validation (k-fold, stratified k-fold, etc.).  

---

**Further Reading & Upcoming Topics**:
- **Transfer Learning**: Pre-trained architectures like ResNet, BERT, etc., which can mitigate overfitting by using features learned from large, diverse datasets.  
- **Advanced Regularization**: Dropout, Data Augmentation, Batch Normalization, Early Stopping, etc.  
- **Hyperparameter Tuning**: Techniques (e.g., GridSearch, RandomSearch, Bayesian Optimization) to systematically find appropriate model parameters without overfitting.

---

**End of Notes**.  
Use these concepts and code examples as starting blocks for experimentation.  
Remember: Always keep a **held-out test set** or use **cross-validation** to confirm the generalizability of your models.