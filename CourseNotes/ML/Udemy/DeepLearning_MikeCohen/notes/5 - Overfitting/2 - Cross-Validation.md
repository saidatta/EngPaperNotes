## Table of Contents
1. [[Introduction to Cross-Validation]]
2. [[Terminology: Training, Dev (Hold-Out), and Test Sets]]
3. [[How Cross-Validation Works]]
4. [[The Cross-Validation Cycle]]
5. [[K-Fold Cross-Validation]]
6. [[Key Assumption: Independence of Training and Test Data]]
7. [[Is Overfitting Always Bad?]]
8. [[Code Examples]]
   - [[Scikit-Learn K-Fold Example]]
   - [[Deep Learning with Dev/Test Split]]
9. [[Key Takeaways]]

---

## 1. Introduction to Cross-Validation
**Cross-validation** is a technique for estimating how well a predictive model (including deep learning models) generalizes to new, unseen data. The general idea is:

- **Train** your model on one subset of data.
- **Evaluate** it on another subset of data that the model **never sees** during training.
- **Refine** model choices and hyperparameters by observing the performance on this unseen data.

By doing so, cross-validation aims to reduce the risk of **overfitting**, which is training on noise or idiosyncrasies in a specific dataset.

---

## 2. Terminology: Training, Dev (Hold-Out), and Test Sets
In many practical deep learning projects, we often split our data into three parts:

1. **Training set (e.g., 80%)**  
   - Used to **fit** or **train** the model’s parameters (weights in a neural network).
   - The model iteratively optimizes weights by minimizing a loss function.

2. **Development set (Dev set) / Hold-out set (e.g., 10%)**  
   - Used to **fine-tune** model decisions **without retraining** on it.  
   - Helps guide architecture and hyperparameter adjustments (e.g., learning rate, regularization, number of layers).

3. **Test set (e.g., 10%)**  
   - Used **only once** at the very end to **estimate final performance**.
   - **Crucially**, no training or hyperparameter tuning happens on the test set.

In practice, the percentages can vary (e.g., 70/15/15, 90/5/5, etc.), depending on **data size** and **project needs**.

---

## 3. How Cross-Validation Works
1. **Separate** your dataset into the three subsets mentioned above.
2. **Train** the model on the **training** subset only.
3. **Evaluate** on the **dev** (hold-out) subset to see if:
   - The model is overfitting the training data (training accuracy \( \gg \) dev accuracy).
   - The model underperforms across the board (low training and dev accuracy).
4. **Adjust** model hyperparameters and architecture using feedback from dev set performance.
5. Repeat steps 2–4 as needed.
6. **Finally**, evaluate on the **test** set.

---

## 4. The Cross-Validation Cycle
![[CrossValidationCycle.png]]

*(**Illustrative figure**: The cycle of training vs. dev set evaluation)*

1. **Train** → Model is fit on **training** set.
2. **Dev** → Evaluate on **dev** set (no parameter updates here).
3. **Adapt** → Change architecture/hyperparameters as needed.
4. **Repeat** → Continue until satisfied.

> **Researcher Overfitting**:  
> - Each time you adjust your model based on the dev set performance, you “overfit” to both the training set and the dev set as a whole.  
> - This is why the **test set** must remain untouched until the final model is chosen.

---

## 5. K-Fold Cross-Validation
### 5.1. Concept
In **K-fold cross-validation** (common in traditional machine learning, less so in deep learning due to large dataset sizes and training costs):

1. Split the data into **K** equally sized “folds” (e.g., **K = 10**).
2. For each fold:
   - Treat that fold as the **test** set.
   - Train on the remaining **K-1** folds (combined).
   - Compute performance metrics (e.g., accuracy, MSE).
3. **Aggregate** the performance metrics across the **K** folds.

### 5.2. Purpose
- Provides a **more robust** estimate of model performance.
- Particularly useful when data is **scarce**.

> **Note**: In **deep learning**, the process of training is often computationally intensive, so **K-fold** CV may be less common. Instead, standard practice is typically a **train/dev/test** split.

---

## 6. Key Assumption: Independence of Training and Test Data
Cross-validation hinges on the assumption that **dev/test sets are independent** (or at least uncorrelated) from the training set. This independence assumption ensures that:

- **Performance on the dev/test** set reflects true generalization ability.
- The model is not “learning” from any patterns that inadvertently overlap between training and dev/test sets.

### 6.1. Valid Example
- **Image classification** of pets (cats, dogs, birds, etc.).  
  - Training set images come from certain users.  
  - Dev/test set images come from *entirely different* users.  
  - Minimizes correlation since the pictures (and owners) do not overlap.

### 6.2. Violating the Assumption
1. **Face Age Estimation**:  
   - Training set: Person A’s face images.  
   - Test set: Person A’s siblings.  
   - Problem: Strong genetic correlation; the data is *not* truly independent.

2. **Housing Price Predictions**:  
   - Training set: Random houses in a city.  
   - Test set: Neighboring houses.  
   - Problem: Neighborhood houses are highly correlated in price, location, features.

When the assumption is violated, you risk **inflating** dev/test performance in ways that **do not** generalize to genuinely new data.

---

## 7. Is Overfitting Always Bad?
The instructor emphasizes that **overfitting** can be powerful:

- **Scenario**: A model for predicting home prices in **City A** only.  
  - If your goal is solely to predict home prices in City A, “overfitting” specifically to that city can achieve **extremely high accuracy**.  
  - This might be perfectly acceptable or even desirable for that context.

**Conclusion**:  
- **Overfitting** becomes problematic **only** if the **goal** is to generalize broadly to new or different contexts.  
- **If** the scope is very narrow and well-defined, fitting closely to the training data’s domain may be beneficial.

---

## 8. Code Examples

### 8.1. Scikit-Learn K-Fold Example
Below is a small illustration of **K-fold cross-validation** using Scikit-Learn’s built-in utilities. This code randomly generates data, fits a simple model, and performs K-fold cross-validation.

```python
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(1000)  # True: y = 4 + 3x

# Define a simple model
model = LinearRegression()

# Define K-Fold parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)

print("MSE Scores for each fold:", -scores)  # Convert negative MSE to positive
print("Average MSE:", -scores.mean())
```

1. **KFold(n_splits=5)**: Splits the data into 5 folds.  
2. **cross_val_score**: Automates the process of training on 4 folds and testing on the remaining fold, repeating for all folds.  
3. **Scoring**: Using negative MSE in Scikit-Learn is a quirk; it returns negative so that “higher is better.” We negate it back.

### 8.2. Deep Learning with Dev/Test Split
A **typical** deep learning workflow might look like this in **Keras**:

```python
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Synthetic data
X = np.random.rand(5000, 20)
y = (X.sum(axis=1) > 10).astype(int)  # Binary classification

# Train/Dev/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build a simple model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train on Training set ONLY
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=32,
                    validation_data=(X_dev, y_dev),
                    verbose=0)

# Evaluate final performance on Test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f}")
```

- **Validation Data** here plays the role of the **dev** set, guiding us in when to stop training or how to adjust hyperparameters.
- **Test** set is used only after finalizing the model.

---

## 9. Key Takeaways
1. **Cross-Validation**:
   - A robust strategy to reduce the chances of overfitting and estimate generalization.
   - Involves splitting data (train/dev/test) carefully to **avoid** data leakage.
2. **K-Fold**:
   - Common in machine learning; less in deep learning due to training cost/time.
3. **Independence**:
   - Holding out truly *unrelated* data is crucial for genuine performance estimates.
4. **Overfitting Nuance**:
   - Overfitting can be **desired** if the application domain is narrow and well-defined.
   - Be sure you understand your end-goals and scope before labeling overfitting as “bad.”

> **Reminder**: In real-world deep learning projects, always keep a **final** test set aside, **untouched**, to ensure you have a **clean estimate** of your model’s performance.

---

**End of Notes**.  
Use these concepts to structure your cross-validation approach in deep learning, remembering that the ultimate goal is often **generalizability**—unless your application domain specifically values hyper-focused modeling on a single dataset or location.