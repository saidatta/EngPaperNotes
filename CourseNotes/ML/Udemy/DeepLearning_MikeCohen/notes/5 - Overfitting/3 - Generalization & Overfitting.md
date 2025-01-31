## Table of Contents
1. [[Definition of Generalization]]
2. [[Generalization Boundaries]]
3. [[Impact of Generalization on Accuracy]]
4. [[Examples of Generalization Boundaries]]
   - [[Predicting Weight from Height & Calories]]
   - [[Predicting House Prices in Multiple Cities]]
5. [[Practical Implications for Data Splits]]
6. [[Code Example: Demonstrating Generalization Boundaries]]
7. [[Key Takeaways]]

---

## 1. Definition of Generalization
**Generalization** refers to a model’s ability to perform well on **unseen data**, i.e., data that **was not** used during the training phase. In a deep learning context, it’s about ensuring that the learned weights and patterns are not **overly specialized** to the idiosyncrasies of the training set.

### Why Generalization Matters
- **Real-World Performance**: You want your model to handle data it hasn’t seen before.  
- **Overfitting Risk**: Without careful attention, models can memorize **noise** in the training dataset, leading to poor performance in new situations.

---

## 2. Generalization Boundaries
A **generalization boundary** defines **which populations or data domains** your model is intended to work with. You do **not** need (and often **cannot**) make a single model handle every conceivable dataset in the universe.

**Example**: You build a model that **predicts weight** from **height** and **daily caloric intake**:
- **Boundary**: The model applies only to **adult humans** (no children, no other animals).
- **Why**: Children have different growth patterns; animals vary drastically in metabolism (mice vs. humans).

**Key Insight**: You decide in advance what “new data” means for your project and ensure your model is **trained and validated** on data within that boundary.

---

## 3. Impact of Generalization on Accuracy
- **Wider Boundaries** often mean **more variability** in the data, which can reduce performance on any **specific** subset.  
- **Narrow Boundaries** (e.g., focusing on a single city or a very specific population) usually yield higher accuracy **within** that domain but might fail when applied outside that boundary.

> **Trade-off**: The more you want your model to generalize to diverse scenarios, the more potential accuracy you often sacrifice in each scenario.

---

## 4. Examples of Generalization Boundaries

### 4.1. Predicting Weight from Height & Calories
**Goal**: A linear or neural network model that takes `(height, daily_calories)` and predicts `weight`.  
- **Boundary**: 
  1. **Adult humans only** (no children, no animals).  
  2. Potentially include multiple ethnicities/countries if desired.  
  3. Specifically excludes children under 18 because growth patterns differ.

**Reasoning**:  
- Children’s physiology is different.  
- Metabolism in non-human animals is drastically different.

### 4.2. Predicting House Prices in Multiple Cities
From the previous lecture examples:
- **One approach**: A single model for **City A, City B, and City C** → Good generalization across all three, at the cost of lower accuracy within each city.  
- **Another approach**: Overfit a model to **City A** → Very high accuracy for City A, might fail in City B/C.  
- **Boundary**: “We only care about City A.” → The fact that it doesn’t generalize to other cities isn’t a problem if that’s outside our scope.

---

## 5. Practical Implications for Data Splits
- Always reflect on **who** or **what** your data represents.  
- Ensure you have **training**, **development (hold-out)**, and **test sets** that are representative of the **entire boundary** you plan to cover.  
- If your boundaries are **adult humans** only, do **not** include children in any dataset split. Conversely, if you want it to generalize to children as well, include children in your training and testing data.

---

## 6. Code Example: Demonstrating Generalization Boundaries

Below is a **minimal** Python example illustrating a scenario with separate populations. We simulate:
- A dataset of **adult** (height, daily_calories, weight)  
- A dataset of **children** (height, daily_calories, weight)  

We train a simple regression model **only** on adults, then see how it performs on children (out-of-boundary). This demonstrates the **drop in accuracy** when going beyond the intended boundary.

```python
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---- 1. Generate synthetic data ----
np.random.seed(42)

# Adults: heights ~ 150-200 cm, daily_calories ~ 1500-3000
adult_heights = 150 + 50 * np.random.rand(500, 1)  # 150 to 200 cm
adult_calories = 1500 + 1500 * np.random.rand(500, 1)  # 1500 to 3000
# Weight (adults) ~ 0.5*height + 0.01*calories + noise
adult_weights = 0.5 * adult_heights[:, 0] + 0.01 * adult_calories[:, 0] + 5 * np.random.randn(500)

# Children: heights ~ 50-130 cm, daily_calories ~ 1000-2000
child_heights = 50 + 80 * np.random.rand(200, 1)
child_calories = 1000 + 1000 * np.random.rand(200, 1)
# Weight (children) could have different relation
child_weights = 0.4 * child_heights[:, 0] + 0.008 * child_calories[:, 0] + 3 * np.random.randn(200)

# Combine into training-like arrays
X_adults = np.hstack([adult_heights, adult_calories])
y_adults = adult_weights
X_children = np.hstack([child_heights, child_calories])
y_children = child_weights

# ---- 2. Train model on Adult data only ----
reg = LinearRegression()
reg.fit(X_adults, y_adults)

# ---- 3. Evaluate within boundary (adults) ----
adult_preds = reg.predict(X_adults)
adult_mse = mean_squared_error(y_adults, adult_preds)

# ---- 4. Evaluate out of boundary (children) ----
child_preds = reg.predict(X_children)
child_mse = mean_squared_error(y_children, child_preds)

print(f"Training MSE (Adults, In-Bound): {adult_mse:.2f}")
print(f"Test MSE (Children, Out-of-Bound): {child_mse:.2f}")

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(X_adults[:, 0], y_adults, color='blue', alpha=0.5, label='Adult Data')
plt.scatter(X_children[:, 0], y_children, color='green', alpha=0.5, label='Child Data')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg?)")
plt.title("Adult vs. Child Data: Generalization Boundaries")
plt.legend()
plt.show()
```

1. **Adults**: We generate training data meant to represent the “**valid boundary**” for our model.  
2. **Children**: This is **not** in the defined boundary; the model was never trained with their data.  
3. **Result**:  
   - Low MSE (mean squared error) on adult data.  
   - Potentially **high** MSE on child data.  
   - **Illustrates** how going beyond boundaries degrades performance.

---

## 7. Key Takeaways
1. **Generalization**: The cornerstone of building machine learning models that actually solve real-world problems.  
2. **Boundaries**: Define them explicitly—decide **where** your model should apply.  
3. **Trade-offs**: Wider boundaries generally mean more data complexity and potential decreases in performance on any specific subset.  
4. **Practical Tips**:  
   - Always check if your training data covers **all** aspects of the domain you care about.  
   - If you only care about one city or one type of subject, limit your training data to that domain (acknowledging out-of-bound use cases will fail).  
   - If you need broader coverage, **expect** lower performance on each individual sub-domain, but **ensure** your training data is **representative** of all sub-domains.

---

**End of Notes**.  
By carefully considering the **boundaries** for where your deep learning model needs to generalize—and by splitting your data accordingly—you’ll be well-prepared to manage the **inevitable trade-offs** between **fitting** and **generalizing**.