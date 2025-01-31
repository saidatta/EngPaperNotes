title: "Measuring Model Performance: Two Perspectives in the World"
tags: [deep-learning, machine-learning, model-performance, signal-detection-theory]
# : 
## 1. Overview and Motivation
In this lecture, we learned that **accuracy** and **loss** (e.g., cross-entropy loss) are often the first metrics used to evaluate model performance. However, they can sometimes fail to capture deeper insights into how the model is actually making decisions, especially in cases of **unbalanced datasets** or when we need to understand **types of classification errors** more precisely.

### Key Takeaways
- Accuracy and loss provide **overall** performance indicators.  
- **Unbalanced datasets** (e.g., cat vs. boat, with far more cat images) can **mask** poor performance in one category.  
- We need additional metrics (e.g., **sensitivity**, **specificity**, **precision**, **recall**, etc.) to quantify the nature of model predictions more precisely.  
- These additional insights often come from **signal detection theory**, which introduces a framework for analyzing **four** different categories of outcomes.

---

## 2. Two Perspectives of the World
### 2.1. Objective (Real) Perspective
- **Reality**: Does the image actually contain a cat (or not)?  
  - **Present**: The cat is actually there.  
  - **Absent**: The cat is not there (a boat, a dog, or any non-cat object).  

### 2.2. Subjective (Model) Perspective
- **Model Output**: The neural network's prediction.  
  - **Predict Cat**: The model says “Cat.”  
  - **Predict Boat** (or “Not Cat”): The model says “Not Cat.”

---

## 3. The Four Categories of Responses
By combining the two perspectives, we get a **2×2 matrix** of possible outcomes. Each outcome corresponds to whether the model is **correct** or **incorrect** about a cat being present:

|                      | **Reality: Cat**   | **Reality: Not Cat**    |
|----------------------|:------------------:|:------------------------:|
| **Model: Cat**       | **Hit** (True Positive)        | **False Alarm** (False Positive)  |
| **Model: Not Cat**   | **Miss** (False Negative)      | **Correct Rejection** (True Negative) |

### 3.1. Hits (True Positives)
- **Definition**: Reality is “Cat” and model predicts “Cat.”  
- **Example**: Image with a cat → model outputs *“Cat.”*  
- **Also Called**: True Positive (TP).

### 3.2. Misses (False Negatives)
- **Definition**: Reality is “Cat,” but model predicts “Not Cat.”  
- **Example**: Image with a cat → model outputs *“Boat.”*  
- **Also Called**: False Negative (FN).

### 3.3. False Alarms (False Positives)
- **Definition**: Reality is “Not Cat,” but model predicts “Cat.”  
- **Example**: Image with a boat → model outputs *“Cat.”*  
- **Also Called**: False Positive (FP).

### 3.4. Correct Rejections (True Negatives)
- **Definition**: Reality is “Not Cat” and model predicts “Not Cat.”  
- **Example**: Image with a boat → model outputs *“Not Cat.”*  
- **Also Called**: True Negative (TN).

---

## 4. Confusion Matrix
When we fill in the **counts** of hits, misses, false alarms, and correct rejections for a given dataset, we get a **confusion matrix**. This matrix is called “confusion” because it highlights the ways in which the model can “confuse” one class with another.

```text
                 Predicted: Cat     Predicted: Not Cat
Reality: Cat        TP (Hit)           FN (Miss)
Reality: Not Cat    FP (False Alarm)   TN (Correct Rejection)
```

For example, if we have:
- **TP** = 50
- **FN** = 10
- **FP** = 8
- **TN** = 932

We would write the confusion matrix as:

```
              Predicted: Cat  |  Predicted: Not Cat
Reality: Cat         50       |          10
Reality: Not Cat     8        |         932
```

---

## 5. Why We Need More Than Accuracy
- **Accuracy** is:
  \[
  \text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
  \]
- **Issue**: It hides **biases**. If 99% of samples are “cat,” a trivial model that **always** predicts “cat” can get high accuracy but never correctly identifies a “boat.”  
- **Solution**: Additional measures (e.g., **Precision, Recall, F1-score, Specificity, Sensitivity**) give more insight:
  - **Recall (Sensitivity)** highlights how many real cats the model catches (`TP / (TP + FN)`).
  - **Precision** highlights of all predicted cats, how many are actually cats (`TP / (TP + FP)`).
  - **Specificity** highlights how well the model identifies non-cats (`TN / (TN + FP)`).

These will be discussed in more detail in subsequent lectures, but the **foundation** is understanding these **four categories** in the confusion matrix.

---

## 6. Code Example: Generating a Confusion Matrix in Python

Below is a **basic example** using *scikit-learn* to compute a confusion matrix. Suppose we have a set of true labels (0 = Not Cat, 1 = Cat) and predicted labels from our model.

```python
```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Example true labels (ground truth)
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])  # 1 = Cat, 0 = Not Cat
# Example predicted labels
y_pred = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1])  # model output

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.2f}")

# Let's parse the confusion matrix for clarity
TN, FP, FN, TP = cm.ravel()  # or if using a different ordering, adjust accordingly
print(f"\nTrue Positives (TP): {TP}")
print(f"False Negatives (FN): {FN}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
```
```  

**Explanation**:  
1. We define **y_true** as the actual labels.  
2. We define **y_pred** as the model’s predictions.  
3. `confusion_matrix(y_true, y_pred)` returns a 2×2 matrix:  
   \[
   \begin{bmatrix}
   \text{TN} & \text{FP} \\
   \text{FN} & \text{TP}
   \end{bmatrix}
   \]  
4. `accuracy_score(y_true, y_pred)` gives the overall accuracy.  
5. We then parse the confusion matrix values into TP, TN, FP, FN for clarity.

---

## 7. Visualization Example
It is often **very helpful** to visualize the confusion matrix. Here’s a quick snippet using **matplotlib** and `ConfusionMatrixDisplay` from scikit-learn:

```python
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Cat", "Cat"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix Visualization")
plt.show()
```
```  

This code will display a heatmap-like matrix indicating each cell’s value (TP, FN, FP, TN). Such visual aids make it easier to spot if a model is heavily biased towards predicting one class.

---

## 8. Practical Example with Unbalanced Data
Consider a dataset where **cats are rare**: only 1% of the images are actually cats, and 99% are not cats. Let’s say we have 10,000 images:

- **Cats**: 100 (1%)  
- **Not Cats**: 9,900 (99%)  

A naive model that **always** predicts “Not Cat” would yield:
- **TP** = 0  
- **TN** = 9,900  
- **FP** = 0  
- **FN** = 100  
- **Accuracy** = (0 + 9,900) / 10,000 = **99%**  

Despite having **99%** accuracy, the model fails completely at identifying the cat class. The **confusion matrix** reveals this glaring problem:

```
                 Predicted: Cat  |  Predicted: Not Cat
Reality: Cat           0         |         100
Reality: Not Cat       0         |        9900
```

---

## 9. Key Terminologies Across Disciplines
- **Hit** = **True Positive**  
- **Miss** = **False Negative**  
- **False Alarm** = **False Positive**  
- **Correct Rejection** = **True Negative**  
- **Confusion Matrix** = 2×2 matrix capturing counts of Hits, Misses, False Alarms, and Correct Rejections  

Different fields (signal detection theory, statistics, psychology, machine learning) might use different terminologies, but the **conceptual** meaning is the same.

---

## 10. Philosophical Note: "Objective" vs. "Subjective" Reality
In **signal detection theory**, we talk about:
- **Objective Reality**: "The cat is *truly* present."  
- **Subjective Reality**: "The observer (or model) perceives a cat."  

In machine learning, the *objective reality* is what is defined by the **labels** in your training/testing dataset (which may or may not be perfectly correct due to labeling errors, but that’s another topic!). The *subjective reality* is the model’s final classification decision.

---

## 11. Next Steps
In future lectures and notes, you’ll learn about metrics derived from these four outcomes to better characterize model performance, such as:
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **F1 Score** = 2 * (Precision × Recall) / (Precision + Recall)  
- **Sensitivity, Specificity, ROC curves, and AUC**  

All of these can help answer questions like:  
- *How well does the model detect a rare class?*  
- *If the model says it’s a cat, how likely is it to be correct?*  
- *How do we handle the trade-off between missing a cat vs. falsely labeling something a cat?*

---

## 12. Summary
1. **Four Key Responses**: Hits, Misses, False Alarms, and Correct Rejections.  
2. **Confusion Matrix**: Summarizes these outcomes quantitatively.  
3. **Accuracy Alone Can Be Misleading**: Especially when data are **unbalanced**.  
4. **Signal Detection Theory**: Provides a systematic way to dissect and analyze predictive performance across various fields.  

**Understanding** these core concepts is essential for designing and interpreting more sophisticated metrics later on.

---

## 13. References and Further Reading
- **Bishop, C.** *Pattern Recognition and Machine Learning*. (Chapter on classification metrics)  
- **Goodfellow, I., Bengio, Y., & Courville, A.** *Deep Learning*. (Section on model evaluation)  
- **Scikit-learn** documentation: [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  
- **Signal Detection Theory** fundamentals: *Green, D. M., & Swets, J. A. (1966)*, *Signal Detection Theory and Psychophysics*.

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your Obsidian vault.  
2. Copy the entire code block above (including the frontmatter `---` lines at the top) into your note.  
3. Make sure you have installed any plugins (if necessary) for syntax highlighting.  
4. Feel free to add internal links to other notes (e.g., `[[Precision, Recall, F1]]`) and reorganize headings as you see fit.

These notes should serve as a **detailed reference** on the topic of confusion matrices, signal detection theory’s fundamental concepts, and why accuracy alone is often insufficient for evaluating model performance, especially in **unbalanced** scenarios.