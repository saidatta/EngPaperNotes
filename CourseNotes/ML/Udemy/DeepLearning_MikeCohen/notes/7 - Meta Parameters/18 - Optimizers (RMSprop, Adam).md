aliases: [RMSprop, Adam, Adaptive Learning Rates]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Optimizers, Neural Networks]

In previous lessons, we introduced **Momentum** as an improvement over vanilla SGD. Now, we explore **RMSprop** and **Adam**, two popular **adaptive learning rate** techniques:

1. **RMSprop**: Scales the learning rate by a running average of the **recent squared gradients**.  
2. **Adam**: Combines **Momentum** and **RMSprop** to adapt both the **direction** and **magnitude** of updates.

**Adam** (Adaptive Momentum Estimation) is often the default choice for training modern deep neural networks.

---
## 1. Recap: Gradient Descent Family

1. **Vanilla SGD**:  
   $\mathbf{w} \leftarrow \mathbf{w} - \eta \,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w})$
2. **Momentum**:  
   $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w})$
   $\quad\Longrightarrow\quad$
   $\mathbf{w} \leftarrow \mathbf{w} - \eta \,\mathbf{v}_t$
3. **RMSprop** & **Adam**: Combine idea of **exponential moving averages** of past gradients and/or squared gradients to **adapt** the effective learning rate over time and per parameter.

---
## 2. RMSprop

### 2.1 Core Idea

RMSprop stands for **Root Mean Square** Propagation. It maintains an **exponential moving average** of the **squared** gradients (rather than just the gradients).

\[
\begin{aligned}
& s_t = \beta \, s_{t-1} + (1 - \beta)\, \bigl(\nabla_{\mathbf{w}}\mathcal{L}\bigr)^2
\\
& \mathbf{w} \;\leftarrow\; \mathbf{w} \;-\; \frac{\eta}{\sqrt{s_t} + \varepsilon} \,\nabla_{\mathbf{w}}\mathcal{L} 
\end{aligned}
\]

- \(\beta \approx 0.9\) – a typical smoothing factor for squared gradients.  
- \(\varepsilon \approx 10^{-8}\) – a small constant to avoid division by zero.  
- \(\eta\) – the (initial) learning rate, though RMSprop *adapts* it per parameter dimension.

### 2.2 Interpretation

- **Scaling** the update by \(1 / \sqrt{s_t}\) means:  
  - If gradients are **large** for a parameter, \(s_t\) becomes **large**, hence step size is **smaller**.  
  - If gradients are **small**, we speed up learning for that parameter by making steps **bigger**.  
- Essentially, RMSprop **balances** step sizes based on how frequently each weight experiences large vs. small gradients.

### 2.3 RMS vs. Variance

- **RMS** is close to **standard deviation** for mean-centered data.  
- RMS in this context \(\approx\) “magnitude” or “energy” of recent gradients.  

---

## 3. Adam: Adaptive Momentum Estimation

### 3.1 Why Adam?

Adam combines:
1. **Momentum** on gradients (first moment)  
2. **RMSprop** on squared gradients (second moment)

Hence, it adaptively adjusts both:
- The **direction** (via accumulated momentum of gradients).  
- The **learning rate** (via accumulated squared gradients).

### 3.2 Formula

Adam maintains two **exponential moving averages**:

\[
\begin{aligned}
& v_t = \beta_1\, v_{t-1} + (1-\beta_1)\,\nabla_{\mathbf{w}}\mathcal{L}_t
\\
& s_t = \beta_2\, s_{t-1} + (1-\beta_2)\,\bigl(\nabla_{\mathbf{w}}\mathcal{L}_t\bigr)^2
\\[6pt]
& \hat{v}_t = \frac{v_t}{1 - \beta_1^t}, 
\quad
\hat{s}_t = \frac{s_t}{1 - \beta_2^t}
\\[6pt]
& \mathbf{w} \;\leftarrow\; \mathbf{w} \;-\;
\eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \varepsilon}
\end{aligned}
\]

- \(\beta_1 \approx 0.9\) – momentum parameter for 1st moment.  
- \(\beta_2 \approx 0.999\) – smoothing parameter for 2nd moment (RMS).  
- \(\hat{v}_t, \hat{s}_t\) are **bias-corrected** versions, ensuring they are **unbiased** estimators early in training.  
- \(\varepsilon \approx 10^{-8}\) again prevents division by 0.

### 3.3 Typical Defaults

- **\(\eta \approx 10^{-3}\)**  
- **\(\beta_1 = 0.9\)**, **\(\beta_2 = 0.999\)**, **\(\varepsilon=10^{-8}\)**

### 3.4 Big Picture

Adam is effectively **Momentum + RMSprop**. It’s widely regarded as the **go-to** optimizer for modern deep learning tasks.

---

## 4. Practical Notes

1. **Adam** typically **converges faster** than vanilla SGD for complex, high-dimensional problems.  
2. **RMSprop** or **Momentum** might suffice for simpler tasks or smaller networks.  
3. Even with these adaptive methods, **learning rate** is still a crucial hyperparameter:
   - Some problems do better with `lr=1e-4`; others might prefer `lr=1e-3`.  
4. **PyTorch Implementation**:
   - `torch.optim.RMSprop(model.parameters(), lr=..., alpha=..., eps=..., momentum=...)`  
   - `torch.optim.Adam(model.parameters(), lr=..., betas=(beta1,beta2), eps=...)`

---

## 5. Example (PyTorch)

```python
import torch.optim as optim

model = MyModel()

# RMSprop
optimizer_rms = optim.RMSprop(
    model.parameters(), 
    lr=0.001, alpha=0.9, eps=1e-08
)

# Adam
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08
)

# Train loop snippet
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer_adam.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer_adam.step()
```

---

## 6. Summary & Comparisons

| Optimizer | Key Idea                         | Pros                             | Cons                                   |
|-----------|----------------------------------|-----------------------------------|----------------------------------------|
| **SGD**   | Plain gradient descent           | Simple, can be good for simpler tasks | Can be slow, sensitive to learning rate |
| **Momentum** | Average gradients (velocity)  | Helps pass small bumps, faster in ravines | 1 extra param (\(\beta\)), may overshoot |
| **RMSprop** | Scales LR by recent squared grads | Adapts step size per parameter     | Ignores gradient sign; focuses on magnitude |
| **Adam**  | Momentum + RMSprop in one        | Often best default choice; robust  | 2 momentum hyperparams (\(\beta_1,\beta_2\)) |

**Adam** is currently the **most popular** optimizer for complex tasks. But sometimes, plain SGD or momentum-based SGD can match or surpass Adam on certain problems if carefully tuned.

---

## 7. What Next?

- Experiment with different **optimizers** on your dataset:
  - Compare training curves for **SGD**, **SGD+Momentum**, **RMSprop**, **Adam**.  
  - Track **training speed** and **final accuracy**.  
- Tweak hyperparameters (\(\eta\), \(\beta\)s) to see **convergence** differences.

Remember: All these methods still fundamentally **descend** the gradient. They just use different **adaptive** strategies to pick the **step size** in each dimension.

---

## 8. References

- [**Adam** Paper: Kingma & Ba, 2014. *Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)  
- [**RMSprop** Reference, Geoff Hinton’s Coursera Lecture (Slide 29)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)  
- [**Deep Learning Book**, Goodfellow et al., Section on Optimization](https://www.deeplearningbook.org/)  
- PyTorch Docs: [**torch.optim**](https://pytorch.org/docs/stable/optim.html)

---

**Created by**: [Your Name / Lab / Date]  
**Lecture Reference**: “Meta parameters: Optimizers (RMSprop, Adam)”  
```