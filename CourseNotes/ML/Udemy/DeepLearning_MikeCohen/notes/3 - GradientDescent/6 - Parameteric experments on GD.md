### Introduction
- Gradient descent is an optimization algorithm used in machine learning and deep learning
- Objective: Minimize the objective function (loss function) to find the best model parameters
- Course: Deep Understanding of Deep Learning by Mike Cohen

### Gradient Descent
- Iterative optimization process
- Equation: 𝑤_𝑡+1 = 𝑤_𝑡 − 𝛼 × ∇𝐹(𝑤_𝑡)
  - 𝑤_𝑡: current model parameters
  - 𝛼: learning rate
  - ∇𝐹(𝑤_𝑡): gradient of the objective function

### Goals of Parametric Experiments
1. Explore gradient descent and understand meta parameters like starting value, learning rate, and the number of training iterations
2. Learn to set up, run, and interpret parametric experiments on gradient descent

### Function Used for Experiments
- Sine wave multiplied by a Gaussian
- Derivative is also defined
- Global minimum around -1.5 and local minimum around 4

### Python Functions for Experiments
- Define python functions for the mathematical functions and their derivatives
- Run vanilla gradient descent using random starting location, specified learning rate, and number of training epochs

### Experiment 1: Systematically Vary Starting Location
- Loop over a range of starting locations and rerun gradient descent using different initial starting locations
- Results: Final guess of the function minimum plotted against initial locations

### Experiment 2: Systematically Vary Learning Rates
- Specify a range of learning rates
- Fix all other parameters in the model (starting location and training epochs)
- Results: Final guess of the function minimum plotted against learning rates
- Observations: A range of learning rates where the model doesn't learn or produces unreliable results, and a range where the model does reasonably well

### Experiment 3: Interaction between Learning Rate and Number of Training Epochs
- Manipulate learning rates and number of training epochs
- Store results in a matrix
- Run gradient descent algorithm inside two for loops (loop over training epochs and loop over learning rates)
- Results: Image showing final guess matrix for different learning rates and training epochs
- Importance of meaningful variable names in multiple for loops

### Results and Observations
- Dark blue color in the matrix plot represents a good result (reached local minimum)
- Yellow color indicates that the algorithm is approaching the local minimum but hasn't reached it yet
- High learning rate and a large number of training epochs lead to good results
- Insufficient training or small learning rate results in suboptimal outcomes
- Interaction between learning rate and the number of epochs observed:
  - A small number of epochs may be sufficient if the learning rate is high enough (faster convergence)

### Alternative Visualization: Line Plot
- X-axis: Learning rates (same as matrix plot)
- Y-axis: Corresponds to the color axis in the matrix plot (objective function value)
- Goal: All lines to converge at -1.4 (correct result)
- The line plot is not the best visualization for this experiment, but it can be useful for other experiments

### Key Takeaways
1. Experiments help build conceptual and coding skills for more complex models
2. Controlling as many factors as possible and systematically varying a small number of variables is essential in scientific experiments
3. Using meaningful variable names in code helps with readability and interpretability

### Conclusion
- Parametric experiments help us understand the behavior of gradient descent
- Visualization techniques can help interpret the results
- More experiments to be conducted throughout the course to deepen understanding of deep