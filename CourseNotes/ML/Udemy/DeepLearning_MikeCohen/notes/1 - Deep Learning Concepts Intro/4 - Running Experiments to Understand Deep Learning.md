# Introduction
- Experimental scientific approach to deep learning
- Different types of scientists: theoretical, ecological, experimental
- Focus on being experimental scientists with deep learning models
# Types of Deep Learning Researchers/Students
1. Theory-driven:
   - Rely heavily on theory and mathematical development
   - Write scientific publications with many formulas
   - May not need to run any data
2. Ecological:
   - Take existing models and apply them to different datasets
   - Learn about transfer learning
3. Experimental:
   - Systematically modify model parameters and observe results
   - Run experiments, collect data, and empirically determine best parameters
   - Main approach in this course
# Parametric Experiments
- Simple concept: repeat an experiment multiple times while systematically manipulating variables
- Example: manipulate learning rate of a model and observe its impact
- Independent variables: parameters being manipulated (e.g., learning rate, batch size, optimizer, loss fn...)
- Dependent variables: outcomes used to determine model's performance (e.g., accuracy, speed)
# ![[Screenshot 2024-09-30 at 3.31.15 PM.png]]Illustrations of Parametric Experiments
1. One independent variable (e.g., starting gas):
   - Correct answer depends on a certain range of parameters
   - Extreme values of the parameter result in incorrect model learning
2. Two independent variables (e.g., learning rate and number of training epochs):
   - Combinations of learning rate and training epochs impact model's performance
   - Both high number of training epochs and relatively large learning rate needed for success
# Interpretation of Deep Learning Experiments
- Correct interpretation:
  - Best set of parameters for a specific model, architecture, and dataset
  - General patterns likely to be seen in other models and datasets
- Incorrect interpretation:
  - Exact optimal parameter for every single model and dataset

# Limitations of Experimental Approach
1. Feasibility:
   - Small and simple models compute quickly, but larger models take longer
   - Running hundreds of tests may not be feasible for models that take a long time to train
2. Generalizability:
   - Findings from one model may not be reproducible in different architectures or configurations

# Solution
- Use the experimental approach to build intuition and expertise about deep learning in general
- Focus on understanding how deep learning models work, rather than just one specific model
- Science is more of an art than a science

In this video, we learned about the experimental scientific approach to understanding deep learning, the concept of parametric experiments, and how to interpret the results of such experiments. We also discussed the limitations of this approach and the importance of building intuition and expertise about deep learning models in general.