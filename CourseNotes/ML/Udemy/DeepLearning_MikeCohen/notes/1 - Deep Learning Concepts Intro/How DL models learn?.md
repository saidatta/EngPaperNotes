
Tags: #forwardpropagation, #backwardpropagation

# Introduction
-   Deep learning models learn through a process involving forward propagation and backward propagation, also known as backpropagation or backprop.
-   In this note, we'll use an analogy to illustrate these concepts, keeping in mind that all analogies eventually break down and do not perfectly reflect reality.

# Peanut Butter and Jelly Sandwich Analogy

1.  **Ingredients and Weights**
    -   Ingredients:
        -   Bread (X0): no weight
        -   Peanut Butter (X1): weight W1
        -   Jelly (X2): weight W2
    -   The weights (W1 and W2) represent the amounts of peanut butter and jelly in the sandwich.
2.  **Forward Propagation**
    -   The process of combining the weighted ingredients to create a sandwich (the output).
    -   This is represented by the equation: Output = W1 * X1 + W2 * X2.
    -   In a deep learning model with multiple layers, forward propagation involves moving from the input data (ingredients) through the layers to the output.
3.  **Backward Propagation (Backprop)**
    -   The process of adjusting the weights based on feedback (error) from the output.
    -   The error signal is used to adjust the weights, propagating back through the model from the output to the input.
    -   The specific adjustments made to the weights are determined by mathematical concepts like calculus and gradient descent, which we will explore later in the course.

# Company Analogy: PB&J Sandwich Business

1.  **Company Structure**
    -   Raw ingredients: bread, peanut butter, jelly
    -   Kitchen staff: combine ingredients to create sandwiches
    -   Marketing department: promote and advertise sandwiches
    -   Owner/CEO: oversees the company and aims to maximize profit
2.  **Forward Propagation**
    -   Money, services, and labor flow through the company, from raw ingredients to the kitchen staff, marketing department, and finally the CEO.
    -   The outcome (profit) is calculated based on the combined efforts of each department.
3.  **Backward Propagation**
    -   When there's a mismatch between expected and actual profit, the CEO identifies this as an error.
    -   The error message is communicated back through the company, from the CEO to the marketing department and then to the kitchen staff.
    -   Each department makes adjustments to improve their performance, without needing to know the specifics of the other departments' operations.

# Key Takeaways

-   Forward propagation is the flow of information from the input data through the layers of a deep learning model to produce an output.
-   Backward propagation (backprop) is the process of using an error signal to adjust the weights of the model, flowing back from the output through the layers to the input.
-   The specific adjustments made to the weights are unique to each individual node and layer in the model.