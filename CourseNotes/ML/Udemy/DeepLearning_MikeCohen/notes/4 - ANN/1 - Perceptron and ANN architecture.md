## Artificial Neural Networks (ANN)
- A computational model inspired by biological neural networks.
- Composed of artificial neurons or nodes, and connections between nodes carry numerical values.
- ANNs are used in machine learning algorithms to solve complex tasks such as image recognition, speech recognition, natural language processing, and many more.

## Node or Neuron
- Node takes inputs, does some computation on them, and produces an output.
- Node's inputs are numerical values or features and each input has a corresponding weight.
- Node computes the weighted sum of the inputs and adds a bias term.
- The weighted sum is passed through an activation function which provides the final output.
- Example: In an image recognition task, the inputs could be pixel intensities, the weights could be parameters learned by the network during training, the bias is another parameter, and the activation function could be a ReLU (Rectified Linear Unit) or sigmoid function.
## Linear Models
- Common in statistics and machine learning, e.g., regression, general linear models, ANOVA, factor analysis.
- Great for linearly separable problems like predicting house prices based on area, number of rooms, etc., but many problems such as image recognition or text translation cannot be solved through linear models.
## Nonlinear Operations
- Needed for problems that cannot be solved by linear models.
- Example: The sign function, a simple nonlinear function used in the perceptron model.
    - Outputs 1 for any positive numbers and -1 for any negative numbers.
    - The output is either -1 or +1 depending on the sign of the input.
- In deep learning, there are many nonlinear functions such as sigmoid, tanh, ReLU, and softmax.

![[Pasted image 20250131132102.png]]
![[Pasted image 20250131132244.png]]
## Perceptron Model
- A perceptron performs a linear operation, followed by a nonlinear operation.
    - Linear Operation: Dot product between the input numbers and the weight vector.
    - Nonlinear Operation: Passes the result through a nonlinear function, often referred to as an activation function.
- Output of a perceptron: $Y_a = f(W*X + B)$
- Importance of the bias term:
    - Without a bias term, the model is constrained to pass through the origin, which can result in suboptimal results.
    - A bias term allows the model to fit any arbitrary data positioned anywhere in the plane.
    - In practice, the bias term is always included.
![[Pasted image 20250131154231.png]]
## Math of Deep Learning
- Much of the math in deep learning involves computing a linear weighted sum of inputs and passing that value through a nonlinear function.
- Example equation: Y_hat = f(W*X + B), where W*X is a linear operation, and f is a nonlinear function.
![[Pasted image 20250131154046.png]]
## Bias Term
- Also called the intercept in statistics.
- Needed in order to avoid suboptimal results when the line is constrained to pass through the origin.
- The bias term is a constant set to one, and its weight, $W0$, is a parameter that the model optimizes.
- For example, in a simple linear regression model, the bias is the y-intercept of the line, allowing the line to best fit the data.
---
Note: In the notes, X represents inputs, W represents weights, B represents bias, f represents a nonlinear function, and Y_hat represents the output of the model.
### Real-life examples of ANN
Artificial Neural Networks (ANNs) are used in a wide variety of real-world applications. Here are a few examples:

Yes, both GPT (Generative Pre-trained Transformers) models, like GPT-3 and GPT-4, and other large language models (LLMs) are a form of artificial neural networks (ANNs). They specifically fall under the category of transformer-based models, a type of deep learning model that is particularly well-suited for understanding the structure and meaning of human language.

Here's a brief overview:

1. **GPT Models**: GPT models are transformer-based models developed by OpenAI. They are designed to generate human-like text based on the input they receive. They work by predicting the next word in a sentence and can generate coherent and contextually relevant sentences. GPT models have been pre-trained on a diverse range of internet text, but they can be fine-tuned on specific tasks as well.

2. **Transformer Models**: Transformer models, introduced in the paper "Attention is All You Need" by Vaswani et al., are a type of model that uses self-attention mechanisms instead of recurrence (like in Recurrent Neural Networks) or convolution (like in Convolutional Neural Networks). These models are particularly effective for natural language processing tasks because they can understand the context of a word in a sentence by looking at all the other words in the sentence, not just the ones before it.

So, in essence, GPT and other LLMs are not only forms of ANNs but are sophisticated versions that leverage the power of transformer architectures for understanding and generating human language.

| Application Area | Example |
|------------------|---------|
| Image Recognition | ANNs, particularly Convolutional Neural Networks (CNNs), are used for image recognition tasks. Examples include Facebook's automatic tagging feature and Google Photos' ability to recognize and categorize photos into various groups like people, places, or things. |
| Speech Recognition | Recurrent Neural Networks (RNNs), a type of ANN, are used in speech recognition technologies like Siri, Alexa, and Google Assistant. These systems are trained to understand and respond to human speech patterns. |
| Fraud Detection | In the banking and finance industry, ANNs are used to detect unusual patterns or anomalies indicating fraudulent activities. If a series of transactions deviate significantly from a customer's typical behavior, the ANN can alert the bank or block the transactions. |
| Medical Diagnosis | ANNs are used in systems that assist in diagnosing diseases. They can analyze medical images to detect signs of diseases, such as cancerous tumors in mammograms or retinal diseases in eye scans. They can also predict a patient's risk levels based on various health factors. |
| Recommendation Systems | Websites like Amazon, Netflix, and YouTube use ANNs to suggest products, movies, or videos based on a user's past behavior and the behavior of other users with similar tastes. These systems enhance user experience and engagement. |