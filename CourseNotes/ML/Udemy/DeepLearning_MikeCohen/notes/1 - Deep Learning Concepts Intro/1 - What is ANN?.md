### Introduction
-   Deep learning models are a way to transform an input into an output
-   The goal is to provide a high-level overview of deep learning models
-   Artificial Neural Networks (ANN) are at the core of deep learning
### Examples of ANN Applications
1.  Web browser history to predict advertisements, videos, or sites you are likely to click on
2.  Credit card transaction data to predict fraudulent activity
3.  Personal medical history and diagnostic test outcomes to predict diseases
4.  Self-driving cars using cameras to detect pedestrians
5.  Language translation (e.g. English to French)
### ANN Visualization Example
-   Predicting whether a student will pass an exam based on hours studied and hours slept
-   ANN can be used to separate the students who passed (blue dots) from those who failed (red dots)
-   A linear solution like a straight line can be used, but deep learning can also solve nonlinear problems
![[Screenshot 2024-09-30 at 2.59.12 PM.png]]
### ANN Math
-   Prediction: Y' = X1 * W1 + X2 * W2
    -   Y': Prediction about the world (e.g. pass or fail, pedestrian detection, fraud prediction)
    -   X: Data values (can be dozens, thousands, or millions)
    -   W: Weights, importance of each data feature for the output (learned by the model through backpropagation)
-   Nonlinear function (e.g. sigmoid) is added to the linear expression to create a nonlinear model
![[FireShot Capture 002 - Course_ A deep understanding of deep learning (with Python intro) - U_ - cisco.udemy.com.png]]
### ANN Architectures
1.  Data table and predicting an outcome - Feedforward Neural Network (FNN): 
	1. Example: Predicting housing prices based on factors such as square footage, number of bedrooms, and location. In this case, you would use a feedforward neural network to model the relationship between these input features and the output (price).
2.  Images - Convolutional Neural Network (CNN): 
	1. Example: Image classification, such as identifying if an image contains a cat or a dog. The CNN architecture is designed to capture spatial patterns and hierarchies within images, making it highly effective for tasks like this.
3.  Time series or sequence data - Recurrent Neural Network (RNN): 
	1. Example: Predicting the next word in a sentence, given the previous words. In this case, you'd use an RNN architecture that takes into account the order of the words in the sentence and can handle varying lengths of input sequences.
4.  Sequence-to-sequence tasks - RNN with attention mechanisms or Transformer:
	1. Example: Machine translation, where the task is to translate text from one language to another. For this type of problem, you might use an RNN with attention mechanisms or a Transformer model, both of which are designed to handle the complexities of language and effectively capture long-range dependencies within the text.
![[Screenshot 2024-09-30 at 3.10.12 PM.png]]
### Conclusion
-   Deep learning models can be visualized in a variety of ways
-   Different models are suited for different purposes but share similar underlying principles
-   This video provides a big-picture overview of different kinds of deep learning models and some of the math involved