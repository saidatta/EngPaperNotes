# Obsidian Notes for Staff+:

## Overview
Neural networks (NNs) play a vital role in improving the accuracy and personalization of recommendation systems by capturing complex, non-linear patterns in data. This document covers how NNs are used in collaborative filtering, content-based recommendations, and sequential modeling. We also discuss techniques like **autoencoders** and **variational autoencoders (VAEs)**, side information integration, and **deep reinforcement learning**. 

## 1. Applying Neural Networks to Recommendation Systems
### 1.1 Collaborative Filtering with NNs
NNs can be used to model user-item interactions by embedding users and items into continuous vector spaces. These embeddings capture latent features representing user preferences and item characteristics. The embeddings are then combined using multi-layer neural network architectures to predict ratings or interactions.

**Mathematical Formulation**:
Given a user \( u \) and an item \( i \), we represent them with embeddings \( \mathbf{e}_u \) and \( \mathbf{e}_i \). The predicted interaction \( \hat{y}_{ui} \) can be expressed as:
\[
\hat{y}_{ui} = f(\mathbf{e}_u, \mathbf{e}_i; \theta)
\]
where \( f \) is a neural network with parameters \( \theta \) that outputs a prediction.

### 1.2 Content-Based Recommendations
NNs can learn representations of item content such as text, images, or audio to make personalized recommendations. **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** are often used to process these types of data.

**Example**:
- **CNNs** can process images or video content, learning features such as genre or style.
- **RNNs**, particularly **LSTM** networks, can analyze text data like movie descriptions or plot summaries to capture sequential dependencies.

### 1.3 Sequential Models
**RNNs** and **LSTMs** are used to model user interaction sequences over time, such as browsing history or clickstreams. These models capture temporal dependencies and make context-aware, time-sensitive recommendations.

**Mathematical Formulation**:
Given a sequence of user interactions \( x_1, x_2, \ldots, x_T \), the hidden state \( h_t \) of an LSTM is updated as:
\[
h_t = \text{LSTM}(x_t, h_{t-1})
\]
The output \( \hat{y}_{t+1} \) for the next interaction is:
\[
\hat{y}_{t+1} = \sigma(W h_t + b)
\]
where \( \sigma \) is a non-linear activation function, \( W \) and \( b \) are learnable parameters.

### 1.4 Autoencoders and Variational Autoencoders (VAEs)
**Autoencoders** are used for learning low-dimensional representations of data. They consist of:
- **Encoder**: Maps the input data to a latent space.
- **Decoder**: Reconstructs the input from the latent representation.

**VAEs** extend traditional autoencoders by introducing a probabilistic component, modeling the latent space distribution. VAEs can generate new data samples, making them suitable for recommendation systems involving complex, multi-modal user-item interactions.

**Mathematical Formulation of VAEs**:
Given input data \( x \), the encoder outputs \( \mu \) and \( \sigma \) for the latent space distribution \( q(z|x) \):
\[
z \sim \mathcal{N}(\mu, \sigma^2)
\]
The decoder reconstructs the data \( \hat{x} \) using:
\[
\hat{x} = f(z; \theta)
\]
The training objective is the **variational lower bound**:
\[
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
\]

**Python Example**:
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Instantiate the model
model = Autoencoder(input_dim=1000, latent_dim=50)
```

## 2. Advanced Techniques in NN-Based Recommendations
### 2.1 Side Information Integration
NNs can incorporate additional side information like demographic data, location, and social connections. This multi-input approach can leverage **multi-modal architectures** to improve recommendation accuracy.

**Example**:
Combining user embeddings from demographic data with item embeddings from textual or visual data for a holistic view.

### 2.2 Deep Reinforcement Learning (DRL)
In **DRL**, the system learns to optimize recommendations by interacting with the environment and receiving feedback. This method is suited for scenarios requiring **long-term user engagement**.

**Mathematical Formulation**:
In a Markov Decision Process (MDP):
- **State \( s_t \)**: User profile at time \( t \)
- **Action \( a_t \)**: Recommended item
- **Reward \( r_t \)**: User feedback (e.g., click, purchase)
- **Policy \( \pi \)**: Maps states to actions

**Q-Learning Update Rule**:
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
\]

**Python Example**:
```python
import numpy as np

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Q-table initialization
Q = np.zeros((num_states, num_actions))

# Update rule
def update_q(state, action, reward, next_state):
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
```

## 3. Challenges with Neural Networks in Recommendation Systems
### 3.1 Increased Complexity
Deep neural networks can become highly complex, with numerous hidden layers and parameters. This increases both the training time and computational requirements.

### 3.2 Training Requirements
NNs often require GPUs and specialized hardware, making them expensive to train.

### 3.3 Overfitting
NNs can overfit to the training data, leading to poor generalization on unseen data. Techniques like **dropout**, **early stopping**, and **regularization** are used to mitigate this.

## 4. How LLMs are Revolutionizing Recommendation Systems
### Customizing LLMs for Recommendations
LLMs can be customized through:
- **Pre-training**: Initial large-scale training on general data.
- **Fine-tuning**: Adapting pre-trained models to specific recommendation tasks.
- **Prompting**: Providing task-specific prompts without altering the model's weights.

**Example Model**: **P5** (Recommendation as Language Processing)
- **Pretrain**: Based on T5, trained on a diverse web corpus.
- **Personalized Prompt**: User-specific prompts based on behavior.
- **Predict**: Generates recommendations using the pre-trained LLM.

### Fine-Tuning Strategies
1. **Full-model fine-tuning**: Adjusts all model weights.
2. **Parameter-efficient fine-tuning**: Modifies only a small subset of weights, often with trainable adapters.

### Prompting Techniques
1. **Conventional Prompting**: Simple task descriptions with examples.
2. **In-Context Learning**: Teaches the model tasks using contextual examples.
3. **Chain-of-Thought**: Provides reasoning steps in the prompts to enhance decision-making.

## 5. Implementing an LLM-Powered Recommendation System: MovieHarbor
### Data Preprocessing
1. **Format 'genres' Column**:
```python
import pandas as pd
import ast

md['genres'] = md['genres'].apply(ast.literal_eval)
md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x])
```

2. **Calculate Weighted Ratings**:
```python
def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
    return (vote_count / (vote_count + min_vote_count)) * vote_average + \
           (min_vote_count / (vote_count + min_vote_count)) * 5.0

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
min_vote_count = vote_counts.quantile(0.95)

md['weighted_rate'] = md.apply(
    lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1
)
```

3. **Combine Information**:
```python
md_final['combined_info'] = md_final.apply(
    lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", 
    axis=1
).astype(str)
```



### Tokenize and Embed Data
```python
import tiktoken

embedding_encoding = "cl100k_base"
max_tokens = 8000
encoding = tiktoken.get_encoding(embedding_encoding)

md_final["n_tokens"] = md_final.combined_info.apply(lambda x: len(encoding.encode(x)))
md_final = md_final[md_final.n_tokens <= max_tokens]

md_final["embedding"] = md_final.overview.apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
```

### Store in VectorDB
```python
import lancedb

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table = db.create_table("movies", md_final)
```

### Build LangChain Components for QA
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB

embeddings = OpenAIEmbeddings()
docsearch = Lance