## Overview
In a content-based recommendation system, we use the historical data and attributes of users to recommend items similar to what they have shown interest in. This approach contrasts with cold-start scenarios, as we assume the system has access to user data and preferences. In this note, we will explore building a content-based recommendation system using LangChain and other Python tools.

## High-Level Architecture
The content-based recommendation system architecture consists of the following main components:
1. **User Data Retrieval**: Collects user information, including age, gender, and movies watched with ratings.
2. **Prompt Engineering**: Customizes prompts to embed user information and guide LLMs for generating recommendations.
3. **Retrieval Mechanism**: Leverages vector databases (VectorDB) to find the most relevant content.
4. **QA Chain**: Uses LangChain’s `RetrievalQA` to construct responses based on context.

### Figure: High-level architecture diagram for a content-based recommendation system

## 1. User Data Preparation
Create a sample dataset that includes users' attributes and their movie preferences.

**Python Code for Sample Dataset**:
```python
import pandas as pd

# Sample user data with attributes and watched movies with ratings
data = {
    "username": ["Alice", "Bob"],
    "age": [25, 32],
    "gender": ["F", "M"],
    "movies": [
        [("Transformers: The Last Knight", 7), ("Pokémon: Spell of the Unknown", 5)],
        [("Bon Cop Bad Cop 2", 8), ("Goon: Last of the Enforcers", 9)]
    ]
}

# Convert the movies column into a dictionary format
for i, row_movies in enumerate(data["movies"]):
    movie_dict = {movie: rating for movie, rating in row_movies}
    data["movies"][i] = movie_dict

# Create a DataFrame
df = pd.DataFrame(data)
print(df.head())
```

**Sample Output**:
| username | age | gender | movies |
|----------|-----|--------|-------------------------------------------|
| Alice    | 25  | F      | {'Transformers: The Last Knight': 7, ...} |
| Bob      | 32  | M      | {'Bon Cop Bad Cop 2': 8, ...}             |

## 2. Prompt Engineering
Create a custom prompt that embeds user data, enabling the LLM to tailor recommendations based on known user preferences.

**Prompt Structure**:
- **Template Prefix**: General instruction for the model.
- **User Information**: Dynamic user-specific context.
- **Template Suffix**: Query structure for the user's question.

**Python Code for Prompt Preparation**:
```python
template_prefix = """You are a movie recommender system that helps users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}"""

user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}
Movies already seen alongside with rating: {movies}"""

template_suffix = """Question: {question}
Your response:"""

# Format user data into the prompt
age = df.loc[df['username'] == 'Alice']['age'].values[0]
gender = df.loc[df['username'] == 'Alice']['gender'].values[0]
movies = ''.join([f"Movie: {movie}, Rating: {rating}\n" for movie, rating in df['movies'][0].items()])

formatted_user_info = user_info.format(age=age, gender=gender, movies=movies)
COMBINED_PROMPT = f"{template_prefix}\n{formatted_user_info}\n{template_suffix}"
print(COMBINED_PROMPT)
```

**Sample Prompt Output**:
```
You are a movie recommender system that helps users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
This is what we know about the user, and you can use this information to better tune your research:
Age: 25
Gender: F
Movies already seen alongside with rating: 
Movie: Transformers: The Last Knight, Rating: 7
Movie: Pokémon: Spell of the Unknown, Rating: 5
Question: {question}
Your response:
```

## 3. Setting Up the RetrievalQA Chain
Use LangChain’s `RetrievalQA` to set up the question-answering system that leverages a VectorDB for content retrieval.

**Python Code for QA Chain**:
```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create a prompt template for the chain
PROMPT = PromptTemplate(template=COMBINED_PROMPT, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Set up the QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# User query
query = "Can you suggest me some action movies based on my background?"
result = qa({'query': query})
print(result['result'])
```

**Sample Result**:
```
"Based on your age, gender, and the movies you've already seen, I would suggest the following action movies: The Raid 2 (Action, Crime, Thriller; Rating: 6.71), Ong Bak 2 (Adventure, Action, Thriller; Rating: 5.24), Hitman: Agent 47 (Action, Crime, Thriller; Rating: 5.37), and Kingsman: The Secret Service (Crime, Comedy, Action, Adventure; Rating: 7.43)."
```

## 4. Front-End Development with Streamlit
Integrate the backend logic into a user-friendly front-end using Streamlit.

**Python Code for Streamlit App**:
```python
import streamlit as st
from dotenv import load_dotenv
import lancedb
import pandas as pd

# Configure the app
st.set_page_config(page_title="MovieHarbor", page_icon="")
st.header("Welcome to MovieHarbor, your favorite movie recommender")

# User input widgets
st.sidebar.title("User Information")
age = st.sidebar.slider("What is your age?", 1, 100, 25)
gender = st.sidebar.radio("What is your gender?", ["Male", "Female", "Other"])
genre = st.sidebar.selectbox("What is your favorite movie genre?", md.explode('genres')["genres"].unique())

# Create the chain and set up user interaction
query = st.text_input("Enter your question:", placeholder="What action movies do you suggest?")
if query:
    result = qa({'query': query})
    st.write(result['result'])
```

**Running the App**:
Execute the following command to run the Streamlit app:
```bash
streamlit run movieharbor.py
```

### Figure: Sample Front-End for MovieHarbor with Streamlit

## 5. Using Feature Stores for Enhanced Context
For real-world applications, feature stores like **Feast**, **Tecton**, **Featureform**, and **AzureML Managed Feature Store** can be used to handle large-scale user data and contextual information.

**Key Integrations**:
- **Feast**: Open-source feature store supporting batch and streaming data.
- **Tecton**: Managed platform for defining, versioning, and serving features.
- **Featureform**: Virtual feature store that integrates with existing data infrastructure.
- **AzureML**: Allows creation and operationalization of features with native support for machine learning pipelines.

**Reference Links**:
- [LangChain's Feature Store Blog](https://blog.langchain.dev/feature-stores-and-llms/)
- [Feast Documentation](https://docs.feast.dev/)

## Summary
In this guide, we explored building a content-based recommendation system that leverages user-specific information for better recommendations. By embedding user data into prompts and using LangChain’s `RetrievalQA`, we were able to create a robust, adaptable system that tailors suggestions based on historical data. Finally, we developed a simple Streamlit app for a user-friendly interface and considered using feature stores for production-grade solutions.