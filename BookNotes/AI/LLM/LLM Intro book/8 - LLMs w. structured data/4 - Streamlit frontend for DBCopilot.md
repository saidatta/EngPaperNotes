## Overview
In this section, we explore building a front-end for our DBCopilot using **Streamlit**, providing users with an interactive, web-based interface to interact with structured databases through natural language queries. This application leverages **LangChain's SQL Agent**, and we’ll incorporate memory management to make the interaction conversational. 

## Prerequisites
- **Streamlit**: A Python library for creating web applications quickly and easily.
- **LangChain**: A framework for developing applications powered by large language models (LLMs).
- **Chinook Database**: A sample SQLite database for SQL practice.

## Key Steps for Building DBCopilot

### 1. Setting Up the Streamlit Web Page
To start, we set up the basic configuration for the Streamlit app:

```python
import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="DBCopilot", page_icon="")
st.header('Welcome to DBCopilot, your copilot for structured databases.')
```

### 2. Importing Credentials and Establishing Database Connection
Ensure that the necessary environment variables are loaded, and establish a connection to the Chinook database using **LangChain's SQLDatabase**:

```python
from langchain.sql_database import SQLDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.environ['OPENAI_API_KEY']

# Establish a connection to the SQLite database
db = SQLDatabase.from_uri('sqlite:///chinook.db')
```

### 3. Initializing the LLM and SQL Toolkit
Set up the LLM and its associated SQL toolkit for database interaction:

```python
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Initialize the LLM
llm = OpenAI()

# Initialize the SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
```

### 4. Creating the Agent with Custom Prompts
Initialize the **SQL Agent** with a custom prompt to ensure the agent explains the SQL query it constructs:

```python
from langchain.agents import create_sql_agent

# Create the SQL agent
agent_executor = create_sql_agent(
    prefix=prompt_prefix,  # Custom prompt defined in previous notes
    format_instructions=prompt_format_instructions,  # Custom format instructions
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    top_k=10  # Limit results to enhance performance and readability
)
```

### 5. Implementing Session States for Memory Management
To maintain a conversational UI, use **Streamlit's session state** to store and display chat history:

```python
# Initialize session state for messages or clear history if requested
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display existing chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
```

### 6. Handling User Queries
Add logic to process user input and display responses from the agent:

```python
user_query = st.text_input("Ask a question about the database:", placeholder="E.g., 'What are the top 5 best-selling albums?'")

if user_query:
    # Append user query to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    # Run the agent with Streamlit's callback handler for real-time output
    with st.chat_message("assistant"):
        from langchain.callbacks.streamlit_callback_handler import StreamlitCallbackHandler
        
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(user_query, callbacks=[st_cb], handle_parsing_errors=True)
        
        # Append agent's response to session state and display it
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
```

### 7. Running the Application
Run the application from the terminal with the following command:
```bash
streamlit run dbcopilot.py
```

**Output**: This sets up a user-friendly web page where users can input questions about the database and receive structured, conversational responses.

### Visual Examples
- **Figure 8.8**: Shows the initial interface of DBCopilot, providing users with a simple input box and displaying responses in a chat format.
- **Figure 8.9**: Demonstrates the agent’s detailed action sequence when queried, showing its decision-making process.

## Explanation of Key Components

### StreamlitCallbackHandler
The `StreamlitCallbackHandler` is used to display real-time responses, allowing users to see each action taken by the agent during query execution. This transparency is vital for debugging and user trust.

### Session State in Streamlit
Session state helps persist data across different runs of the app. Here, it enables maintaining chat history, making the interaction conversational:
- **Initialization**: Checks if the session state exists; if not, it initializes it.
- **Persistence**: Updates the session state with new user queries and agent responses.

## Example of a Full Implementation Code Snippet
```python
# Complete code for dbcopilot.py

import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Streamlit app
st.set_page_config(page_title="DBCopilot", page_icon="")
st.header('Welcome to DBCopilot, your copilot for structured databases.')

# Establish database connection
db = SQLDatabase.from_uri('sqlite:///chinook.db')
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize LLM and toolkit
llm = OpenAI()
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create SQL Agent with custom prompts
agent_executor = create_sql_agent(
    prefix=prompt_prefix,
    format_instructions=prompt_format_instructions,
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    top_k=10
)

# Session state for conversation
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
user_query = st.text_input("Ask a question about the database:", placeholder="E.g., 'What are the top 5 best-selling albums?'")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(user_query, callbacks=[st_cb], handle_parsing_errors=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
```

## Summary
In this chapter, we learned how to:
1. Build a front-end for DBCopilot using **Streamlit**.
2. Establish a connection to a database and use **LangChain** agents for SQL interactions.
3. Implement **session state** to maintain conversation history.
4. Enhance user experience by showing step-by-step agent actions with **StreamlitCallbackHandler**.

This showcases the potential of using **Streamlit** combined with **LangChain** to create intuitive, LLM-powered applications for structured data analysis. In the following chapters, we will explore how LLMs can handle code-related tasks, expanding their analytical capabilities.