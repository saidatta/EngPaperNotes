## Overview
This chapter delves into the powerful use case of integrating LLMs as natural language interfaces for structured data systems, such as relational databases. This integration allows LLMs to bridge the gap between business users and complex data queries by offering conversational interactions that return contextually rich responses. We will build a "DBCopilot" system capable of querying structured data conversationally, leveraging the LangChain framework, SQLite, and Python.

### Key Topics:
- Overview of structured data and its importance.
- Using tools and plugins to connect LLMs to structured, tabular data.
- Building a natural language interface (DBCopilot) for querying databases.

## 1. Understanding Structured Data

### Types of Data:
1. **Unstructured Data**: Lacks a predefined format, making it flexible but harder to process (e.g., documents, audio).
2. **Structured Data**: Organized into a clear schema (rows and columns), making it easy to query using SQL.
3. **Semi-Structured Data**: Has some structure but is flexible (e.g., JSON, XML).

### Example of Structured Data:
Relational databases store data in tables with defined columns and rows. A table in a library database may look like this:

| **BookID** | **Title**           | **AuthorID** |
|------------|---------------------|--------------|
| 1          | "Pride and Prejudice" | 101          |
| 2          | "1984"              | 102          |

### Key Concepts:
- **Primary Key**: Unique identifier for each row in a table.
- **Foreign Key**: A field that links to a primary key in another table to establish relationships between tables.

**Example**:
The **Authors** table and **Books** table are linked by the `AuthorID`, which is a foreign key in the Books table referencing the primary key in the Authors table.

### Example SQL Query:
```sql
SELECT Books.Title, Authors.Name
FROM Books
JOIN Authors ON Books.AuthorID = Authors.AuthorID;
```
This query retrieves book titles along with their respective authors' names.

## 2. Introduction to Relational Databases
Relational databases manage structured data and use SQL for data retrieval. Some popular relational database systems include:

- **SQL Databases**: MySQL, PostgreSQL, SQLite.
- **Oracle Database**: Robust and scalable.
- **SQLite**: Lightweight, serverless, and easy to integrate with Python.

**SQLite Overview**:
SQLite is a self-contained, serverless database that doesn’t require setup, making it ideal for embedded systems and mobile apps.

## 3. Using LLMs for Structured Data Queries
LLMs can be connected to structured data sources to interpret natural language questions and return structured, context-rich responses. This involves:
- Leveraging **LLM plugins**.
- Utilizing an **agentic approach** to create a copilot for database interactions.

## 4. Connecting Python to SQLite

### Python Setup:
Ensure the following Python packages are installed:
```bash
pip install langchain python-dotenv streamlit sqlite3
```

### Sample Database: Chinook
The **Chinook** database is a popular sample database used for learning. It contains tables such as **Albums**, **Artists**, **Tracks**, etc.

### Connecting to SQLite in Python:
```python
import sqlite3
import pandas as pd

# Connect to the Chinook database
conn = sqlite3.connect('Chinook_Sqlite.sqlite')

# Query to explore tables
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(query, conn)
print("Tables in the database:", tables)
```

**Output**:
```
Tables in the database:
| name        |
|-------------|
| Albums      |
| Artists     |
| Customers   |
| ...         |
```

## 5. Creating DBCopilot Using LangChain

### Step 1: Setting Up the LangChain Environment
```python
from langchain.chains import SQLDatabaseChain
from langchain.llms import OpenAI
from langchain_sql_database import SQLDatabase

# Connect to the SQLite database using LangChain
db = SQLDatabase.from_uri('sqlite:///Chinook_Sqlite.sqlite')

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-3.5-turbo")

# Create a SQL Database Chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
```

### Step 2: Querying with Natural Language
```python
query = "Which artists have released albums with more than 10 tracks?"
result = db_chain.run(query)
print(result)
```

**Expected Output**:
```
Artists like 'Led Zeppelin' and 'The Beatles' have released albums with more than 10 tracks.
```

## 6. Enhancing DBCopilot with Contextual Prompts
To provide rich, contextual responses, custom prompts can be designed:
```python
from langchain.prompts import PromptTemplate

template = """You are a database assistant that helps users by querying the database and providing detailed answers.
Use the provided database context to answer the user's question in a clear and comprehensive way.
{context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
db_chain.set_prompt(PROMPT)
```

### Running an Example Query:
```python
result = db_chain.run("List the top 5 albums by track count.")
print(result)
```

## 7. Building the Front-End with Streamlit
Create a user-friendly interface using **Streamlit**:

**Code**:
```python
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="DBCopilot", page_icon="🗄️")
st.title("DBCopilot: Your Database Assistant")

# User input
query = st.text_input("Ask your database question here:")

if query:
    result = db_chain.run(query)
    st.write(result)
```

**Run the App**:
```bash
streamlit run dbc_assistant.py
```

## 8. Example Use Cases
### Business Intelligence:
- **Question**: "What was the total revenue generated by the top 3 customers last quarter?"
- **Answer**: The DBCopilot retrieves the relevant data and summarizes the revenue details.

### Data Insights:
- **Question**: "Which albums were released between 2005 and 2010?"
- **SQL Generated**:
```sql
SELECT Title FROM Albums WHERE ReleaseYear BETWEEN 2005 AND 2010;
```

## 9. Expanding Capabilities with Plugins
To enhance DBCopilot, consider integrating with plugins and additional tools:
- **Plugins for Excel**: Enables direct interaction with Excel sheets.
- **GraphQL Plugins**: Facilitates querying GraphQL APIs.
- **Visualization Plugins**: Integrates with plotting libraries like `matplotlib` for visual data representation.

## 10. Summary
This chapter showcased how to extend the capabilities of LLMs to handle structured data effectively. By building a DBCopilot with LangChain, OpenAI, and Streamlit, we demonstrated how to create an intuitive, natural language interface for querying relational databases. This approach can bridge the technical gap for business users, offering insightful, conversational responses from structured datasets.

## References
- **LangChain Documentation**: [LangChain](https://langchain.com/)
- **SQLite Official Site**: [SQLite](https://www.sqlite.org/index.html)
- **Streamlit Documentation**: [Streamlit](https://docs.streamlit.io/)
- **Chinook Database**: [Chinook](https://github.com/lerocha/chinook-database)

By mastering these techniques, Staff+ engineers can facilitate the creation of more accessible data tools that empower non-technical users to derive value from structured datasets using natural language interfaces.