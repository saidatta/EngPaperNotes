## Overview of the Chinook Database
The **Chinook database** is a popular sample database that models a digital media store. It is often used for learning SQL and database management. The database structure is inspired by a typical music store and includes tables related to artists, albums, tracks, customers, invoices, and more.

### Key Features:
- **Realistic Data**: Based on an iTunes library for practical learning.
- **Clear Schema**: Simplifies understanding and querying.
- **Comprehensive SQL Use**: Supports complex queries involving joins, views, subqueries, and triggers.
- **Multi-Platform**: Available for SQL Server, Oracle, MySQL, PostgreSQL, SQLite, and DB2.

### Table Relationships:
The Chinook database includes **11 main tables** connected by primary and foreign keys, enabling complex data relationships. The main tables are:

- **Albums**: Stores album information.
- **Artists**: Contains artist names and IDs.
- **Tracks**: Includes track details and links to albums and media types.
- **Customers**: Contains customer data.
- **Invoices**: Stores invoice details with customer references.
- **Invoice_Line**: Tracks individual items on an invoice.
- **Genres**: Details different music genres.
- **Media_Type**: Stores media type information.
- **Playlists**: Contains playlists.
- **Playlist_Track**: Maps playlists to tracks.
- **Employees**: Contains employee information for the media store.

**Diagram**:
To visualize table relationships, refer to a database diagram such as the one available at [Chinook Database GitHub](https://github.com/arjunchndr/Analyzing-Chinook-Database-using-SQL-and-Python).

---

## Connecting Python to the Chinook Database

### Python Libraries for Relational Databases:
- **sqlite3**: For SQLite connections.
- **SQLAlchemy**: A powerful SQL toolkit and Object-Relational Mapper (ORM).
- **Psycopg**: For PostgreSQL.
- **MySQLdb**: For MySQL interactions.
- **cx_Oracle**: For Oracle Database connections.

### Example: Connecting to SQLite with Python
**Step-by-step guide**:
1. **Download** the Chinook database from [SQLite Tutorial](https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip) and extract `chinook.db`.
2. **Connect to the database** and inspect its tables.

```python
import sqlite3
import pandas as pd

# Create a connection
database = 'chinook.db'
conn = sqlite3.connect(database)

# List all tables in the database
tables = pd.read_sql("""
SELECT name, type
FROM sqlite_master
WHERE type IN ("table", "view");
""", conn)

print("Tables in the database:")
print(tables)
```

**Output**:
A list of tables such as:
```
| name           | type   |
|----------------|--------|
| albums         | table  |
| artists        | table  |
| customers      | table  |
| ...            | ...    |
```

### Inspecting Table Schema
To understand table structures, use the `PRAGMA` command:

```python
customer_columns = pd.read_sql("PRAGMA table_info(customers);", conn)
print(customer_columns)
```

**Output**:
Details the columns, types, and constraints of the `customers` table.

### Query Example: Top Countries by Sales
```python
query = """
SELECT c.country AS Country, SUM(i.total) AS Sales
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY Country
ORDER BY Sales DESC
LIMIT 5;
"""
top_countries = pd.read_sql(query, conn)
print(top_countries)
```

**Output**:
Displays the top 5 countries by sales.

### Visualizing Data with Matplotlib
```python
import matplotlib.pyplot as plt

# SQL query for track counts by genre
sql = """
SELECT g.Name AS Genre, COUNT(t.track_id) AS Tracks
FROM genre g
JOIN track t ON g.genre_id = t.genre_id
GROUP BY Genre
ORDER BY Tracks DESC;
"""
genre_data = pd.read_sql(sql, conn)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(genre_data.Genre, genre_data.Tracks)
plt.title("Number of Tracks by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Tracks")
plt.xticks(rotation=90)
plt.show()
```

**Output**:
A bar chart showing the distribution of tracks by genre.

---

## Implementing DBCopilot with LangChain

### LangChain Agents for SQL
**LangChain** provides agents that act as intelligent decision-makers capable of interacting with databases. This chapter introduces **SQL Agents** using `LangChain` to translate natural language questions into SQL queries.

### Components:
- **`create_sql_agent`**: Creates an agent for SQL interactions.
- **`SQLDatabaseToolkit`**: Provides non-parametric knowledge to the agent.
- **`OpenAI`**: Used as the LLM backend for reasoning and generating answers.

### Setting Up the SQL Agent
```python
from langchain.agents import create_sql_agent
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Initialize the connection
db = SQLDatabase.from_uri('sqlite:///chinook.db')

# Setup LLM and toolkit
llm = OpenAI(model_name="gpt-3.5-turbo")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
```

### Available Agent Tools:
```python
print([tool.name for tool in toolkit.get_tools()])
```

**Output**:
```
['sql_db_query', 'sql_db_schema', 'sql_db_list_tables', 'sql_db_query_checker']
```

### Running Queries with the Agent
**Example**: Describe the `playlist_track` table.
```python
agent_executor.run("Describe the playlist_track table")
```

**Output**:
```
The playlist_track table contains the playlist_id and track_id columns. It has a primary key of playlist_id and track_id...
```

### Example: Aggregated Query
**Question**: "What is the total number of tracks and average length by genre?"
```python
result = agent_executor.run("What is the total number of tracks and the average length of tracks by genre?")
print(result)
```

**Output**:
A detailed response listing top genres with track counts and average lengths.

---

## Customizing the Agent's Prompt
The default prompt template for the SQL Agent ensures secure and structured interactions:
```python
print(agent_executor.agent.llm_chain.prompt.template)
```

**Output**:
```
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQL query to run...
```

### Enhancing the Agent:
- **Ensure SQL correctness**: Use `sql_db_query_checker`.
- **Limit queries**: Prevent full table scans by restricting query results.
- **Avoid DML statements**: Safeguard against unwanted data modifications.

---

## Conclusion
The Chinook database offers an excellent playground for practicing SQL and database-driven LLM integrations. By leveraging LangChain and SQLite, you can create advanced tools like DBCopilot to enable natural language interactions with structured data, fostering a deeper connection between technical and business insights.

## References
- **Chinook Database**: [SQLite Tutorial](https://www.sqlitetutorial.net/sqlite-sample-database/)
- **LangChain Documentation**: [LangChain](https://langchain.com/)
- **SQLAlchemy**: [SQLAlchemy](https://www.sqlalchemy.org/)

These notes are structured to guide Staff+ engineers through understanding, connecting, and interacting with the Chinook database using both SQL and advanced LLM techniques.