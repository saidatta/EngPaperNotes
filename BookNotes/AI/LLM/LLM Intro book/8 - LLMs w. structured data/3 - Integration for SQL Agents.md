# Advanced Prompt Engineering and Multi-Tool Integration for SQL Agents
## Overview of Prompt Engineering for SQL Agents
Prompt engineering is crucial for enhancing the behavior and response quality of agents, including LangChain's SQL agents. By customizing prompts, we can control how agents construct their output, ensure they explain their reasoning, and adapt their responses for specific use cases.

### Default Prompt Structure
LangChain SQL agents use a default prompt template with specific components:
- **Prompt Prefix**: Sets the initial context and rules for the agent.
- **Format Instructions**: Guides the agent on how to format its output.

These components form the agent's base instructions for querying databases, formulating SQL, and presenting the results.

### Example of Default Prompt
```plaintext
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
...
If the question does not seem related to the database, just return "I don't know" as the answer.
```

---

## Customizing Prompts for Enhanced Explanations
### Goal: Include SQL Query Explanations
To make our SQL agent more transparent, we can modify the prompt to include detailed explanations with the final answer. We do this by customizing the `prompt_prefix` and `format_instructions`.

### Custom `prompt_prefix`
```python
prompt_prefix = '''
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
...
As part of your final answer, ALWAYS include an explanation of how you got to the final answer, including the SQL query you ran. Include the explanation and the SQL query in the section that starts with "Explanation:".
'''
```

### Custom `format_instructions` for Explanations
Use few-shot learning to show the agent how to structure explanations:
```python
prompt_format_instructions = '''
Explanation:
<===Beginning of an Example of Explanation:
I joined the invoices and customers tables on the customer_id column, which is the common key between them. This allowed me to access the Total and Country columns from both tables. Then I grouped the records by the country column and calculated the sum of the Total column for each country, ordered them in descending order, and limited the SELECT to the top 5.
```sql
SELECT c.country AS Country, SUM(i.total) AS Sales
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY Country
ORDER BY Sales DESC
LIMIT 5;
```sql
===>End of an Example of Explanation
'''
```

### Integrating Custom Prompts into the Agent
```python
agent_executor = create_sql_agent(
    prefix=prompt_prefix,
    format_instructions=prompt_format_instructions,
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    top_k=10
)

# Run the agent with a query
result = agent_executor.run("What are the top 5 best-selling albums and their artists?")
print(result)
```

**Expected Output**:
```plaintext
The top 5 best-selling albums and their artists are 'A Matter of Life and Death' by Iron Maiden, 'BBC Sessions [Disc 1] [live]' by Led Zeppelin, ...
Explanation:
I joined the album and invoice tables on the album_id column and joined the album and artist tables on the artist_id column. This allowed me to access the title and artist columns...
```sql
SELECT al.title AS Album, ar.name AS Artist, SUM(i.total) AS Sales
FROM album al
JOIN invoice i ON al.invoice_id = i.invoice_id
JOIN artist ar ON al.artist_id = ar.artist_id
GROUP BY ar.name
ORDER BY Sales
```

---

## Extending DBCopilot Capabilities with Additional Tools
### Adding Python and File Management Tools
To enhance the versatility of DBCopilot, we can integrate tools like `PythonREPLTool` for executing Python code and `FileManagementToolkit` for file operations.

### Tool Definitions:
- **`PythonREPLTool`**: Enables Python code execution directly from the agent's environment.
- **`FileManagementToolkit`**: Provides file system interaction capabilities, such as reading, writing, and listing files.

### Implementation Code:
```python
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.python import PythonREPL
from langchain.agents.agent_toolkits import FileManagementToolkit
import os

# Define the working directory
working_directory = os.getcwd()

# Initialize file management tools
file_tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["read_file", "write_file", "list_directory"]
).get_tools()

# Add Python REPL tool
file_tools.append(PythonREPLTool())

# Extend with SQL Database tools
file_tools.extend(SQLDatabaseToolkit(db=db, llm=llm).get_tools())
```

### Creating a Multi-Tool Agent:
```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

model = ChatOpenAI()
agent = initialize_agent(
    tools=file_tools,
    model=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent to generate and save a plot
agent.run("generate a matplotlib bar chart of the top 5 countries for sales from the chinook database. Save the output as 'figure.png'.")
```

### Example Execution Chain:
```plaintext
> Entering new AgentExecutor chain...
Action: sql_db_query
Action Input: "SELECT billing_country AS Country, SUM(total) AS Sales FROM invoices GROUP BY billing_country ORDER BY Sales DESC LIMIT 5"
Observation: [('USA', 10405.89), ('Canada', 5489.55), ('Brazil', 4059.0), ('France', 3972.87), ('Germany', 3441.24)]
...
Action: Python_REPL
Action Input:
"""
import matplotlib.pyplot as plt
sales_data = [('USA', 10405.89), ('Canada', 5489.55), ('Brazil', 4059.0), ('France', 3972.87), ('Germany', 3441.24)]
x = [item[0] for item in sales_data]
y = [item[1] for item in sales_data]
plt.bar(x, y)
plt.xlabel('Country')
plt.ylabel('Sales')
plt.title('Top 5 Countries for Sales')
plt.savefig('figure.png')
plt.show()
"""
Observation: Bar chart displayed and saved as 'figure.png'.
> Finished chain.
```

**Result**: The agent dynamically queries the database, generates the chart using Python, and saves the result to the current directory.

---

## Customizing Multi-Tool Agent Prompts
We can customize the multi-tool agent to enhance its explanatory capabilities for Python code generation:

### Custom Prompt for Detailed Explanations:
```python
prompt_prefix = """
You are an advanced agent capable of interacting with SQL databases, running Python code, and managing files. Always provide a detailed explanation of both the SQL and Python code you use.
"""
prompt_format_instructions = """
Explanation:
<=== Example Explanation:
To create the chart, I first queried the database for the top 5 countries with the highest sales. I then used Python's matplotlib to generate a bar chart from the results. The code saved the plot as a PNG file in the working directory.
```python
# SQL code used
SELECT billing_country AS Country, SUM(total) AS Sales
FROM invoices
GROUP BY billing_country
ORDER BY Sales DESC
LIMIT 5;

# Python code for the chart
import matplotlib.pyplot as plt
sales_data = [('USA', 10405.89), ('Canada', 5489.55), ('Brazil', 4059.0), ('France', 3972.87), ('Germany', 3441.24)]
x = [item[0] for item in sales_data]
y = [item[1] for item in sales_data]
plt.bar(x, y)
plt.xlabel('Country')
plt.ylabel('Sales')
plt.title('Top 5 Countries for Sales')
plt.savefig('figure.png')
plt.show()
```python
<=== End of Example Explanation
"""
```

### Initializing Agent with Custom Prompts:
```python
agent = initialize_agent(
    tools=file_tools,
    model=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        'prefix': prompt_prefix,
        'format_instructions': prompt_format_instructions
    }
)
```

### Running Enhanced Agent:
```python
result = agent.run("Retrieve the top 5 genres by number of tracks and plot them as a bar chart. Save it as 'genres_chart.png'.")
print(result)
```

---

## Conclusion
By using prompt engineering and integrating multiple tools, we can enhance the capabilities of LangChain agents to interact with structured data, generate visual outputs, and manage files seamlessly. This flexibility allows the creation of sophisticated assistants like **DBCopilot** that bridge the gap between complex SQL queries and user-friendly interactions.

### References
- LangChain Documentation: [LangChain](https://python.langchain.com/)
- Python `matplotlib`: [Matplotlib](https://matplotlib.org/)
- SQL Tutorial: [W3Schools SQL](https://www.w3schools.com/sql/)