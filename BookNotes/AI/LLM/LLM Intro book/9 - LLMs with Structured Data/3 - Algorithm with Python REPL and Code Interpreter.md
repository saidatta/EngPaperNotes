## Overview
Complex problems often require more than just analytical reasoning; they may need algorithmic problem-solving capabilities. LLMs, while proficient in reasoning, can leverage their code generation capabilities to function as an algorithm, solving problems dynamically by executing Python code. This can be accomplished using **LangChain's Python REPL** tool, which allows LLM-powered agents to execute Python commands to perform complex calculations, generate visualizations, and solve algorithmic tasks.

## Key Tools
- **Python REPL**: An interactive Python shell that allows LLMs to execute Python commands and return results in real-time.
- **LangChain Framework**: Provides the tools to build LLM-powered agents capable of running Python code.
- **Code Interpreter API**: An open-source project that simulates the Code Interpreter plugin by OpenAI, capable of executing code, performing complex calculations, and generating visual outputs.

## Initializing Python REPL Agent

### Prerequisites
- **LangChain** installed via `pip install langchain`
- **Python 3.7.1** or later
- OpenAI API key for LLM access

### Setting Up the Python REPL Agent
```python
import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools import PythonREPLTool

# Load environment variables
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the LLM and the Python REPL tool
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
agent_executor = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

### Default Prompt Inspection
```python
print(agent_executor.agent.llm_chain.prompt.template)
```
This reveals the template guiding the agent's behavior when reasoning through Python-based tasks.

## Use Case Examples

### 1. Visualizing Player Statistics
**Objective**: Create a scatter plot of basketball player stats.
**Prompt**:
```python
query = """
In a basketball game, we have the following player stats:
- Player A: 38 points, 10 rebounds, 7 assists
- Player B: 28 points, 9 rebounds, 6 assists
- Player C: 19 points, 6 rebounds, 3 assists
- Player D: 12 points, 4 rebounds, 2 assists
- Player E: 7 points, 2 rebounds, 1 assist
Create a scatter plot where the y-axis represents points, the x-axis represents rebounds, and each point is labeled with the player's name.
"""
agent_executor.run(query)
```

**Result**:
The output includes Python code for creating the scatter plot using `matplotlib` and `seaborn`, along with a visualization labeled "Team Players."

### 2. Training a Regression Model
**Objective**: Predict house prices based on synthetic data.

**Prompt**:
```python
query = """
I want to predict the price of a house given the following features:
- Number of rooms
- Number of bathrooms
- Size in square meters
Design and train a regression model using synthetic data and predict the price for a house with 2 rooms, 1 bathroom, and 100 square meters.
"""
agent_executor.run(query)
```

**Output**:
The agent generates synthetic data, trains a linear regression model using `scikit-learn`, and returns the predicted price:
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 houses with 3 features
y = 100000 * X[:, 0] + 200000 * X[:, 1] + 300000 * X[:, 2] + 50000  # Price formula

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict for the given features
features = np.array([[2, 1, 100]])
predicted_price = model.predict(features)
print(f"The predicted price is approximately ${predicted_price[0]:,.2f}")
```

### 3. Optimization Problem: HVAC Energy Cost Minimization
**Objective**: Optimize HVAC setpoints to minimize energy cost while maintaining comfort.

**Prompt**:
```python
query = """
Optimize the HVAC setpoints for three zones to minimize energy costs while ensuring comfort. The initial temperatures and comfort ranges are as follows:
- Zone 1: Initial 72°F, Range 70°F-74°F, Cost $0.05/degree/hour
- Zone 2: Initial 75°F, Range 73°F-77°F, Cost $0.07/degree/hour
- Zone 3: Initial 70°F, Range 68°F-72°F, Cost $0.06/degree/hour
Calculate the minimum energy cost.
"""
agent_executor.run(query)
```

**Solution Logic**:
The agent uses `scipy.optimize` to find the optimal temperature setpoints that minimize the energy cost.

```python
import scipy.optimize as opt

# Define the cost function
def cost_function(x):
    zone1_cost = 0.05 * abs(x[0] - 72)
    zone2_cost = 0.07 * abs(x[1] - 75)
    zone3_cost = 0.06 * abs(x[2] - 70)
    return zone1_cost + zone2_cost + zone3_cost

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 70},  # Zone 1 lower bound
    {'type': 'ineq', 'fun': lambda x: 74 - x[0]},  # Zone 1 upper bound
    {'type': 'ineq', 'fun': lambda x: x[1] - 73},  # Zone 2 lower bound
    {'type': 'ineq', 'fun': lambda x: 77 - x[1]},  # Zone 2 upper bound
    {'type': 'ineq', 'fun': lambda x: x[2] - 68},  # Zone 3 lower bound
    {'type': 'ineq', 'fun': lambda x: 72 - x[2]}   # Zone 3 upper bound
]

# Optimization
initial_guess = [72, 75, 70]
result = opt.minimize(cost_function, initial_guess, constraints=constraints)
min_cost = result.fun
print(f"Minimum total energy cost: ${min_cost:.2f} per hour")
```

**Result**:
The agent returns the optimal setpoints and the minimized energy cost.

## Leveraging the Code Interpreter API
The **Code Interpreter API** allows for real-time code execution and is particularly useful for visualizations, file analyses, and real-time data retrieval.

### Setup
```python
!pip install "codeinterpreterapi[all]"

from codeinterpreterapi import CodeInterpreterSession, File
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

# Create a session
async with CodeInterpreterSession() as session:
    response = await session.generate_response(
        "Generate a plot of COVID-19 cases from March to June 2020."
    )
    print("AI: ", response.content)
    for file in response.files:
        file.show_image()
```

### Use Case: Plotting Data from a File
```python
async with CodeInterpreterSession() as session:
    user_request = "Analyze this dataset and plot something interesting about it."
    files = [File.from_path("titanic.csv")]

    response = await session.generate_response(user_request, files=files)
    print("AI: ", response.content)
    for file in response.files:
        file.show_image()
```

**Result**:
- The Code Interpreter can analyze files and generate visual plots, e.g., survival rates by class or gender.

## Applications Beyond Code Execution
1. **Supply Chain Optimization**
2. **Portfolio Management in Finance**
3. **Route Planning for Logistics**
4. **Healthcare Resource Allocation**
5. **Agricultural Planning**

## Limitations and Caveats
- **Python REPL Limitations**:
  - No File I/O support.
  - Variables are reset after each run.
- **Code Interpreter Advantages**:
  - Can execute and remember code across runs.
  - Provides real-time data retrieval and visualization.

## Summary
By leveraging LangChain’s Python REPL and Code Interpreter API, LLMs can act as real-time algorithms, enabling solutions for complex problems like optimization and data visualization. This integration bridges the gap between human intent and executable code, pushing the boundaries of automation and data-driven problem-solving in software development.

## References
- [Code Interpreter API on GitHub](https://github.com/shroominic/codeinterpreter-api)
- [LangChain Python REPL Toolkit](https://python.langchain.com/docs/integrations/toolkits/python)
- [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file)