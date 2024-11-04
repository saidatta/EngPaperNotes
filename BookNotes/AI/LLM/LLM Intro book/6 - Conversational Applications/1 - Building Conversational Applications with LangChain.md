This section delves into the implementation of LLM-powered conversational applications, laying the groundwork for building sophisticated chatbots that leverage various LangChain components. We'll create a travel assistant chatbot called **GlobeBotter** and incrementally enhance its functionality with memory, non-parametric knowledge, and external tools. By the end, we'll develop a front-end using **Streamlit** for user interaction.

## Table of Contents
1. [Introduction to Conversational Applications](#introduction-to-conversational-applications)
2. [Creating a Basic Bot](#creating-a-basic-bot)
   - [Code Example: Initializing the Basic Bot](#code-example-initializing-the-basic-bot)
3. [Adding Memory to the Bot](#adding-memory-to-the-bot)
   - [ConversationBufferMemory Example](#conversationbuffermemory-example)
4. [Enhancing the Bot with Non-Parametric Knowledge](#enhancing-the-bot-with-non-parametric-knowledge)
   - [Building the Conversational Retrieval Chain](#building-the-conversational-retrieval-chain)
5. [Creating an Agentic Chatbot](#creating-an-agentic-chatbot)
   - [Implementing a Conversational Agent with Tools](#implementing-a-conversational-agent-with-tools)
6. [Adding External Tools](#adding-external-tools)
   - [Google SerpApi Integration Example](#google-serpapi-integration-example)
7. [Developing the Front-End with Streamlit](#developing-the-front-end-with-streamlit)
8. [Summary](#summary)
9. [References](#references)

---

## Introduction to Conversational Applications
Conversational applications enable natural language interactions with users, serving various purposes such as providing information, assistance, and entertainment. Leveraging **Large Language Models (LLMs)**, these applications can enhance user experiences by incorporating reasoning capabilities and integrating parametric and non-parametric knowledge.

---

## Creating a Basic Bot

To build **GlobeBotter**, we start by creating a simple chatbot using LangChain that responds based on predefined prompts. The bot will respond to user queries related to travel plans.

### Code Example: Initializing the Basic Bot

```python
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Initialize the chat model
chat = ChatOpenAI()

# Define the messages schema
messages = [
    SystemMessage(content="You are a helpful assistant that helps the user plan an optimized itinerary."),
    HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")
]

# Generate output
output = chat(messages)
print(output.content)
```

**Expected Output**:
```
Day 1:
1. Visit the Colosseum...
Day 2:
1. Explore Vatican City...
```

---

## Adding Memory to the Bot

Memory enhances the bot's ability to keep track of previous user interactions, enabling coherent follow-up responses.

### ConversationBufferMemory Example

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize memory
memory = ConversationBufferMemory()

# Create the conversation chain with memory
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

# Start interacting
conversation.run("Hi there!")
conversation.run("What is the most iconic place in Rome?")
conversation.run("What kind of other events?")
```

**Output**:
```
> Entering new ConversationChain chain...
'Hello! How can I assist you today?'
'The most iconic place in Rome is probably the Colosseum...'
'Other events that took place at the Colosseum include...'
```

### Interactive Loop

```python
while True:
    query = input('you: ')
    if query == 'q':
        break
    output = conversation({"input": query})
    print('AI system: ', output['response'])
```

---

## Enhancing the Bot with Non-Parametric Knowledge

Adding access to external data, such as travel guides, allows the bot to retrieve information beyond its parametric knowledge.

### Building the Conversational Retrieval Chain

#### Code Example: Setting Up Document Retrieval

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Load and split the PDF document
loader = PyPDFLoader('italy_travel.pdf')
raw_documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
documents = splitter.split_documents(raw_documents)

# Create a vector store
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Initialize memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create a retrieval chain
llm = OpenAI()
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=memory, verbose=True)

# Run a query
response = qa_chain.run({'question': 'Give me some review about the Pantheon'})
print(response)
```

**Expected Output**:
```
"Miskita: 'Angelic and non-human design,' was how Michelangelo described the Pantheon..."
```

---

## Creating an Agentic Chatbot

Agents enable the chatbot to dynamically decide the appropriate action or tool to use based on user input.

### Implementing a Conversational Agent with Tools

#### Code Example: Creating a Conversational Agent

```python
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

# Create a tool for document retrieval
tool = create_retriever_tool(
    db.as_retriever(),
    "italy_travel",
    "Searches and returns documents regarding Italy."
)

# Initialize the agent with tools
tools = [tool]
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
llm = ChatOpenAI(temperature=0)
agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

# Run a query
agent_executor({"input": "Tell me something about the Pantheon"})
```

**Output**:
```
> Invoking: `italy_travel` with `Pantheon`
...document content...
```

---

## Adding External Tools

### Google SerpApi Integration Example

To enable real-time search capabilities, integrate the **SerpApi** tool.

#### Code Example: Integrating SerpApi

```python
from langchain import SerpAPIWrapper
from langchain.agents import Tool
import os
from dotenv import load_dotenv

# Load SerpApi key
load_dotenv()
os.environ["SERPAPI_API_KEY"]

# Initialize SerpApi tool
search = SerpAPIWrapper()
tools.append(
    Tool.from_function(
        func=search.run,
        name="Search",
        description="Useful for answering questions about current events."
    )
)

# Update agent with new tools
agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

# Test agent with new capabilities
agent_executor({"input": "What is the weather in Delhi?"})
```

**Output**:
```
> Invoking: `Search` with `{'query': 'current weather in Delhi'}`
...weather details...
```

---

## Developing the Front-End with Streamlit

To create a user-friendly interface, use **Streamlit** to build the web application.

### Code Example: Basic Streamlit Interface

```python
import streamlit as st

# Initialize the conversation chain
st.title("GlobeBotter - Your Travel Assistant")
query = st.text_input("Ask GlobeBotter")

if st.button("Submit"):
    output = agent_executor({"input": query})
    st.write("AI system:", output['response'])
```

**Expected Interface**:
- Text input box for user queries
- Output area displaying bot responses

---

## Summary
In this section, we built **GlobeBotter**, a conversational travel assistant that uses LangChain to manage memory, integrate external documents, and dynamically select tools for improved responses. We extended its functionality with SerpApi for real-time data and developed a basic web front-end using **Streamlit**.

---

## References
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [SerpApi Documentation](https://serpapi.com/)