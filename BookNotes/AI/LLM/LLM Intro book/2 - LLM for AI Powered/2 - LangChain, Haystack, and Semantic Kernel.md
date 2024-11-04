# AI Orchestrators for LLM-Powered Applications

In this section, we delve into three prominent AI orchestrators—**LangChain**, **Haystack**, and **Semantic Kernel**—that facilitate the integration of Large Language Models (LLMs) into applications. These frameworks provide essential components to build sophisticated AI-powered solutions.

---

## Table of Contents

1. [LangChain](#langchain)
   - [Overview](#overview)
   - [Key Components](#key-components)
   - [Benefits](#benefits)
   - [Example Usage](#example-usage)
2. [Haystack](#haystack)
   - [Overview](#overview-1)
   - [Core Components](#core-components)
   - [Benefits](#benefits-1)
   - [Example Usage](#example-usage-1)
3. [Semantic Kernel](#semantic-kernel)
   - [Overview](#overview-2)
   - [Main Components](#main-components)
   - [Benefits](#benefits-2)
   - [Example Usage](#example-usage-2)
4. [Choosing the Right Framework](#choosing-the-right-framework)
   - [Criteria for Selection](#criteria-for-selection)
   - [Comparison Table](#comparison-table)
5. [Summary](#summary)
6. [References](#references)

---

## LangChain

### Overview

**LangChain** is an open-source framework launched by Harrison Chase in October 2022. It supports both Python and JavaScript/TypeScript (JS/TS) and is designed to develop applications powered by language models. LangChain aims to make these applications data-aware (through grounding) and agentic (able to interact with external environments).

![LangChain Components](https://example.com/figure2.6.png)

*Figure 2.6: LangChain’s components*

### Key Components

1. **Models**:
   - **Definition**: Language models or Large Foundation Models (LFMs) that serve as the engine of the application.
   - **Supported Models**: Proprietary models (e.g., OpenAI's GPT-3/4) and open-source models from the Hugging Face Hub.
   - **Hugging Face Hub**:
     - A platform hosting over 120k models, 20k datasets, and 50k demos in various domains like audio, vision, and language.
     - Allows for the discovery and collaboration on state-of-the-art models.

2. **Data Connectors**:
   - **Purpose**: Retrieve external knowledge to enhance the model's responses.
   - **Examples**:
     - **Document Loaders**: Load data from PDFs, Word documents, websites, etc.
     - **Text Embedding Models**: Convert text into numerical vectors for similarity searches.

3. **Memory**:
   - **Function**: Retain references to user interactions over both short and long terms.
   - **Implementation**:
     - Utilizes vectorized embeddings stored in **Vector Databases (VectorDB)**.
     - **VectorDB Examples**: Chroma, Elasticsearch, Milvus, Pinecone, Qdrant, Weaviate, FAISS.

4. **Chains**:
   - **Definition**: Predetermined sequences of actions and calls to LLMs.
   - **Purpose**: Simplify the construction of complex applications requiring multiple steps.
   - **Example Chain**:
     1. Take the user query.
     2. Chunk it into smaller pieces.
     3. Embed those chunks.
     4. Search for similar embeddings in VectorDB.
     5. Use the top three most similar chunks as context.
     6. Generate the answer.

5. **Agents**:
   - **Definition**: Entities that drive decision-making within applications.
   - **Capabilities**:
     - Access a suite of tools.
     - Decide which tool to call based on user input and context.
     - Dynamic and adaptive in action.

### Benefits

- **Modular Abstractions**: Provides components like prompts, memory, and plug-ins for easier management.
- **Pre-built Chains**: Offers structured concatenations of components for specific use cases.
- **Flexibility**: Components can be customized to fit unique application requirements.
- **Community and Support**: Active development and a growing user community.

### Example Usage

#### Setting Up LangChain with OpenAI's GPT-3

```python
from langchain import OpenAI, ConversationChain

# Initialize the language model
llm = OpenAI(openai_api_key='YOUR_API_KEY', temperature=0.7)

# Create a conversation chain with memory
conversation = ConversationChain(llm=llm, verbose=True)

# Start the conversation
output = conversation.predict(input="Hello, how are you?")
print(output)
```

#### Implementing a Simple Retrieval-Augmented Generation (RAG)

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load documents
documents = ["Document 1 content", "Document 2 content", ...]

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key='YOUR_API_KEY')
vectorstore = FAISS.from_texts(documents, embeddings)

# Create RetrievalQA chain
qa = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
answer = qa.run("What is the main topic of Document 1?")
print(answer)
```

---

## Haystack

### Overview

**Haystack** is a Python-based framework developed by **Deepset**, a Berlin-based startup founded in 2018. It provides tools for building NLP applications, especially those involving search and question-answering systems.

![Haystack Components](https://example.com/figure2.7.png)

*Figure 2.7: Haystack’s components*

### Core Components

1. **Nodes**:
   - **Definition**: Components that perform specific tasks.
   - **Examples**:
     - **Retriever**: Fetches relevant documents.
     - **Reader**: Extracts answers from documents.
     - **Generator**: Generates text responses.
     - **Summarizer**: Summarizes content.
   - **Supported Models**: Proprietary and open-source models, including those from the Hugging Face Hub.

2. **Pipelines**:
   - **Definition**: Sequences of nodes performing NLP tasks.
   - **Types**:
     - **Query Pipelines**: Perform searches and answer queries.
     - **Indexing Pipelines**: Prepare and index documents for search.
   - **Characteristics**: Predetermined and hardcoded.

3. **Agent**:
   - **Definition**: An entity using LLMs to generate responses to complex queries.
   - **Capabilities**:
     - Access a set of tools (nodes or pipelines).
     - Decide which tool to use based on context.
     - Dynamic and adaptive.

4. **Tools**:
   - **Definition**: Functions an agent can call.
   - **Purpose**: Perform NLP tasks or interact with resources.
   - **Toolkits**: Grouped sets of tools for specific objectives.

5. **DocumentStores**:
   - **Definition**: Backends storing and retrieving documents.
   - **Examples**: Elasticsearch, FAISS, Milvus.

### Benefits

- **Ease of Use**: User-friendly for rapid prototyping and lighter tasks.
- **Documentation Quality**: High-quality resources aiding development.
- **End-to-End Framework**: Covers the entire LLM project lifecycle.
- **Deployment Options**: Can be deployed as a REST API for easy consumption.

### Example Usage

#### Setting Up a Basic QA System

```python
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Initialize the document store
document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")

# Write documents to the store
document_store.write_documents([{"content": "Document content here", "meta": {"name": "Doc1"}}])

# Initialize retriever and reader
retriever = DensePassageRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Update embeddings
document_store.update_embeddings(retriever)

# Build pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# Ask a question
prediction = pipe.run(query="What is the main topic of Doc1?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
print(prediction["answers"][0].answer)
```

#### Deploying as a REST API

```bash
# Install Haystack REST API
pip install uvicorn fastapi

# Run the REST API
haystack-api
```

---

## Semantic Kernel

### Overview

**Semantic Kernel** is an open-source SDK developed by **Microsoft**, originally in C# and now also available in Python. It serves as the engine that addresses user input by chaining and concatenating components into pipelines, encouraging **function composition**.

![Semantic Kernel Anatomy](https://example.com/figure2.8.png)

*Figure 2.8: Anatomy of Semantic Kernel*

### Main Components

1. **Models**:
   - **Definition**: LLMs or LFMs serving as the application's engine.
   - **Supported Models**: Proprietary models (e.g., OpenAI, Azure OpenAI) and open-source models from the Hugging Face Hub.

2. **Memory**:
   - **Function**: Retain references to user interactions.
   - **Access Methods**:
     - **Key-Value Pairs**: Store simple information (e.g., names, dates).
     - **Local Storage**: Save information to files (e.g., CSV, JSON).
     - **Semantic Memory Search**: Use embeddings to represent and search text based on meaning.

3. **Functions**:
   - **Definition**: Skills that mix LLM prompts and code to make user requests interpretable and actionable.
   - **Types**:
     - **Semantic Functions**:
       - Templated prompts specifying input and output formats.
       - Incorporate prompt configurations for LLM parameters.
     - **Native Functions**:
       - Native code that routes the intent captured by the semantic function.
       - Perform tasks like data retrieval, API calls, or executing commands.

4. **Plug-ins**:
   - **Definition**: Connectors to external sources or systems.
   - **Purpose**: Provide additional information or perform autonomous actions.
   - **Examples**:
     - **Microsoft Graph Connector Kit**: Access to Microsoft services.
     - **Custom Plug-ins**: Built using combinations of native and semantic functions.

5. **Planner**:
   - **Definition**: A function that takes user tasks and produces action plans.
   - **Purpose**: Auto-create chains or pipelines to address new user needs.

### Benefits

- **Lightweight and C# Support**: Ideal for C# developers and those using the .NET framework.
- **Versatility**: Supports various LLM-related tasks.
- **Industry-Led**: Developed by Microsoft, making it suitable for enterprise-scale applications.
- **Function Composition**: Encourages modular design and reusability.

### Example Usage

#### Setting Up Semantic Kernel in Python

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion

# Initialize the kernel
kernel = sk.Kernel()

# Configure the AI service
kernel.config.add_text_completion_service(
    "davinci", OpenAITextCompletion("text-davinci-003", api_key="YOUR_API_KEY")
)

# Define a semantic function
prompt_template = """{{$input}}
Summarize the above text in one sentence."""

summary_function = kernel.create_semantic_function(prompt_template, max_tokens=50, temperature=0.7)

# Use the function
input_text = "Artificial Intelligence is transforming industries by automating tasks and providing insights."
summary = await kernel.run_async(summary_function(input_text))
print(summary)
```

#### Using Functions and Plug-ins

```python
# Define a native function to fetch data
def get_current_weather(location: str) -> str:
    # Code to fetch weather data
    return f"The current weather in {location} is sunny."

# Register the native function
kernel.register_native_function("GetCurrentWeather", get_current_weather)

# Define a semantic function that uses the native function
prompt_template = """You are a weather assistant.
{{$input}}
Provide the weather forecast using the GetCurrentWeather function."""

weather_function = kernel.create_semantic_function(prompt_template)

# Use the function
user_input = "What's the weather like in New York?"
response = await kernel.run_async(weather_function(user_input))
print(response)
```

---

## Choosing the Right Framework

### Criteria for Selection

1. **Programming Language Support**:
   - **Semantic Kernel**: C#, Python, Java.
   - **LangChain**: Python, JS/TS.
   - **Haystack**: Python.

2. **Task Complexity and Type**:
   - **LangChain**: Excellent for chaining complex tasks.
   - **Haystack**: Ideal for search systems, QA, and conversational AI.
   - **Semantic Kernel**: Versatile, with function composition for complex pipelines.

3. **Customization and Control**:
   - **Semantic Kernel**: Offers granular control with functions and plug-ins.
   - **LangChain**: Highly customizable components and chains.
   - **Haystack**: Flexible pipelines and node configurations.

4. **Documentation and Community Support**:
   - **LangChain**: Active community and extensive examples.
   - **Haystack**: High-quality documentation and tutorials.
   - **Semantic Kernel**: Backed by Microsoft with robust resources.

### Comparison Table

| Feature             | LangChain          | Haystack           | Semantic Kernel     |
|---------------------|--------------------|--------------------|---------------------|
| **LLM Support**     | Proprietary & Open-Source | Proprietary & Open-Source | Proprietary & Open-Source |
| **Languages Supported** | Python, JS/TS    | Python             | C#, Python, Java    |
| **Process Orchestration** | Chains          | Pipelines of Nodes | Pipelines of Functions |
| **Deployment**      | No REST API        | REST API           | No REST API         |
| **Use Cases**       | General-purpose LLM applications | Search, QA, Conversational AI | Enterprise-scale applications with function composition |

---

## Summary

- **LangChain**, **Haystack**, and **Semantic Kernel** are powerful AI orchestrators facilitating the integration of LLMs into applications.
- **LangChain** excels in building complex chains and offers modular components for flexibility.
- **Haystack** is user-friendly, ideal for search and QA systems, and can be deployed as a REST API.
- **Semantic Kernel** encourages function composition, is industry-led by Microsoft, and is suitable for enterprise-scale applications.
- The choice of framework depends on factors like programming language preference, task complexity, customization needs, and available resources.

---

## References

1. **LangChain Repository**: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
2. **Semantic Kernel Documentation**: [https://learn.microsoft.com/en-us/semantic-kernel/](https://learn.microsoft.com/en-us/semantic-kernel/)
3. **Copilot Stack Session**: [Microsoft Build 2023](https://build.microsoft.com/en-US/sessions/bb8f9d99-0c47-404f-8212-a85fffd3a59d)
4. **The Copilot System Video**: [YouTube](https://www.youtube.com/watch?v=E5g20qmeKpg)
5. **Haystack Documentation**: [https://haystack.deepset.ai/](https://haystack.deepset.ai/)
6. **Hugging Face Hub**: [https://huggingface.co/](https://huggingface.co/)
7. **FAISS Library**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

---

# Tags

- #ArtificialIntelligence
- #MachineLearning
- #LargeLanguageModels
- #LangChain
- #Haystack
- #SemanticKernel
- #LLMIntegration
- #AIOrchestrators
- #VectorDatabases
- #FunctionComposition

---

**Next Steps**:

- Experiment with LangChain to build a custom chain for a specific use case.
- Explore deploying a Haystack pipeline as a REST API for a QA system.
- Utilize Semantic Kernel's function composition to create a complex workflow.