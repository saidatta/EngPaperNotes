## Introduction to Streamlit for LLM-Powered Applications
Streamlit is a powerful Python library designed to simplify the development of web applications. It is user-friendly and does not require advanced front-end development skills, making it ideal for rapid prototyping and sharing interactive data-driven applications. As of July 2023, Streamlit integrated with **LangChain**, providing a robust platform for building conversational applications with clear, user-friendly interfaces.

### Key Capabilities:
- **Rapid GUI Development**: Build web apps with minimal code.
- **LangChain Integration**: Use LangChain’s callback handler for visualizing LLM interactions and agent decision-making steps.
- **Customizable UI**: Easily incorporate text inputs, buttons, containers, and session states.

## Creating Your Streamlit App: GlobeBotter

### Step 1: Setting Up the Streamlit App Configuration
The first step in building a Streamlit app is configuring the app layout and appearance.

```python
import streamlit as st

# Set up the page configuration for the app
st.set_page_config(page_title="GlobeBotter", page_icon="🌍")
st.header('Welcome to GlobeBotter, your travel assistant with Internet access. What are you planning for your next trip?')
```

### Step 2: Initializing LangChain Components
To power our chatbot, we need to initialize key LangChain components such as LLMs, memory, and tools.

```python
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import SerpAPIWrapper, Tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["SERPAPI_API_KEY"] = "your_serpapi_api_key"

# Set up document loader, vector store, and memory
search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")

# Initialize LLM and create agent
llm = OpenAI()
tools = [
    Tool.from_function(func=search.run, name="Search", description="useful for when you need to answer questions about current events"),
    create_retriever_tool(db.as_retriever(), "italy_travel", "Searches and returns documents regarding Italy.")
]
agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)
```

### Step 3: Creating the User Input Box
Create a text input widget where users can type their questions.

```python
user_query = st.text_input(
    "**Where are you planning your next vacation?**",
    placeholder="Ask me anything!"
)
```

### Step 4: Configuring Session State for Memory Persistence
Session state in Streamlit allows variables to persist across reruns for each user session. This is essential for maintaining conversation history and memory.

```python
# Initialize session states for storing messages and memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory
```

### Step 5: Displaying the Conversation
Create a loop to display the conversation history stored in `st.session_state["messages"]`.

```python
# Display conversation history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
```

### Step 6: Responding to User Input
Set up logic for the AI assistant to respond when given a user query. This example also uses a callback handler to display intermediate agent actions.

```python
from langchain.callbacks.streamlit import StreamlitCallbackHandler

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())  # Callback for real-time updates
        response = agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
```

### Step 7: Adding a Button to Clear Conversation History
Allow users to clear the chat history and start a new session.

```python
# Add a reset button to clear chat history
if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []
    st.experimental_rerun()  # Refresh the app to reflect changes
```

## Example of Real-Time Streaming Responses
Implementing a streaming response provides users with an engaging, real-time experience. This is achieved using the `BaseCallbackHandler` class.

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
import streamlit as st

# Custom callback handler for streaming LLM tokens
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Integrate the handler into the app
with st.chat_message("assistant"):
    stream_handler = StreamHandler(st.empty())
    llm = OpenAI(streaming=True, callbacks=[stream_handler])
    response = llm.invoke(st.session_state.messages)
    st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
```

### Explanation of Streaming Logic:
- **`StreamHandler`**: A custom class to capture and display tokens as they are generated by the LLM.
- **Real-Time Updates**: The `on_llm_new_token` method appends tokens to the displayed text, giving the user a sense of continuous interaction.

## Final Result
The result is an interactive, context-aware travel assistant capable of handling complex queries, accessing real-time web data, and displaying memory-based, multi-step conversational outputs. The app showcases intermediate steps when using LangChain’s callback handler, enriching the user’s understanding of the LLM’s decision-making process.

## Conclusion
Building the **GlobeBotter** app with Streamlit and LangChain provides an end-to-end demonstration of how to leverage LLMs in practical applications. The use of LangChain's modules, such as memory, agents, and retrievers, combined with the simplicity of Streamlit's interface, makes it a powerful combination for rapid development and deployment of conversational applications.

## References:
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **LangChain GitHub Repository**: [LangChain Repo](https://github.com/langchain-ai/langchain)
- **SerpAPI Documentation**: [SerpAPI](https://serpapi.com/)
- **LangChain’s Streamlit Callback**: [LangChain Callback Code](https://github.com/langchain-ai/streamlit-agent)

By following these detailed steps, Staff+ engineers will be able to create complex, interactive applications that harness the power of LLMs, external data sources, and user-friendly web interfaces.