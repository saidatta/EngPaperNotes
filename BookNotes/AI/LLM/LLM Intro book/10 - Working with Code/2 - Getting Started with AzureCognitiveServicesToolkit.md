## Overview
LangChain provides native integration with **Azure AI Services** through the **AzureCognitiveServicesToolkit**, enabling developers to create powerful multimodal applications. This toolkit facilitates the use of Azure's image analysis, form recognition, speech-to-text, and text-to-speech capabilities within LangChain agents. By leveraging this toolkit, you can build applications capable of interacting with various media types, enhancing the scope and flexibility of LLM-powered solutions.

### Key Components of AzureCognitiveServicesToolkit:
- **AzureCogsImageAnalysisTool**: Analyzes and extracts metadata from images.
- **AzureCogsSpeech2TextTool**: Converts speech to text.
- **AzureCogsText2SpeechTool**: Synthesizes text to speech with neural voices.
- **AzureCogsFormRecognizerTool**: Performs Optical Character Recognition (OCR) to extract text and data from documents.

### Definition: OCR
**Optical Character Recognition (OCR)** is a technology that converts different types of documents (e.g., scanned papers, PDFs, images) into editable and searchable data. OCR automates data entry and improves accessibility by digitizing and processing printed text.

## Setting Up AzureCognitiveServicesToolkit

### Step-by-Step Guide:

1. **Create a Multi-Service Instance of Azure AI Services**:
   - Follow the instructions [here](https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource?tabs=windows&pivots=azportal) to set up a multi-service resource.
   - This provides a single key and endpoint for accessing multiple AI services.

2. **Configure Environment Variables**:
   Save the following variables in your `.env` file:
   ```plaintext
   AZURE_COGS_KEY = "your-api-key"
   AZURE_COGS_ENDPOINT = "your-endpoint"
   AZURE_COGS_REGION = "your-region"
   OPENAI_API_KEY = "your-openai-api-key"
   ```

3. **Load Environment Variables in Python**:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   azure_cogs_key = os.environ["AZURE_COGS_KEY"]
   azure_cogs_endpoint = os.environ["AZURE_COGS_ENDPOINT"]
   azure_cogs_region = os.environ["AZURE_COGS_REGION"]
   openai_api_key = os.environ['OPENAI_API_KEY']
   ```

4. **Initialize the Toolkit**:
   ```python
   from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit

   toolkit = AzureCognitiveServicesToolkit()
   tools = toolkit.get_tools()

   # Display tool descriptions
   [(tool.name, tool.description) for tool in tools]
   ```

   **Expected Output**:
   ```plaintext
   [
     ('azure_cognitive_services_form_recognizer', 'A wrapper around Azure Cognitive Services Form Recognizer...'),
     ('azure_cognitive_services_speech2text', 'A wrapper around Azure Cognitive Services Speech2Text...'),
     ('azure_cognitive_services_text2speech', 'A wrapper around Azure Cognitive Services Text2Speech...'),
     ('azure_cognitive_services_image_analysis', 'A wrapper around Azure Cognitive Services Image Analysis...')
   ]
   ```

5. **Initialize the Agent**:
   ```python
   from langchain.agents import initialize_agent, AgentType
   from langchain.chat_models import ChatOpenAI

   llm = ChatOpenAI(api_key=openai_api_key)
   agent = initialize_agent(
       tools=tools,
       llm=llm,
       agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   ```

## Leveraging Single and Multiple Tools

### Example 1: Single Tool – Image Analysis
**Goal**: Describe the contents of an image using the Azure image analysis tool.

```python
description = agent.run("What shows the following image?: https://www.stylo24.it/wp-content/uploads/2020/03/fionda.jpg")
print(description)
```

**Expected Output**:
```plaintext
> Entering new AgentExecutor chain...
Action: azure_cognitive_services_image_analysis
Observation: Caption: a person holding a slingshot
...
Final Answer: The image is of a person holding a slingshot.
```

### Example 2: Reasoning with Images
**Goal**: Ask the agent to reason about the consequences depicted in an image.

```python
agent.run("What happens if the person lets the slingshot go?: https://www.stylo24.it/wp-content/uploads/2020/03/fionda.jpg")
```

**Output**:
```plaintext
> Entering new AgentExecutor chain...
Action: azure_cognitive_services_image_analysis
Observation: Caption: a person holding a slingshot
...
Final Answer: If the person lets go of the slingshot, the object being launched would be released and propelled forward by the tension of the stretched rubber bands.
```

### Example 3: Using Multiple Tools – Storytelling and Audio
**Goal**: Generate a story related to an image and read it aloud.

**Input Image**: ![Sample Image of a Dog](https://i.redd.it/diawvlriobq11.jpg)

```python
agent.run("Tell me a story related to the following picture and read the story aloud to me: https://i.redd.it/diawvlriobq11.jpg")
```

**Output Chain**:
```plaintext
> Entering new AgentExecutor chain...
Action: azure_cognitive_services_image_analysis
Observation: Caption: a dog standing on a snowy hill
...
Action: azure_cognitive_services_text2speech
Observation: Audio saved to temporary file
Final Answer: I hope you enjoyed the story of Snowy the Savior of the Snow...
```

**Python Code to Play Audio**:
```python
from IPython import display

audio = agent.run("Tell me a story related to the following picture and read the story aloud to me: https://i.redd.it/diawvlriobq11.jpg")
display.display(audio)
```

## Customizing the Agent's Behavior

### Inspect the Default Prompt Template:
```python
print(agent.agent.llm_chain.prompt.messages[0].prompt.template)
```

### Customize the Prompt Prefix:
```python
PREFIX = """
You are a story teller for children. 
You read aloud stories based on pictures that the user passes to you.
Always start your story with a welcome message that makes children laugh.
You can use multiple tools to answer the question.
ALWAYS use the tools.
"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={'prefix': PREFIX}
)
```

**Run Custom Agent**:
```python
agent.run("Tell a story related to the following image: https://i.redd.it/diawvlriobq11.jpg")
```

## Building an End-to-End Application: CoPenny for Invoice Analysis

### Tools Used:
- **AzureCogsFormRecognizerTool**: Extracts text and data from invoices.
- **AzureCogsText2SpeechTool**: Reads the extracted data aloud.

### Initialize with Specific Tools:
```python
toolkit = AzureCognitiveServicesToolkit().get_tools()
tools = [toolkit[0], toolkit[2]]  # FormRecognizer and Text2Speech

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### Example Query: Extracting and Reading Invoice Details
**Query**:
```python
agent.run("What are all the men's SKUs in this invoice?: https://www.whiteelysee.fr/design/wp-content/uploads/2022/01/custom-t-shirt-order-form-template-free.jpg")
```

**Audio Response**:
```python
agent.run("Extract women's SKUs from the invoice and read it aloud: https://www.whiteelysee.fr/design/wp-content/uploads/2022/01/custom-t-shirt-order-form-template-free.jpg")
```

### Customize Agent for Automation:
```python
PREFIX = """
You are an AI assistant for invoice analysis.
You extract information from invoices and read it aloud.
Always respond with audio output after extracting information.
Use multiple tools as necessary.
"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={'prefix': PREFIX}
)
```

**Run Custom Agent**:
```python
agent.run("What are the women's SKUs in this invoice?: https://www.whiteelysee.fr/design/wp-content/uploads/2022/01/custom-t-shirt-order-form-template-free.jpg")
```

## Summary
- **AzureCognitiveServicesToolkit** enables multimodal capabilities within LangChain by integrating Azure AI Services tools.
- **Agent Customization** allows tailoring behavior for specific use cases.
- **Multitool Integration** supports complex queries requiring a combination of image, text, and audio processing.
- **Custom Prompts** enhance agent interaction and output format.

## Next Steps
Explore creating agents with even more flexibility using custom models and other cloud services to build comprehensive multimodal applications.