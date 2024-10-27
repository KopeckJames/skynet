# RAG Chatbot with Document Processing and API/URL Data Fetching

This is a Streamlit-based application that allows users to:
- Upload documents (PDF, DOCX, TXT).
- Extract text from webpages and store it.
- Fetch data from external APIs.
- Store and manage documents in a Weaviate database.
- Query the stored data using a chat interface with context-aware responses powered by GPT.

## Features
- **Document Uploads**: Supports PDF, DOCX, and TXT files, with previews before storage.
- **API Data Fetching**: Fetches and stores content from external APIs.
- **URL Content Extraction**: Extracts text content from any given webpage URL.
- **Chat Interface**: Uses OpenAI’s GPT-3.5 to answer questions based on stored content.
- **Document Management**: View stored documents and delete any unwanted entries.

## Prerequisites
- Python 3.8+
- [Weaviate](https://weaviate.io/) instance set up and running (locally or on the cloud)
- OpenAI API key

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

pip install -r requirements.txt

# Add API Keys
Create a .streamlit/secrets.toml file in the root of the project to store your OpenAI and Weaviate API credentials:
# .streamlit/secrets.toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
WEAVIATE_URL = "your-weaviate-instance-url"
WEAVIATE_API_KEY = "your-weaviate-api-key"
#Replace your-openai-api-key, your-weaviate-instance-url, and your-weaviate-api-key with your actual credentials.

Start the Streamlit app using the following command: streamlit run app.py

