# RAG Chatbot with Document Processing and API/URL Data Fetching

This is a Streamlit-based application that allows users to:
- Upload documents (PDF, DOCX, TXT).
- Extract text from webpages and store it.
- Fetch data from external APIs.
- Store and manage documents in a Weaviate database.
- Query the stored data using a chat interface with context-aware responses.

## Features
- **Document Uploads**: Supports PDF, DOCX, and TXT files, with previews before storage.
- **API Data Fetching**: Fetches and stores content from external APIs.
- **URL Content Extraction**: Extracts text content from any given webpage URL.
- **Chat Interface**: Uses OpenAI’s GPT-3.5 to answer questions based on stored content.
- **Document Management**: View and delete documents directly from the interface.

## Installation

Follow these steps to set up the project:

### Prerequisites
- Python 3.8+
- [Weaviate](https://weaviate.io/) instance set up and running (locally or on the cloud)
- OpenAI API key
