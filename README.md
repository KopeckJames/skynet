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
ou also need to have Tesseract installed on your system. For installation:

On Windows: Download and install Tesseract from here.
On macOS: Use Homebrew:
bash
Copy code
brew install tesseract
On Linux: Install using apt:
bash
Copy code
sudo apt-get install tesseract-ocr
Running the Application:
Clone the repository and navigate to the project directory.
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Add your API keys in .streamlit/secrets.toml as explained before.
Run the Streamlit app:
bash
Copy code
streamlit run app.py
This code ensures that the user has a seamless experience with OCR processing and improved text handling while keeping the integrity of the original content intact. Let me know if you need any more adjustments or explanations!
