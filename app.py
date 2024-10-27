import streamlit as st
from openai import OpenAI
import weaviate
from bs4 import BeautifulSoup
import requests
import PyPDF2
import io
import docx
from typing import List, Dict, Tuple
import os
from datetime import datetime
import pandas as pd
import time
import threading
from queue import Queue

# Initialize OpenAI and Weaviate clients
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = weaviate.Client(
    url=st.secrets["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
    additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]}
)

# Define the schema
schema = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        }
    },
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
        {"name": "timestamp", "dataType": ["date"]}
    ]
}

def format_timestamp() -> str:
    """Format current timestamp in RFC3339 format"""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

class BatchProcessor:
    def __init__(self):
        self.progress_queue = Queue()
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = []
        self.processing_lock = threading.Lock()

    def reset(self):
        """Reset the processor state"""
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = []
        while not self.progress_queue.empty():
            self.progress_queue.get()

    def process_file(self, file_data: Tuple[str, bytes, str]) -> Tuple[str, str, bool]:
        """Process a single file and return its content for user preview."""
        filepath, content, file_type = file_data
        try:
            if file_type == "pdf":
                text = extract_text_from_pdf(io.BytesIO(content))
            elif file_type == "docx":
                text = extract_text_from_docx(io.BytesIO(content))
            elif file_type == "txt":
                text = content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            return filepath, text, True
        except Exception as e:
            self.failed_files.append((filepath, str(e)))
            return filepath, str(e), False

def setup_weaviate():
    """Initialize Weaviate schema if it doesn't exist"""
    try:
        existing_schema = client.schema.get()
        schema_exists = any(class_obj["class"] == "Document" for class_obj in existing_schema["classes"]) if "classes" in existing_schema else False
        if not schema_exists:
            client.schema.create_class(schema)
            st.success("Schema created successfully")
    except Exception as e:
        st.error(f"Error setting up schema: {str(e)}")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_docx(docx_file) -> str:
    """Extract text content from DOCX file"""
    doc = docx.Document(docx_file)
    return " ".join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_webpage(url: str) -> str:
    """Extract text content from a webpage."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching webpage content: {str(e)}")
        return ""

def fetch_data_from_api(api_url: str, params: Dict[str, str] = None, headers: Dict[str, str] = None) -> str:
    """Fetch data from an external API."""
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {str(e)}")
        return ""

def add_to_weaviate(content: str, source: str) -> bool:
    """Add content to Weaviate database"""
    chunks = chunk_text(content)
    try:
        for chunk in chunks:
            client.data_object.create(
                class_name="Document",
                data_object={
                    "content": chunk,
                    "source": source,
                    "timestamp": format_timestamp()
                }
            )
        return True
    except Exception as e:
        st.error(f"Error adding to Weaviate: {str(e)}")
        return False

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(' '.join(chunk_words))
        i += (chunk_size - overlap)
    return chunks

def fetch_all_documents() -> List[Dict]:
    """Fetch all documents from the Weaviate database."""
    try:
        response = (
            client.query
            .get("Document", ["content", "source", "timestamp"])
            .with_additional(["id"])
            .do()
        )
        return response["data"]["Get"]["Document"]
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def delete_document_from_weaviate(doc_id: str) -> bool:
    """Delete a document from the Weaviate database using its ID."""
    try:
        client.data_object.delete(doc_id, class_name="Document")
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def query_weaviate(query: str, limit: int = 3) -> List[Dict]:
    """Query Weaviate database for relevant content"""
    try:
        response = (
            client.query
            .get("Document", ["content", "source"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        return response["data"]["Get"]["Document"]
    except Exception as e:
        st.error(f"Error querying Weaviate: {str(e)}")
        return []

def get_chatgpt_response(query: str, context: List[Dict]) -> str:
    """Get response from ChatGPT using retrieved context"""
    context_text = "\n".join([doc["content"] for doc in context])
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting ChatGPT response: {str(e)}")
        return "I apologize, but I encountered an error generating a response."

def main():
    st.title("RAG Chatbot with Document Processing and API/URL Data Fetching")
    setup_weaviate()
    
    if 'batch_processor' not in st.session_state:
        st.session_state.batch_processor = BatchProcessor()
    
    with st.sidebar:
        st.header("Add Documents or Fetch Data")
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        st.header("Fetch Data from an External API")
        api_url = st.text_input("Enter the API URL")
        api_params = st.text_area("Enter API parameters (JSON format)", value="{}")
        api_headers = st.text_area("Enter API headers (JSON format)", value="{}")
        
        if api_url and st.button("Fetch Data from API"):
            try:
                params = eval(api_params)
                headers = eval(api_headers)
                fetched_data = fetch_data_from_api(api_url, params=params, headers=headers)
                if fetched_data:
                    st.text_area("Fetched Data", fetched_data[:5000], height=300)
                    if st.button("Add Fetched Data to Weaviate"):
                        if add_to_weaviate(fetched_data, api_url):
                            st.success("Successfully added fetched data to Weaviate.")
                        else:
                            st.error("Failed to add fetched data to Weaviate.")
            except Exception as e:
                st.error(f"Error parsing parameters or headers: {str(e)}")
        
        st.header("Add Webpage Content")
        url = st.text_input("Enter webpage URL")
        if url and st.button("Extract Content from URL"):
            content = extract_text_from_webpage(url)
            if content:
                st.text_area("Extracted Content", content[:5000], height=300)
                if st.button("Add Extracted Content to Weaviate"):
                    if add_to_weaviate(content, url):
                        st.success(f"Content from {url} added to Weaviate.")
                    else:
                        st.error("Failed to add content to Weaviate.")
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            batch_processor = st.session_state.batch_processor
            batch_processor.reset()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                file_type = file.type.split('/')[-1]
                if file_type == 'plain':
                    file_type = 'txt'
                file_name, content, success = batch_processor.process_file((file.name, file.getvalue(), file_type))
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name} ({i + 1} of {len(uploaded_files)})")
                
                if success:
                    st.write(f"**{file_name}** - Preview of extracted content:")
                    st.text_area(f"Content from {file_name}", content[:5000], height=300)
                    if st.button(f"Add {file_name} to Weaviate", key=f"add_{i}"):
                        if add_to_weaviate(content, file_name):
                            st.success(f"Successfully added {file_name} to Weaviate.")
                        else:
                            st.error(f"Failed to add {file_name} to Weaviate.")
                else:
                    st.error(f"Failed to process {file_name}: {content}")
            
            progress_bar.empty()
            status_text.empty()
    
    st.header("Document Library")
    documents = fetch_all_documents()
    if documents:
        df = pd.DataFrame(documents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.dataframe(df[['source', 'timestamp']].rename(columns={'source': 'Document Source', 'timestamp': 'Uploaded At'}))
        
        for idx, doc in enumerate(documents):
            doc_source = doc['source']
            doc_id = doc['_additional']['id']
            if st.button(f"Delete '{doc_source}'", key=f"delete_{idx}"):
                if delete_document_from_weaviate(doc_id):
                    st.success(f"Deleted '{doc_source}' successfully.")
                    st.experimental_set_query_params(refresh=str(time.time()))
    else:
        st.write("No documents found in the library.")
    
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        relevant_docs = query_weaviate(prompt)
        
        with st.chat_message("assistant"):
            response = get_chatgpt_response(prompt, relevant_docs)
            st.markdown(response)
            
            if relevant_docs:
                st.write("Sources used in the response:")
                for doc in relevant_docs:
                    source = doc['source']
                    snippet = doc['content'][:300] + "..."
                    st.markdown(f"**Source**: {source}")
                    st.markdown(f"> {snippet}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
