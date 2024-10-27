import streamlit as st
from openai import OpenAI
import weaviate
from bs4 import BeautifulSoup
import requests
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
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

# Function to fetch all documents from Weaviate
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

# Function to delete a document from Weaviate
def delete_document_from_weaviate(doc_id: str) -> bool:
    """Delete a document from the Weaviate database using its ID."""
    try:
        client.data_object.delete(doc_id, class_name="Document")
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

# Function for OCR using pytesseract
def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text from images in a PDF using OCR."""
    try:
        images = convert_pdf_to_images(pdf_bytes)
        extracted_text = ""
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        st.error(f"Error during OCR processing: {str(e)}")
        return ""

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert each page of a PDF to an image for OCR processing using pdf2image."""
    try:
        images = convert_from_bytes(pdf_bytes)
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []

def improve_text_clarity(text: str, max_tokens_per_chunk: int = 3000) -> str:
    """Enhance the readability and clarity of the text using GPT in manageable chunks."""
    chunks = chunk_text(text, chunk_size=max_tokens_per_chunk)
    improved_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            messages = [
                {"role": "system", "content": "You are an assistant tasked with improving the clarity and readability of a text without introducing any factual inaccuracies."},
                {"role": "user", "content": f"Improve the following text:\n\n{chunk}"}
            ]
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            improved_text = response.choices[0].message.content
            improved_chunks.append(improved_text)
        except Exception as e:
            st.error(f"Error improving chunk {i+1} with GPT: {str(e)}")
            improved_chunks.append(chunk)  # Add the original chunk if improvement fails
    
    return "\n\n".join(improved_chunks)

def verify_no_hallucinations(original_text: str, improved_text: str, max_tokens_per_chunk: int = 3000) -> bool:
    """Check if the improved text introduces any hallucinations or inaccuracies in manageable chunks."""
    original_chunks = chunk_text(original_text, chunk_size=max_tokens_per_chunk)
    improved_chunks = chunk_text(improved_text, chunk_size=max_tokens_per_chunk)
    
    for i, (orig_chunk, imp_chunk) in enumerate(zip(original_chunks, improved_chunks)):
        try:
            messages = [
                {"role": "system", "content": "You are a verification assistant. Compare two versions of a text and ensure the improved version does not introduce any hallucinations or inaccuracies."},
                {"role": "user", "content": f"Original text:\n{orig_chunk}\n\nImproved text:\n{imp_chunk}\n\nDoes the improved version introduce any inaccuracies? Respond with 'yes' or 'no'."}
            ]
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=10
            )
            verification = response.choices[0].message.content.strip().lower()
            if "yes" in verification:
                return False  # If any chunk introduces inaccuracies, fail the verification
        except Exception as e:
            st.error(f"Error verifying chunk {i+1} with GPT: {str(e)}")
            return False
    
    return True

def setup_weaviate():
    """Initialize Weaviate schema if it doesn't exist."""
    try:
        existing_schema = client.schema.get()
        schema_exists = any(class_obj["class"] == "Document" for class_obj in existing_schema["classes"]) if "classes" in existing_schema else False
        if not schema_exists:
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
            client.schema.create_class(schema)
            st.success("Schema created successfully")
    except Exception as e:
        st.error(f"Error setting up schema: {str(e)}")

def add_to_weaviate(content: str, source: str) -> bool:
    """Add content to Weaviate database."""
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
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(' '.join(chunk_words))
        i += (chunk_size - overlap)
    return chunks

def format_timestamp() -> str:
    """Format current timestamp in RFC3339 format."""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def query_weaviate(query: str, limit: int = 10) -> List[Dict]:
    """Query Weaviate database for relevant content based on a user query."""
    try:
        response = (
            client.query
            .get("Document", ["content", "source", "timestamp"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        documents = response.get("data", {}).get("Get", {}).get("Document", [])
        return documents
    except Exception as e:
        st.error(f"Error querying Weaviate: {str(e)}")
        return []

def get_chatgpt_response(query: str, context: List[Dict]) -> str:
    """Get response from ChatGPT using retrieved context."""
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
    st.title("RAG Chatbot with OCR and Enhanced Text Processing")
    setup_weaviate()
    
    with st.sidebar:
        st.header("Add Documents with OCR Option")
        use_ocr = st.radio("Use OCR for document processing?", options=["No", "Yes"])
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                file_type = file.type.split('/')[-1]
                if file_type == 'plain':
                    file_type = 'txt'
                
                # Process with or without OCR based on user selection
                if use_ocr == "Yes" and file_type == "pdf":
                    st.write(f"Performing OCR on {file.name}...")
                    content = extract_text_with_ocr(file.getvalue())
                else:
                    if file_type == "pdf":
                        content = extract_text_from_pdf(io.BytesIO(file.getvalue()))
                    elif file_type == "docx":
                        content = extract_text_from_docx(io.BytesIO(file.getvalue()))
                    else:
                        content = file.getvalue().decode('utf-8')
                
                # Improve text clarity using GPT
                st.write(f"Improving readability of {file.name}...")
                improved_content = improve_text_clarity(content)
                
                # Verify that no hallucinations are introduced
                st.write(f"Verifying improvements for {file.name}...")
                is_verified = verify_no_hallucinations(content, improved_content)
                
                if is_verified:
                    st.success(f"{file.name} has been verified and improved successfully.")
                    if st.button(f"Add {file.name} to Weaviate", key=f"add_{i}"):
                        if add_to_weaviate(improved_content, file.name):
                            st.success(f"Successfully added {file.name} to Weaviate.")
                        else:
                            st.error(f"Failed to add {file.name} to Weaviate.")
                else:
                    st.warning(f"Verification failed for {file.name}. The improved version may contain inaccuracies and has not been added.")

                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processed {file.name} ({i + 1} of {len(uploaded_files)})")
            
            # Reset progress bar and status after processing
            progress_bar.empty()
            status_text.empty()
    
    # Document Library Section
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
    
    # Chat Interface
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Query Weaviate for relevant documents
        relevant_docs = query_weaviate(prompt)
        
        # Get GPT response
        with st.chat_message("assistant"):
            response = get_chatgpt_response(prompt, relevant_docs)
            st.markdown(response)
            
            # Highlight sources used in the response
            if relevant_docs:
                st.write("Sources used in the response:")
                for doc in relevant_docs:
                    source = doc['source']
                    full_content = doc['content']
                    st.markdown(f"**Source**: {source}")
                    st.text_area(f"Full content from {source}:", full_content, height=300)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
