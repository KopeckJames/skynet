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
from docx import Document as DocxDocument
from fpdf import FPDF
from typing import List, Dict, Tuple
import os
from datetime import datetime
import pandas as pd
import time
import threading
from queue import Queue
import re
import random

# Initialize OpenAI and Weaviate clients
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = weaviate.Client(
    url=st.secrets["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
    additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]}
)

# Initialize session state
if "processing_queue" not in st.session_state:
    st.session_state.processing_queue = Queue()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_citations" not in st.session_state:
    st.session_state.show_citations = True

def verify_weaviate_setup():
    """Verify and setup Weaviate schema."""
    try:
        schema = client.schema.get()
        
        # Check if Document class exists
        document_class = next((class_obj for class_obj in schema.get("classes", []) 
                             if class_obj["class"] == "Document"), None)
        
        if not document_class:
            # Create schema if it doesn't exist
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
            st.success("Weaviate schema initialized successfully")
        
        return True
            
    except Exception as e:
        st.error(f"Error in Weaviate setup: {str(e)}")
        return False

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into smaller chunks with overlap."""
    if not text:
        return []
    
    # Split into sentences and filter empty ones
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chars and current_chunk:
            # Store current chunk and start new one with overlap
            chunks.append(' '.join(current_chunk))
            # Keep last few sentences for overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text."""
    return len(text.split()) * 1.3  # Approximate tokens per word
def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """Extract text content from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = []
        for page in pdf_reader.pages:
            content = page.extract_text() or ""
            if content.strip():
                text.append(content)
        return "\n\n".join(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file: io.BytesIO) -> str:
    """Extract text content from a DOCX file."""
    try:
        doc = DocxDocument(docx_file)
        text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n\n".join(text)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text from images in a PDF using OCR."""
    try:
        images = convert_from_bytes(pdf_bytes)
        text_chunks = []
        for image in images:
            text = pytesseract.image_to_string(image)
            if text.strip():
                text_chunks.append(text)
        return "\n\n".join(text_chunks)
    except Exception as e:
        st.error(f"Error performing OCR: {str(e)}")
        return ""

def extract_text_from_webpage(url: str) -> str:
    """Enhanced webpage content extraction."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript', 'meta', 'link']):
            element.decompose()

        # Extract content with multiple strategies
        content = []
        
        # Strategy 1: Main content areas
        main_content = (
            soup.find('article') or 
            soup.find('main') or 
            soup.find('div', class_=['content', 'main', 'post', 'article', 'entry-content', 'page-content']) or
            soup.find('div', {'role': 'main'})
        )
        
        if main_content:
            # Get text from headers and paragraphs in main content
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = element.get_text().strip()
                if text and len(text) > 15:  # Filter very short snippets
                    content.append(text)
        else:
            # Strategy 2: Get all substantive text elements
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = element.get_text().strip()
                if text and len(text) > 15:
                    content.append(text)

        # Process and structure content
        processed_content = []
        current_section = ""
        
        for text in content:
            # Check if it's a heading
            if len(text) < 100 and (text.endswith(':') or text.isupper() or 
                any(text.startswith(h) for h in ['Introduction', 'What', 'How', 'Why'])):
                if current_section:
                    processed_content.append(current_section)
                current_section = f"## {text}\n\n"
            else:
                current_section += f"{text}\n\n"
        
        if current_section:
            processed_content.append(current_section)

        # Join and clean text
        final_text = "\n".join(processed_content)
        
        # Clean up the text
        final_text = re.sub(r'\n\s*\n', '\n\n', final_text)  # Fix newlines
        final_text = re.sub(r'\s+', ' ', final_text)  # Fix spaces
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)  # Limit consecutive newlines
        
        if not final_text.strip():
            st.error("No content could be extracted from the URL")
            return ""
        
        st.success(f"Successfully extracted {len(final_text)} characters from URL")
        return final_text
        
    except Exception as e:
        st.error(f"Error extracting content from URL: {str(e)}")
        return ""

def add_to_weaviate(content: str, source: str) -> bool:
    """Add content to Weaviate database with enhanced verification."""
    try:
        if not content or not content.strip():
            st.error(f"Empty content for source: {source}")
            return False
            
        # Debug info
        st.write(f"Processing content for {source} (Length: {len(content)} chars)")
        
        # Create smaller chunks for storage
        chunks = chunk_text(content, max_chars=2000)
        st.write(f"Created {len(chunks)} chunks")
        
        success_count = 0
        for i, chunk in enumerate(chunks, 1):
            try:
                # Create the document object
                object_uuid = client.data_object.create(
                    class_name="Document",
                    data_object={
                        "content": chunk,
                        "source": f"{source} (Part {i}/{len(chunks)})" if len(chunks) > 1 else source,
                        "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    }
                )
                
                # Verify storage
                stored_doc = client.data_object.get_by_id(object_uuid, class_name="Document")
                if stored_doc:
                    success_count += 1
                    st.success(f"✓ Chunk {i}/{len(chunks)} stored successfully")
                else:
                    st.error(f"Failed to verify storage of chunk {i}")
                    
            except Exception as e:
                st.error(f"Error storing chunk {i}: {str(e)}")
                continue
        
        return success_count > 0
        
    except Exception as e:
        st.error(f"Error in add_to_weaviate: {str(e)}")
        return False

def process_url_content(url: str) -> bool:
    """Process URL content with enhanced verification."""
    try:
        with st.spinner("Extracting content from URL..."):
            content = extract_text_from_webpage(url)
            
            if not content:
                st.error("No content extracted from URL")
                return False
            
            st.info(f"Extracted {len(content)} characters from URL")
            
            # Store content
            if add_to_weaviate(content, url):
                # Verify storage
                if verify_document_in_weaviate(url):
                    st.success("URL content stored successfully")
                    return True
                else:
                    st.error("Failed to verify URL content storage")
                    return False
            else:
                st.error("Failed to store URL content")
                return False
            
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return False

def verify_document_in_weaviate(source_name: str) -> bool:
    """Enhanced verification of document existence in Weaviate."""
    try:
        response = (
            client.query
            .get("Document", ["content", "source"])
            .with_where({
                "path": ["source"],
                "operator": "Like",
                "valueString": f"*{source_name}*"
            })
            .do()
        )
        
        documents = response.get("data", {}).get("Get", {}).get("Document", [])
        
        if documents:
            st.write(f"Found {len(documents)} chunks for {source_name}")
            return True
            
        st.warning(f"No documents found for {source_name}")
        return False
        
    except Exception as e:
        st.error(f"Error verifying document: {str(e)}")
        return False

def improved_query_weaviate(query: str, limit: int = 2) -> List[Dict]:
    """Enhanced Weaviate query with debug info."""
    try:
        st.write("Searching documents...")
        
        # Semantic search
        semantic_response = (
            client.query
            .get("Document", ["content", "source", "timestamp"])
            .with_near_text({
                "concepts": [query],
                "certainty": 0.6  # Lowered threshold for better recall
            })
            .with_limit(limit * 2)  # Get more results initially for better filtering
            .do()
        )
        
        semantic_matches = semantic_response.get("data", {}).get("Get", {}).get("Document", [])
        st.write(f"Found {len(semantic_matches)} semantic matches")

        # Keyword search
        keywords = query.lower().split()
        where_filter = {
            "operator": "Or",
            "operands": [
                {
                    "path": ["content"],
                    "operator": "Like",
                    "valueString": f"*{word}*"
                } for word in keywords
            ]
        }
        
        keyword_response = (
            client.query
            .get("Document", ["content", "source", "timestamp"])
            .with_where(where_filter)
            .with_limit(limit * 2)
            .do()
        )
        
        keyword_matches = keyword_response.get("data", {}).get("Get", {}).get("Document", [])
        st.write(f"Found {len(keyword_matches)} keyword matches")

        # Combine and deduplicate results
        all_matches = []
        seen_sources = set()
        
        for matches in [semantic_matches, keyword_matches]:
            for doc in matches:
                source = doc['source'].split(" (Part")[0]
                if source not in seen_sources:
                    query_words = set(query.lower().split())
                    content_words = set(doc['content'].lower().split())
                    if query_words & content_words:
                        all_matches.append(doc)
                        seen_sources.add(source)

        st.write(f"Returning {len(all_matches[:limit])} most relevant matches")
        return all_matches[:limit]

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def process_materials():
    """Process queue with enhanced verification."""
    if st.session_state.processing_queue.empty():
        st.warning("No materials in the processing queue.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = st.session_state.processing_queue.qsize()
    files_processed = 0

    while not st.session_state.processing_queue.empty():
        try:
            file_info = st.session_state.processing_queue.get()
            filename = file_info["name"]
            
            status_text.text(f"Processing {filename}...")
            
            # Extract content
            content = process_document(file_info)
            
            if content:
                st.info(f"Extracted {len(content)} characters from {filename}")
                
                # Store content
                if add_to_weaviate(content, filename):
                    # Verify storage
                    if verify_document_in_weaviate(filename):
                        st.success(f"Successfully processed {filename}")
                    else:
                        st.error(f"Failed to verify storage of {filename}")
                else:
                    st.error(f"Failed to store content for {filename}")
            else:
                st.warning(f"No content extracted from {filename}")

            files_processed += 1
            progress_bar.progress(files_processed / total_files)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()
    
    # Final verification
    st.write("Verifying processed documents...")
    list_all_documents()

def process_document(file_info: Dict) -> str:
    """Process a document and extract its text content."""
    try:
        content = ""
        if file_info["type"] == "pdf":
            if file_info.get("use_ocr", False):
                content = extract_text_with_ocr(file_info["bytes"])
            else:
                content = extract_text_from_pdf(io.BytesIO(file_info["bytes"]))
        elif file_info["type"] == "docx":
            content = extract_text_from_docx(io.BytesIO(file_info["bytes"]))
        elif file_info["type"] == "txt":
            content = file_info["bytes"].decode("utf-8")
            
        return content.strip()
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return ""
def get_chatgpt_response(prompt: str, context: List[Dict]) -> str:
    """Generate response with strict token management."""
    if not context:
        return "I don't have any relevant information about that in my document library."
    
    def truncate_text(text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        words = text.split()
        estimated_words = int(max_tokens / 1.3)
        return ' '.join(words[:estimated_words])
    
    try:
        # Calculate token budgets
        system_prompt_tokens = 500
        user_prompt_tokens = len(prompt) // 4
        response_tokens = 4000
        
        # Available tokens for context
        max_context_tokens = 16385 - system_prompt_tokens - user_prompt_tokens - response_tokens
        tokens_per_doc = max_context_tokens // max(len(context), 1)
        
        # Process context
        processed_context = []
        for doc in context:
            content = doc['content']
            if estimate_tokens(content) > tokens_per_doc:
                content = truncate_text(content, tokens_per_doc)
            processed_context.append(f"Document '{doc['source']}':\n{content}")
        
        context_text = "\n\n".join(processed_context)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Your task is to provide accurate, "
                    "comprehensive answers based on the provided documents. When answering:\n"
                    "1. Use only the information from the provided documents\n"
                    "2. If you find relevant information about a person, provide thorough background details\n"
                    "3. Always cite your sources using the document names\n"
                    "4. Structure your response with clear sections when appropriate\n"
                    "5. If you're not sure about something, say so clearly"
                )
            },
            {
                "role": "user",
                "content": f"Based on the provided documents, please answer this question: {prompt}\n\nRelevant documents:\n\n{context_text}"
            }
        ]
        
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I encountered an error while generating a response. Please try again."



def delete_document_from_weaviate(doc_id: str) -> bool:
    """Delete a document from Weaviate."""
    try:
        client.data_object.delete(doc_id, class_name="Document")
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def delete_all_documents():
    """Delete all documents from Weaviate."""
    try:
        response = (
            client.query
            .get("Document")
            .with_additional(["id"])
            .do()
        )
        
        documents = response.get("data", {}).get("Get", {}).get("Document", [])
        for doc in documents:
            delete_document_from_weaviate(doc['_additional']['id'])
            
        st.success("All documents deleted successfully")
        return True
    except Exception as e:
        st.error(f"Error deleting all documents: {str(e)}")
        return False

def list_all_documents() -> None:
    """List all documents in Weaviate."""
    try:
        response = (
            client.query
            .get("Document", ["content", "source", "timestamp"])
            .with_additional(["id"])
            .do()
        )
        
        documents = response.get("data", {}).get("Get", {}).get("Document", [])
        
        st.write(f"Total documents in database: {len(documents)}")
        
        if documents:
            # Group by source (without part numbers)
            grouped_docs = {}
            for doc in documents:
                source = doc['source'].split(" (Part")[0]
                if source not in grouped_docs:
                    grouped_docs[source] = []
                grouped_docs[source].append(doc)
            
            # Display grouped documents
            for source, docs in grouped_docs.items():
                st.write(f"\n**Source:** {source}")
                st.write(f"Number of chunks: {len(docs)}")
                st.write(f"Total content length: {sum(len(d['content']) for d in docs)} characters")
                st.write("Preview of first chunk:")
                st.text(docs[0]['content'][:200] + "...")
                st.write("---")
        else:
            st.warning("No documents found in the database")
            
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file."""
    if not uploaded_file:
        return
        
    file_info = {
        "name": uploaded_file.name,
        "bytes": uploaded_file.getvalue(),
        "type": uploaded_file.type.split('/')[-1],
        "use_ocr": False
    }
    
    st.write(f"Processing {file_info['name']}...")
    
    try:
        content = process_document(file_info)
        
        if content:
            chunks = chunk_text(content)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                chunk_source = file_info["name"] if total_chunks == 1 else f"{file_info['name']} (Part {i}/{total_chunks})"
                if add_to_weaviate(chunk, chunk_source):
                    st.success(f"Added part {i} of {total_chunks}")
                else:
                    st.error(f"Failed to add part {i}")
        else:
            st.warning(f"No content extracted from {file_info['name']}")
            
    except Exception as e:
        st.error(f"Error processing {file_info['name']}: {str(e)}")

def process_url_content(url: str) -> bool:
    """Process URL content."""
    try:
        with st.spinner("Extracting content from URL..."):
            content = extract_text_from_webpage(url)
            
            if not content:
                st.error("No content extracted from URL")
                return False
            
            # Split into chunks and store
            chunks = chunk_text(content)
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                st.error("No valid content chunks created")
                return False
            
            # Process each chunk
            success_count = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_source = f"{url}" if total_chunks == 1 else f"{url} (Part {i}/{total_chunks})"
                
                if add_to_weaviate(chunk, chunk_source):
                    success_count += 1
                    st.success(f"Processed chunk {i}/{total_chunks}")
                else:
                    st.error(f"Failed to process chunk {i}/{total_chunks}")
            
            return success_count > 0
            
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return False
def add_document_management_ui():
    """Add document management UI to sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("Document Management")
    
    # List documents with delete options
    response = (
        client.query
        .get("Document", ["source", "timestamp"])
        .with_additional(["id"])
        .do()
    )
    
    documents = response.get("data", {}).get("Get", {}).get("Document", [])
    
    if documents:
        # Group documents by source
        grouped_docs = {}
        for doc in documents:
            source = doc['source'].split(" (Part")[0]
            if source not in grouped_docs:
                grouped_docs[source] = []
            grouped_docs[source].append(doc)
        
        # Display grouped documents with delete options
        for source, docs in grouped_docs.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"📄 {source}")
                st.write(f"Parts: {len(docs)}")
            with col2:
                if st.button("Delete", key=f"delete_{source}"):
                    with st.spinner(f"Deleting {source}..."):
                        for doc in docs:
                            delete_document_from_weaviate(doc['_additional']['id'])
                        st.success(f"Deleted {source}")
                        st.rerun()
        
        # Add delete all option
        st.sidebar.markdown("---")
        if st.sidebar.button("Delete All Documents", type="secondary"):
            if st.sidebar.checkbox("Confirm deletion of all documents"):
                if delete_all_documents():
                    st.sidebar.success("All documents deleted")
                    st.rerun()
    else:
        st.sidebar.info("No documents in database")
def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="COGNTEXT",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for coloring
    st.markdown("""
    <style>
        .stApp {
            background-color: #a9c9c4;
        }
        .stButton>button {
            background-color: #a7bdcd;
            color: #505a5d;
        }
        .stTextInput>div>div>input {
            background-color: #9fb9e6;
        }
        .stHeader {
            color: #638296;
        }
        /* Additional styles to ensure readability and consistency */
        .stTextArea textarea {
            background-color: #9fb9e6;
            color: #505a5d;
        }
        .stSelectbox>div>div>select {
            background-color: #9fb9e6;
        }
        .stCheckbox>label>span {
            color: #505a5d;
        }
        /* Ensure text color consistency */
        .stApp {
            color: #505a5d;
        }
    </style>
    """, unsafe_allow_html=True)
    # Rest of your main() function code...
    # Initialize session states if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = Queue()
    if "show_citations" not in st.session_state:
        st.session_state.show_citations = True

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Management")
        
        # Document Upload Section
        st.header("Upload Documents")
        # File Upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        # Add document management UI
        add_document_management_ui()
        # URL Input
        url_input = st.text_input(
            "Or enter a URL:",
            placeholder="https://example.com/document"
        )
        
        # OCR Option
        use_ocr = st.checkbox("Use OCR for PDFs")
        
        # Process uploaded files
        if uploaded_files:
            st.markdown("### Uploaded Files")
            for uploaded_file in uploaded_files:
                if not verify_document_in_weaviate(uploaded_file.name):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {uploaded_file.name}")
                    with col2:
                        if st.button("Process", key=f"process_{uploaded_file.name}"):
                            file_info = {
                                "name": uploaded_file.name,
                                "bytes": uploaded_file.getvalue(),
                                "type": uploaded_file.type.split('/')[-1],
                                "use_ocr": use_ocr if uploaded_file.type == "application/pdf" else False
                            }
                            st.session_state.processing_queue.put(file_info)
                            st.success("Added to queue")
                            st.rerun()

        # Process URL
        if url_input and url_input.strip():
            if not verify_document_in_weaviate(url_input):
                if st.button("Process URL"):
                    if process_url_content(url_input):
                        st.success("URL processed successfully")
                        st.rerun()

        # Queue Processing
        st.markdown("---")
        queue_size = st.session_state.processing_queue.qsize()
        if queue_size > 0:
            if st.button(f"Process Queue ({queue_size} items)", use_container_width=True):
                process_materials()
                st.rerun()

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify Database"):
                verify_weaviate_setup()
        with col2:
            if st.button("List Documents"):
                list_all_documents()

        # Citations toggle
        st.markdown("---")
        st.header("Chat Settings")
        st.session_state.show_citations = st.checkbox(
            "Show Source Citations",
            value=st.session_state.show_citations
        )

    # Main Chat Interface
    st.title("💬 Chat with COGNTEXT")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and st.session_state.show_citations:
                with st.expander("View Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source']}**")
                        st.markdown(source['content'][:500] + "...")

    # Chat input
    if prompt := st.chat_input("What would you like to know about?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                relevant_docs = improved_query_weaviate(prompt)
                if relevant_docs:
                    response = get_chatgpt_response(prompt, relevant_docs)
                    st.markdown(response)
                    
                    # Store message with sources
                    message_data = {
                        "role": "assistant",
                        "content": response,
                        "sources": relevant_docs if st.session_state.show_citations else []
                    }
                    st.session_state.messages.append(message_data)
                else:
                    message = "I don't have any relevant information about that. Please upload some relevant documents."
                    st.warning(message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message,
                        "sources": []
                    })

if __name__ == "__main__":
    verify_weaviate_setup()
    main()
