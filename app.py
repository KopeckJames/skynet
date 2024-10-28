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

# Initialize OpenAI and Weaviate clients
client_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = weaviate.Client(
    url=st.secrets["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
    additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]}
)
# Define a processing queue
if "processing_queue" not in st.session_state:
    st.session_state.processing_queue = Queue()
def transcribe_audio(audio_file: io.BytesIO) -> str:
    """Transcribe audio using OpenAI's Whisper API."""
    try:
        audio_bytes = audio_file.read()
        # Use OpenAI's Whisper API to transcribe
        transcription = client_openai.Audio.transcribe(
            model="whisper-1",  # Use the appropriate model here
            file=audio_bytes,
            response_format="text"
        )
        return transcription.get("text", "Transcription failed.")
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""
def extract_text_from_webpage(url: str) -> str:
    """Extract text content from a webpage."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from paragraphs for readability
        text = ' '.join([p.get_text() for p in soup.find_all('p')])

        # Limit the content length to prevent overly long responses
        return text  # Limit to the first 10,000 characters if needed
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch content from the URL: {e}")
        return ""
    except Exception as e:
        st.error(f"An error occurred while processing the webpage: {e}")
        return ""

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """Extract text content from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file: io.BytesIO) -> str:
    """Extract text content from a DOCX file."""
    doc = DocxDocument(docx_file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# Function to create a DOCX file
def create_docx(text: str, filename: str) -> bytes:
    """Create a DOCX file from text."""
    doc = DocxDocument()
    doc.add_paragraph(text)
    byte_io = io.BytesIO()
    doc.save(byte_io)
    byte_io.seek(0)
    return byte_io.getvalue()

# Function to create a PDF file
def create_pdf(text: str, filename: str) -> bytes:
    """Create a PDF file from text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Write each line to the PDF, handling encoding issues gracefully
    for line in text.split('\n'):
        try:
            # Encode each line as latin1, ignoring characters that cannot be encoded
            pdf.multi_cell(0, 10, line.encode('latin1', 'ignore').decode('latin1'))
        except Exception as e:
            st.error(f"Error writing line to PDF: {e}")
    
    # Create a BytesIO object to store the PDF data
    byte_io = io.BytesIO()
    pdf.output(dest='S').encode('latin1')
    byte_io.write(pdf.output(dest='S').encode('latin1'))
    byte_io.seek(0)
    return byte_io.getvalue()

# Function to allow download of the improved OCR document
def download_improved_ocr(content: str):
    """Provide options to download the improved OCR content as PDF or DOCX."""
    st.write("Download the improved OCR document:")
    docx_data = create_docx(content, "improved_ocr.docx")
    pdf_data = create_pdf(content, "improved_ocr.pdf")
    
    st.download_button(
        label="Download as DOCX",
        data=docx_data,
        file_name="improved_ocr.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    
    st.download_button(
        label="Download as PDF",
        data=pdf_data,
        file_name="improved_ocr.pdf",
        mime="application/pdf"
    )
def download_chatbot_document(response: str):
    """Provide options to download the chatbot's response as PDF or DOCX."""
    st.write("Download the generated document:")
    docx_data = create_docx(response, "chatbot_generated.docx")
    pdf_data = create_pdf(response, "chatbot_generated.pdf")
    
    st.download_button(
        label="Download as DOCX",
        data=docx_data,
        file_name="chatbot_generated.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    
    st.download_button(
        label="Download as PDF",
        data=pdf_data,
        file_name="chatbot_generated.pdf",
        mime="application/pdf"
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
                max_tokens=4000
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
                max_tokens=1000
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
def process_materials():
    while not st.session_state.processing_queue.empty():
        file_info = st.session_state.processing_queue.get()
        file_name = file_info["name"]
        file_bytes = file_info["bytes"]
        file_type = file_info["type"]
        use_ocr = file_info["use_ocr"]

        try:
            if use_ocr and file_type == "pdf":
                content = extract_text_with_ocr(file_bytes)
            elif file_type == "pdf":
                content = extract_text_from_pdf(io.BytesIO(file_bytes))
            elif file_type == "docx":
                content = extract_text_from_docx(io.BytesIO(file_bytes))
            else:
                content = file_bytes.decode('utf-8')

            # Improve text clarity using GPT
            improved_content = improve_text_clarity(content)
            # Add to Weaviate database
            add_to_weaviate(improved_content, file_name)

            st.success(f"Processed and added {file_name} to Weaviate.")
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")

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
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting ChatGPT response: {str(e)}")
        return "I apologize, but I encountered an error generating a response."

def start_chat_interface():
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Checkbox to include or exclude sources.
    include_sources = st.checkbox("Include sources in responses", value=True)

    # Display chat history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input.
    if prompt := st.chat_input("What's your question?"):
        # Add user message to chat history.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query Weaviate for relevant documents.
        relevant_docs = query_weaviate(prompt)

        # Get GPT response.
        with st.chat_message("assistant"):
            response = get_chatgpt_response(prompt, relevant_docs)
            st.markdown(response)
            download_chatbot_document(response)

            # Display sources if the checkbox is checked.
            if include_sources and relevant_docs:
                st.write("Sources used in the response:")
                for idx, doc in enumerate(relevant_docs):
                    st.markdown(f"**Source**: {doc['source']}")
                    # Add a unique key using the index or document ID.
                    st.text_area(f"Full content from {doc['source']}:", doc['content'], height=300, key=f"doc_{idx}_{doc['source']}")
        
        # Add assistant response to chat history.
        st.session_state.messages.append({"role": "assistant", "content": response})


# Main UI function
def main():
    st.set_page_config(
        page_title="COGNITEXT",
        page_icon="📚",
        layout="wide"
    )

    # Define tabs for different sections
    tabs = st.tabs(["Main", "Document Library"])

    # Main tab for chat and file handling
    with tabs[0]:
        st.markdown(
            "<h1 style='text-align: center; color: #3366cc;'>Welcome to COGNITEXT</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #666666;'>An AI-powered research and writing assistant that works best with the sources you upload</p>",
            unsafe_allow_html=True
        )
        

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("upload_icon.png", width = 175)
            if st.button("Upload Documents"):
                with st.expander("Upload Documents with OCR Option", expanded=True):
                    use_ocr = st.radio("Use OCR for document processing?", options=["No", "Yes"])
                    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            file_info = {
                                "name": uploaded_file.name,
                                "bytes": uploaded_file.getvalue(),
                                "type": uploaded_file.type.split('/')[-1],
                                "use_ocr": use_ocr == "Yes"
                            }
                            st.session_state.processing_queue.put(file_info)
                            st.success(f"Added {uploaded_file.name} to the processing queue.")

        with col2:
            st.image("url_icon.png", width = 175)
            if st.button("Extract from URL"):
                with st.expander("Extract Content from URL", expanded=True):
                    url = st.text_input("Enter webpage URL")
                    if url:
                        content = extract_text_from_webpage(url)
                        st.text_area("Extracted Content", content[:5000], height=300)
                        if st.button("Add Extracted Content to Weaviate"):
                            if add_to_weaviate(content, url):
                                st.success("Successfully added the extracted content to Weaviate.")
                            else:
                                st.error("Failed to add the extracted content to Weaviate.")

        with col3:
            st.image("audio_icon.png", width = 175)
            if st.button("Upload Audio for Transcription"):
                with st.expander("Upload Audio Files for Transcription", expanded=True):
                    audio_files = st.file_uploader("Upload Audio Files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
                    if audio_files:
                        for audio_file in audio_files:
                            st.info(f"Transcribing {audio_file.name}...")
                            transcription = transcribe_audio(audio_file)
                            st.text_area(f"Transcription of {audio_file.name}", transcription, height=300)
                            if st.button(f"Add Transcription of {audio_file.name} to Weaviate", key=f"add_audio_{audio_file.name}"):
                                if add_to_weaviate(transcription, audio_file.name):
                                    st.success(f"Successfully added the transcription of {audio_file.name} to Weaviate.")
                                else:
                                    st.error(f"Failed to add the transcription of {audio_file.name} to Weaviate.")

        # Process materials button
        st.image("process_icon.png", width = 175)
        if st.button("Process All Materials"):
            process_materials()
            st.success("All materials processed successfully.")

    # Document Library tab for viewing and managing documents
    with tabs[1]:
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
                        st.experimental_rerun()  # Refresh the page to update the list of documents.
        else:
            st.write("No documents found in the library.")

    # Sidebar for Chat Interface
    with st.sidebar:
        st.header("Chat with COGNITEXT")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        include_sources = st.checkbox("Include sources in responses", value=True)

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
                download_chatbot_document(response)

                if include_sources and relevant_docs:
                    st.write("Sources used in the response:")
                    for idx, doc in enumerate(relevant_docs):
                        st.markdown(f"**Source**: {doc['source']}")
                        st.text_area(f"Full content from {doc['source']}:", doc['content'], height=300, key=f"doc_{idx}")

            st.session_state.messages.append({"role": "assistant", "content": response})

# Run the main function
if __name__ == "__main__":
    main()