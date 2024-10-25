import streamlit as st
import weaviate
from typing import List, Dict
import os
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import pdf2image
import fitz  # PyMuPDF
import cv2
import tempfile
from moviepy.editor import VideoFileClip
import base64
from pathlib import Path
import spacy
from keybert import KeyBERT
import requests
import pandas as pd
from pptx import Presentation
import xlrd
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import os

class MistralProcessor:
    def __init__(self):
        """Initialize Mistral model with stable configuration"""
        try:
            # Use a stable, well-tested model
            self.model_name = "TheBloke/Mistral-7B-v0.1-GGUF"
            
            # Initialize tokenizer with basic configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                use_fast=True,
                legacy=True
            )
            
            # Quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
            
            # Set padding token
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.nlp = spacy.load("en_core_web_sm")
            self.kw_model = KeyBERT()
            
        except Exception as e:
            st.error(f"Detailed error initializing Mistral: {str(e)}")
            raise

    def generate_response(self, prompt: str, max_length=1000, temperature=0.7) -> str:
        """Generate a response with basic configuration"""
        try:
            # Basic prompt formatting
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Cleanup
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return response.strip()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return ""
    def generate_response(self, prompt: str, max_length=1000, temperature=0.7) -> str:
        """Generate a response with optimized memory management"""
        try:
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Format prompt for instruction model
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize with proper padding and truncation
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response (remove the prompt)
            response = response.replace(prompt, "").strip()
            
            # Cleanup
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return ""
            
    def __del__(self):
        """Cleanup when the processor is destroyed"""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
class DocumentProcessor:
    def __init__(self, mistral_processor: MistralProcessor):
        self.mistral = mistral_processor
        self.nlp = spacy.load("en_core_web_sm")
        self.kw_model = KeyBERT()
        
    def process_text(self, content: str) -> str:
        """Process plain text content"""
        return content.strip()
    
    def summarize_text(self, content: str) -> str:
        """Summarize the given text using Mistral"""
        try:
            # Split content into smaller chunks if it is too long
            max_chunk_size = 3000
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            summaries = []
            
            for chunk in chunks:
                prompt = f"Summarize this text concisely:\n{chunk}"
                summary = self.mistral.generate_response(prompt, max_length=150)
                summaries.append(summary)
            
            return " ".join(summaries)
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")
            return ""
    
    def process_image(self, file_path: str) -> Dict[str, str]:
        """Process image files using OCR and Mistral for image description"""
        try:
            # Perform OCR
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
            
            # Get image description using Mistral
            prompt = f"Describe this image based on the OCR text: {ocr_text}"
            description = self.mistral.generate_response(prompt)
            
            return {
                "ocr_text": ocr_text,
                "description": description
            }
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return {"ocr_text": "", "description": ""}

    def process_audio(self, file_path: str) -> str:
        """Process audio files using Llama's audio capabilities"""
        # Note: Implementation would depend on Llama's audio processing capabilities
        # For now, return empty string
        return ""

    def process_video(self, file_path: str) -> Dict[str, str]:
        """Process video files - extract frames for analysis"""
        try:
            # Extract frames and analyze
            cap = cv2.VideoCapture(file_path)
            frames = []
            frame_descriptions = []
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // 5)  # Get 5 evenly spaced frames
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0 and len(frames) < 5:
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_frame:
                        cv2.imwrite(temp_frame.name, frame)
                        frame_info = self.process_image(temp_frame.name)
                        frame_descriptions.append(frame_info["description"])
                
                frame_count += 1
            
            cap.release()
            
            return {
                "frame_descriptions": "\n".join(frame_descriptions)
            }
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return {"frame_descriptions": ""}

    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = []
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text.append(page.get_text())
            
            if not ''.join(text).strip():
                images = pdf2image.convert_from_path(file_path)
                for image in images:
                    text.append(pytesseract.image_to_string(image))
            
            return '\n'.join(text)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    def get_file_type(self, filename: str) -> str:
        """Determine file type based on extension"""
        ext = Path(filename).suffix.lower()
        if ext in ['.txt']:
            return 'text/plain'
        elif ext in ['.pdf']:
            return 'application/pdf'
        elif ext in ['.png', '.jpg', '.jpeg']:
            return 'image/' + ext[1:]
        elif ext in ['.mp3', '.wav']:
            return 'audio/' + ext[1:]
        elif ext in ['.mp4', '.avi']:
            return 'video/' + ext[1:]
        else:
            return 'application/octet-stream'

class RAGApplication:
    def __init__(self, weaviate_url: str, weaviate_api_key: str):
        """Initialize RAG application with simpler configuration"""
        try:
            # Initialize Mistral processor
            self.mistral = MistralProcessor()
            
            # Initialize document processor
            self.processor = DocumentProcessor(self.mistral)
            
            # Initialize Weaviate client
            auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
            self.weaviate_client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=auth_config
            )
            
            self.class_name = "Document"
            self._create_schema()
            
        except Exception as e:
            st.error(f"Error in RAG initialization: {str(e)}")
            raise

    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """Generate a response with simple prompt formatting"""
        try:
            # Format context and query
            context = "\n".join([doc['content'] for doc in context_docs])
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            return self.mistral.generate_response(prompt)
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    def _create_schema(self):
        """Create Weaviate schema"""
        schema = {
            "classes": [{
                "class": self.class_name,
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "vectorizer": "text2vec-transformers"
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "vectorizer": "text2vec-transformers"
                    },
                    {
                        "name": "file_type",
                        "dataType": ["string"]
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"]
                    }
                ]
            }]
        }
        
        try:
            if not self.weaviate_client.schema.exists(self.class_name):
                self.weaviate_client.schema.create_class(schema['classes'][0])
        except Exception as e:
            st.error(f"Error creating schema: {str(e)}")

    def process_file(self, file, filename: str) -> Dict[str, str]:
        """Process uploaded file based on its type"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file.getvalue())
            file_path = temp_file.name
        
        try:
            file_type = self.processor.get_file_type(filename)
            
            if file_type.startswith('image/'):
                result = self.processor.process_image(file_path)
                content = f"OCR Text: {result['ocr_text']}\nImage Description: {result['description']}"
            elif file_type.startswith('audio/'):
                content = self.processor.process_audio(file_path)
            elif file_type.startswith('video/'):
                result = self.processor.process_video(file_path)
                content = f"Video Description: {result['frame_descriptions']}"
            elif file_type == 'application/pdf':
                content = self.processor.process_pdf(file_path)
            else:
                content = self.processor.process_text(file.getvalue().decode())
            
            return {
                "content": content,
                "file_type": file_type,
                "metadata": f"Filename: {filename}, Type: {file_type}"
            }
        finally:
            os.unlink(file_path)

    def add_document(self, title: str, file_data: Dict[str, str]) -> bool:
        """Add a document to the Weaviate database"""
        try:
            properties = {
                "title": title,
                "content": file_data["content"],
                "file_type": file_data["file_type"],
                "metadata": file_data["metadata"]
            }
            
            self.weaviate_client.data_object.create(
                class_name=self.class_name,
                data_object=properties
            )
            return True
        except Exception as e:
            st.error(f"Error adding document: {str(e)}")
            return False

    def search_documents(self, query: str, limit: int = 3) -> List[dict]:
        """Search for relevant documents"""
        try:
            result = (
                self.weaviate_client.query
                .get(self.class_name, ["title", "content", "file_type", "metadata"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            return result.get("data", {}).get("Get", {}).get(self.class_name, [])
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []

    

def verify_environment():
    """Verify that all components are properly installed"""
    try:
        import transformers
        import tokenizers
        import torch
        import bitsandbytes
        
        st.write("Package versions:")
        st.write(f"transformers: {transformers.__version__}")
        st.write(f"tokenizers: {tokenizers.__version__}")
        st.write(f"torch: {torch.__version__}")
        st.write(f"bitsandbytes: {bitsandbytes.__version__}")
        
        if torch.cuda.is_available():
            st.write(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("CUDA is not available")
            
        return True
    except Exception as e:
        st.error(f"Environment verification failed: {str(e)}")
        return False

# Add to main()
def main():
    st.title("📚 Mistral-powered RAG Application")
    
    if not verify_environment():
        st.error("Please check your package installations")
        return
    
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not all([weaviate_url, weaviate_api_key]):
        st.error("Missing required environment variables. Please check your .env file.")
        return
    
    try:
        # Initialize RAG application with Mistral
        rag_app = RAGApplication(weaviate_url, weaviate_api_key)
        st.success("Successfully connected to Weaviate and initialized Mistral model!")
        
        # Sidebar for uploads
        st.sidebar.header("📁 Document Management")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi"]
        )
        
        doc_title = st.sidebar.text_input("Document Title", key="doc_title")
        
        if st.sidebar.button("Process and Add Document"):
            if uploaded_file is not None and doc_title:
                with st.spinner("Processing document..."):
                    try:
                        file_data = rag_app.process_file(uploaded_file, uploaded_file.name)
                        if file_data:
                            if rag_app.add_document(doc_title, file_data):
                                st.success("Document added successfully!")
                                
                                # Show processing results
                                st.markdown("### Document Analysis")
                                
                                # Summary
                                summary = rag_app.processor.summarize_text(file_data['content'])
                                st.markdown("**Summary:**")
                                st.write(summary)
                                
                                # Keywords
                                # Keywords
                                keywords = rag_app.processor.extract_keywords(file_data['content'])
                                st.markdown("**Keywords:**")
                                st.write(", ".join(keywords))
                                
                                # Named Entities
                                entities = rag_app.processor.extract_entities(file_data['content'])
                                st.markdown("**Named Entities:**")
                                for entity_type, entity_list in entities.items():
                                    st.write(f"{entity_type}: {', '.join(entity_list)}")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            else:
                st.warning("Please provide both a file and a title.")
        
        # Main content area
        st.header("🔍 Ask Questions")
        
        # Query interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about your documents?"
            )
        
        with col2:
            search_button = st.button("🔍 Search")
        
        if search_button and query:
            with st.spinner("Searching and generating response..."):
                # Search for relevant documents
                relevant_docs = rag_app.search_documents(query)
                
                if relevant_docs:
                    # Create tabs for response and sources
                    response_tab, sources_tab = st.tabs(["🤖 AI Response", "📚 Sources"])
                    
                    with response_tab:
                        # Generate and display response
                        response = rag_app.generate_response(query, relevant_docs)
                        st.markdown("### Response:")
                        st.markdown(response)
                    
                    with sources_tab:
                        # Display source documents
                        st.markdown("### Retrieved Documents:")
                        for i, doc in enumerate(relevant_docs, 1):
                            with st.expander(f"📄 Source {i}: {doc['title']}"):
                                st.markdown("**Content:**")
                                st.write(doc['content'])
                                st.markdown("**Metadata:**")
                                st.info(doc.get('metadata', 'No metadata available'))
                else:
                    st.warning("No relevant documents found. Try adding some documents first or modify your query.")
        
        # Chat history section
        st.header("💬 Chat History")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Chat input
        user_message = st.text_input("Type your message:", key="chat_input")
        if st.button("Send") and user_message:
            # Add user message to history
            st.session_state['chat_history'].append({"role": "user", "message": user_message})
            
            # Generate response using relevant context
            relevant_docs = rag_app.search_documents(user_message)
            with st.spinner("Generating response..."):
                response = rag_app.generate_response(user_message, relevant_docs)
                st.session_state['chat_history'].append({"role": "assistant", "message": response})
        
        # Display chat history
        for chat in st.session_state['chat_history']:
            if chat['role'] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**Assistant:** {chat['message']}")
        
        # Settings section in sidebar
        st.sidebar.header("⚙️ Settings")
        with st.sidebar.expander("Advanced Settings"):
            st.number_input("Max Search Results", min_value=1, max_value=10, value=3)
            st.slider("Response Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Built with Streamlit, Llama, Alexa's fine ass and Weaviate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()