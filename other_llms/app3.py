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
import gc

# Load environment variables
load_dotenv()

def verify_cuda_setup():
    """Verify CUDA setup and GPU availability"""
    try:
        if not torch.cuda.is_available():
            st.error("CUDA is not available")
            return False
            
        # Get GPU information
        gpu_info = {
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda,
            "GPU Device": torch.cuda.get_device_name(0),
            "GPU Memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "Current Device": torch.cuda.current_device(),
            "Device Count": torch.cuda.device_count()
        }
        
        st.write("GPU Information:")
        for key, value in gpu_info.items():
            st.write(f"{key}: {value}")
        
        # Test CUDA computation
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            test_result = test_tensor * 2
            st.write("✅ CUDA Computation Test: Passed")
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            st.error(f"❌ CUDA Computation Test Failed: {e}")
            return False
        
        return True
    except Exception as e:
        st.error(f"CUDA verification failed: {str(e)}")
        return False

def verify_environment():
    """Verify that all components are properly installed"""
    try:
        import transformers
        import torch
        
        st.write("Package versions:")
        st.write(f"transformers: {transformers.__version__}")
        st.write(f"torch: {torch.__version__}")
        st.write(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            st.write(f"GPU Device: {torch.cuda.get_device_name(0)}")
            
        return True
    except Exception as e:
        st.error(f"Environment verification failed: {str(e)}")
        return False

class MistralProcessor:
    def __init__(self):
        """Initialize Mistral model using Windows-compatible configuration"""
        try:
            # Use a specific GPTQ model known to work on Windows
            self.model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load the model with Windows-compatible settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                revision="main",
                use_safetensors=True  # Add this for better Windows compatibility
            )
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.nlp = spacy.load("en_core_web_sm")
            self.kw_model = KeyBERT()
            
            # Force model to GPU if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.empty_cache()  # Clear GPU memory
                self.model = self.model.to(self.device)
            else:
                self.device = torch.device("cpu")
                
        except Exception as e:
            st.error(f"Detailed error initializing Mistral: {str(e)}")
            raise

    def generate_response(self, prompt: str, max_length=1000, temperature=0.7) -> str:
        """Generate a response with Windows-optimized settings"""
        try:
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Format prompt
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            if torch.cuda.is_available():
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
    
    def extract_keywords(self, content: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text using KeyBERT"""
        try:
            keywords = self.kw_model.extract_keywords(content, 
                                                    keyphrase_ngram_range=(1, 2),
                                                    stop_words='english',
                                                    top_n=num_keywords)
            return [kw[0] for kw in keywords]
        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy"""
        try:
            doc = self.nlp(content)
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            return entities
        except Exception as e:
            st.error(f"Error extracting entities: {str(e)}")
            return {}

    def summarize_text(self, content: str) -> str:
        """Summarize the given text using Mistral"""
        try:
            # Split content into smaller chunks if it is too long
            max_chunk_size = 3000
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            summaries = []
            
            for chunk in chunks:
                prompt = f"Please summarize this text concisely and professionally:\n\n{chunk}"
                summary = self.mistral.generate_response(prompt, max_length=200)
                summaries.append(summary)
            
            return " ".join(summaries)
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")
            return ""

    def process_image(self, file_path: str) -> Dict[str, str]:
        """Process image files using OCR and generate description"""
        try:
            # Perform OCR
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
            
            # Generate image description
            prompt = f"Based on this OCR text, describe the image content professionally:\n\n{ocr_text}"
            description = self.mistral.generate_response(prompt)
            
            return {
                "ocr_text": ocr_text,
                "description": description
            }
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return {"ocr_text": "", "description": ""}

    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = []
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text.append(page.get_text())
            
            # Use OCR as fallback if no text was extracted
            if not ''.join(text).strip():
                images = pdf2image.convert_from_path(file_path)
                for image in images:
                    text.append(pytesseract.image_to_string(image))
            
            return '\n'.join(text)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    def process_video(self, file_path: str) -> Dict[str, str]:
        """Process video files - extract frames and analyze"""
        try:
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
            
            # Combine frame descriptions into a cohesive summary
            frames_text = "\n".join(frame_descriptions)
            summary_prompt = f"Summarize these video frame descriptions into a coherent video description:\n\n{frames_text}"
            video_summary = self.mistral.generate_response(summary_prompt)
            
            return {
                "frame_descriptions": frames_text,
                "video_summary": video_summary
            }
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return {"frame_descriptions": "", "video_summary": ""}

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
        """Initialize RAG application"""
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
            elif file_type.startswith('video/'):
                result = self.processor.process_video(file_path)
                content = f"Video Analysis:\n{result['video_summary']}\n\nDetailed Frame Descriptions:\n{result['frame_descriptions']}"
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
            # Split content into manageable chunks if it's too long
            max_chunk_size = 2000
            content_chunks = [file_data["content"][i:i+max_chunk_size] 
                            for i in range(0, len(file_data["content"]), max_chunk_size)]
            
            for i, chunk in enumerate(content_chunks):
                chunk_title = f"{title} (Part {i+1}/{len(content_chunks)})" if len(content_chunks) > 1 else title
                
                properties = {
                    "title": chunk_title,
                    "content": chunk,
                    "file_type": file_data["file_type"],
                    "metadata": f"{file_data['metadata']}, Chunk: {i+1}/{len(content_chunks)}"
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

    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """Generate a response with context"""
        try:
            # Format context and query
            context_parts = []
            for doc in context_docs:
                doc_text = f"Title: {doc['title']}\nContent: {doc['content']}\nType: {doc['file_type']}"
                context_parts.append(doc_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"""Based on the following documents, answer this question:

Context Documents:
{context}

Question: {query}

Please provide a detailed and accurate answer based solely on the information provided in the context documents. If the context doesn't contain relevant information, please state that."""
            
            return self.mistral.generate_response(prompt)
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.title("📚 Mistral-powered RAG Application")
    
    # Verify environment and CUDA setup
    if not verify_environment():
        st.error("Environment verification failed. Please check package installations.")
        return
    
    if not verify_cuda_setup():
        st.error("CUDA setup failed. Please check your GPU configuration.")
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
        # Initialize RAG application
        rag_app = RAGApplication(weaviate_url, weaviate_api_key)
        st.success("Successfully connected to Weaviate and initialized Mistral model!")
        
        # GPU Memory Monitor
        if torch.cuda.is_available():
            gpu_monitor = st.empty()
            def update_gpu_stats():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                gpu_monitor.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            update_gpu_stats()
        
        # Sidebar for document management
        st.sidebar.header("📁 Document Management")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "png", "jpg", "jpeg", "mp4", "avi"]
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
                                
                                # Document Analysis
                                st.markdown("### Document Analysis")
                                
                                with st.expander("View Analysis", expanded=True):
                                    # Summary
                                    summary = rag_app.processor.summarize_text(file_data['content'])
                                    st.markdown("**Summary:**")
                                    st.write(summary)
                                    
                                    # Keywords
                                    keywords = rag_app.processor.extract_keywords(file_data['content'])
                                    st.markdown("**Keywords:**")
                                    st.write(", ".join(keywords))
                                    
                                    # Named Entities
                                    entities = rag_app.processor.extract_entities(file_data['content'])
                                    st.markdown("**Named Entities:**")
                                    for entity_type, entity_list in entities.items():
                                        st.write(f"{entity_type}: {', '.join(entity_list)}")
                                
                                update_gpu_stats()
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
                    
                    update_gpu_stats()
                else:
                    st.warning("No relevant documents found. Try adding some documents first or modify your query.")
        
        # Chat history
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
                update_gpu_stats()
        
        # Display chat history
        for chat in st.session_state['chat_history']:
            if chat['role'] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**Assistant:** {chat['message']}")
        
        # Settings
        st.sidebar.header("⚙️ Settings")
        with st.sidebar.expander("Advanced Settings"):
            st.number_input("Max Search Results", min_value=1, max_value=10, value=3)
            st.slider("Response Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Built with Streamlit, Mistral, and Weaviate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()       