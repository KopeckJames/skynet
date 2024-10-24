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
from pptx import Presentation  # Use python-pptx for parsing PowerPoint files
import xlrd  # For Excel support
import csv
import openai
import json

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.nlp = spacy.load("en_core_web_sm")
        self.kw_model = KeyBERT()
        
    def process_text(self, content: str) -> str:
        """Process plain text content"""
        return content.strip()
    
    def summarize_text(self, content: str) -> str:
        """Summarize the given text using OpenAI"""
        try:
            # Split content into smaller chunks if it is too long
            max_chunk_size = 3000
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            summaries = []
            
            for chunk in chunks:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": f"Summarize the following text:\n{chunk}"}
                    ],
                    max_tokens=150
                )
                summaries.append(response.choices[0].message.content.strip())
            
            return " ".join(summaries)
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")
            return ""
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy"""
        try:
            doc = self.nlp(content)
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            return entities
        except Exception as e:
            st.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def extract_keywords(self, content: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text using KeyBERT"""
        try:
            keywords = self.kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
            return [kw[0] for kw in keywords]
        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = []
        try:
            # Using PyMuPDF for text extraction
            doc = fitz.open(file_path)
            for page in doc:
                text.append(page.get_text())
            
            # If text extraction yields poor results, use OCR as fallback
            if not ''.join(text).strip():
                images = pdf2image.convert_from_path(file_path)
                for image in images:
                    text.append(pytesseract.image_to_string(image))
            
            return '\n'.join(text)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    def process_image(self, file_path: str) -> Dict[str, str]:
        """Process image files using OCR and image analysis"""
        try:
            # Perform OCR
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
            
            # Get image description using OpenAI Vision
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return {
                "ocr_text": ocr_text,
                "description": response.choices[0].message.content
            }
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return {"ocr_text": "", "description": ""}

    def process_audio(self, file_path: str) -> str:
        """Process audio files using OpenAI Whisper API"""
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return ""
    def process_webpage(self, url: str) -> str:
        """Extract text from a webpage"""
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Try to get the webpage's encoding
            if response.encoding is None:
                response.encoding = response.apparent_encoding
            
            # Use BeautifulSoup to parse the HTML and extract text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text(separator='\n')
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            # Remove empty lines and join
            text = '\n'.join(line for line in lines if line)
            
            return text
            
        except requests.RequestException as e:
            st.error(f"Error fetching webpage: {str(e)}")
            return ""
        except Exception as e:
            st.error(f"Error processing webpage: {str(e)}")
            return ""

    def process_video(self, file_path: str) -> Dict[str, str]:
        """Process video files - extract audio for transcription and frames for analysis"""
        try:
            # Extract audio and transcribe
            video = VideoFileClip(file_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                video.audio.write_audiofile(temp_audio.name)
                transcription = self.process_audio(temp_audio.name)
                os.unlink(temp_audio.name)
            
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
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Analyze key frames
            for frame in frames:
                with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_frame:
                    cv2.imwrite(temp_frame.name, frame)
                    frame_info = self.process_image(temp_frame.name)
                    frame_descriptions.append(frame_info["description"])
            
            return {
                "transcription": transcription,
                "frame_descriptions": "\n".join(frame_descriptions)
            }
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return {"transcription": "", "frame_descriptions": ""}

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
    def __init__(self, openai_api_key: str, weaviate_url: str, weaviate_api_key: str):
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize document processor
        self.processor = DocumentProcessor(self.openai_api_key)
        
        # Initialize Weaviate client with authentication
        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
        
        self.weaviate_client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": openai_api_key
            }
        )
        
        self.class_name = "Document"
        self._create_schema()

    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """Generate a response using OpenAI with retrieved context"""
        try:
            # Combine context but limit total length
            context_parts = []
            total_length = 0
            max_context_length = 4000  # Conservative limit for GPT-4
            
            for doc in context_docs:
                doc_text = f"Title: {doc['title']}\nContent: {doc['content']}\nType: {doc['file_type']}"
                if total_length + len(doc_text) > max_context_length:
                    break
                context_parts.append(doc_text)
                total_length += len(doc_text)
            
            context = "\n\n".join(context_parts)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"""Given the following context and question, provide a detailed answer. 
                    If the context doesn't contain relevant information, say so.

                    Context:
                    {context}

                    Question: {query}

                    Answer:"""}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"





    def _create_schema(self):
        schema = {
            "classes": [{
                "class": self.class_name,
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "vectorizer": "text2vec-openai"
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "vectorizer": "text2vec-openai"
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
            existing_schema = self.weaviate_client.schema.get()
            existing_classes = [c['class'] for c in existing_schema['classes']] if existing_schema.get('classes') else []
            
            if self.class_name not in existing_classes:
                self.weaviate_client.schema.create_class(schema['classes'][0])
        except Exception as e:
            print(f"Error with schema: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into smaller chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

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
                content = f"Transcription: {result['transcription']}\nVideo Description: {result['frame_descriptions']}"
            elif file_type == 'application/pdf':
                content = self.processor.process_pdf(file_path)
            elif file_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                content = self.processor.process_powerpoint(file_path)
            elif file_type == 'application/vnd.ms-excel' or file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                content = self.processor.process_excel(file_path)
            elif file_type == 'text/csv':
                content = self.processor.process_csv(file_path)
            else:
                content = self.processor.process_text(file.getvalue().decode())
            
            return {
                "content": content,
                "file_type": file_type,
                "metadata": f"Filename: {filename}, Type: {file_type}"
            }
        finally:
            os.unlink(file_path)


    def search_documents_multimodal(self, query: str, limit: int = 3) -> List[dict]:
        """Search for relevant documents using vector similarity, including different modalities"""
        try:
            result = (
                self.weaviate_client.query
                .get(self.class_name, ["title", "content", "file_type", "metadata"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                return result["data"]["Get"][self.class_name]
            return []
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []

    def chatbot_response(self, user_message: str) -> str:
        """Generate a conversational response using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating chatbot response: {str(e)}")
            return ""

            
    def add_document(self, title: str, file_data: Dict[str, str]) -> bool:
        """Add a document to the Weaviate database"""
        try:
            # Split content into chunks
            content_chunks = self._chunk_text(file_data["content"])
            st.write(f"Splitting document into {len(content_chunks)} chunks")
            
            # Add each chunk as a separate document
            for i, chunk in enumerate(content_chunks):
                chunk_title = f"{title} (Part {i+1}/{len(content_chunks)})" if len(content_chunks) > 1 else title
                
                properties = {
                    "title": chunk_title,
                    "content": chunk,
                    "file_type": file_data["file_type"],
                    "metadata": f"{file_data['metadata']}, Chunk: {i+1}/{len(content_chunks)}"
                }
                
                result = self.weaviate_client.data_object.create(
                    class_name=self.class_name,
                    data_object=properties
                )
                
                if not result:
                    print(f"Failed to add chunk {i+1}")
                    return False
                
                st.write(f"Added chunk {i+1}/{len(content_chunks)}")
            
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            st.error(f"Error adding document: {e}")
            return False

    def search_documents(self, query: str, limit: int = 3) -> List[dict]:
        """Search for relevant documents using vector similarity"""
        try:
            result = (
                self.weaviate_client.query
                .get(self.class_name, ["title", "content", "file_type", "metadata"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                return result["data"]["Get"][self.class_name]
            return []
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """Generate a response using OpenAI with retrieved context"""
        try:
            # Combine context but limit total length
            context_parts = []
            total_length = 0
            max_context_length = 4000  # Conservative limit for GPT-4
            
            for doc in context_docs:
                doc_text = f"Title: {doc['title']}\nContent: {doc['content']}\nType: {doc['file_type']}"
                if total_length + len(doc_text) > max_context_length:
                    break
                context_parts.append(doc_text)
                total_length += len(doc_text)
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Given the following context and question, provide a detailed answer. 
            If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {query}

            Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    st.title("📚 Multi-Modal RAG Application")
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = "https://3hbemlvctn2km7f8fyfjcg.c0.us-west3.gcp.weaviate.cloud"
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    # Verify that all required environment variables are set
    if not all([openai_api_key, weaviate_url, weaviate_api_key]):
        st.error("Missing required environment variables. Please check your .env file.")
        st.info("Required variables: OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY")
        return
    
    try:
        # Initialize RAG application
        rag_app = RAGApplication(openai_api_key, weaviate_url, weaviate_api_key)
        st.success("Successfully connected to Weaviate and OpenAI!")
        
        # Sidebar for configuration and uploads
        st.sidebar.header("📁 Document Management")
        
        # File upload section
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file (Text, PDF, Image, Audio, Video, PowerPoint, Excel, CSV)", 
            type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi", "pptx", "xlsx", "csv"]
        )
        
        # URL input for webpage processing
        webpage_url = st.sidebar.text_input("Enter a webpage URL to process")
        
        doc_title = st.sidebar.text_input("Document Title", key="doc_title")
        
        if st.sidebar.button("Process and Add Document"):
            if (uploaded_file is not None or webpage_url) and doc_title:
                with st.spinner("Processing document..."):
                    try:
                        if uploaded_file is not None:
                            file_data = rag_app.process_file(uploaded_file, uploaded_file.name)
                        elif webpage_url:
                            content = rag_app.processor.process_webpage(webpage_url)
                            file_data = {
                                "content": content,
                                "file_type": "text/html",
                                "metadata": f"URL: {webpage_url}, Type: webpage"
                            }
                        
                        if file_data:
                            st.success("✅ Document processed successfully!")
                            st.markdown("### Summary")
                            summary = rag_app.processor.summarize_text(file_data['content'])
                            st.text_area("Summary", value=summary, height=150)
                            
                            # Named Entity Recognition
                            st.markdown("### Named Entities")
                            entities = rag_app.processor.extract_entities(file_data['content'])
                            for label, ents in entities.items():
                                st.markdown(f"**{label}:** {', '.join(ents)}")
                            
                            # Keyword Extraction
                            st.markdown("### Keywords")
                            keywords = rag_app.processor.extract_keywords(file_data['content'])
                            st.markdown(", ".join(keywords))
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please provide both a file or a webpage URL and a title.")
        
        # Main content area
        st.header("🔍 Ask Questions")
        
        # Create two columns for the query interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter your question:", placeholder="What would you like to know?")
        
        with col2:
            search_button = st.button("🔍 Search")
        
        if search_button:
            if query:
                try:
                    with st.spinner("🔍 Searching and generating response..."):
                        # Search for relevant documents
                        st.info("📚 Searching through documents...")
                        relevant_docs = rag_app.search_documents(query)
                        
                        if relevant_docs:
                            # Create tabs for results
                            results_tab, sources_tab = st.tabs(["🤖 AI Response", "📚 Sources"])
                            
                            with results_tab:
                                st.info("💭 Generating response...")
                                response = rag_app.generate_response(query, relevant_docs)
                                st.markdown("### Response:")
                                st.markdown(response)
                            
                            with sources_tab:
                                st.markdown("### Retrieved Documents:")
                                for i, doc in enumerate(relevant_docs, 1):
                                    with st.expander(f"📄 Source {i}: {doc['title']}"):
                                        st.markdown("**Content:**")
                                        st.write(doc['content'])
                                        st.markdown("**Metadata:**")

                                        st.info(doc.get('metadata', 'No metadata available'))
                        else:
                            st.warning("No relevant documents found. Try adding some documents first or modify your query.")
                except Exception as e:
                    st.error(f"An error occurred during search: {str(e)}")
            else:
                st.warning("Please enter a question.")
        
        # Chatbot interface
        st.header("💬 Chat with the Assistant")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        user_message = st.text_input("Your message:", key="chat_input")
        if st.button("Send"):
            if user_message:
                # Generate chatbot response
                st.session_state['chat_history'].append({"role": "user", "message": user_message})
                with st.spinner("Generating response..."):
                    bot_response = rag_app.generate_response(user_message, [])
                    st.session_state['chat_history'].append({"role": "assistant", "message": bot_response})
        
        # Display chat history
        for chat in st.session_state['chat_history']:
            if chat['role'] == "user":
            

                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**Assistant:** {chat['message']}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit, Gemini, and Weaviate</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API keys and Weaviate URL.")

if __name__ == "__main__":
    main()