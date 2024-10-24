import streamlit as st
import weaviate
import openai
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

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def process_text(self, content: str) -> str:
        """Process plain text content"""
        return content.strip()
    
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
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
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
                                    "url": f"data:image/jpeg;base64,{image_data}"
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
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return response.text
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
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
        self.openai_client = openai.Client(api_key=openai_api_key)
        
        # Initialize document processor
        self.processor = DocumentProcessor(self.openai_client)
        
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
        
        # Add tabs for different operations
        tab1, tab2 = st.sidebar.tabs(["Upload", "Stats"])
        
        with tab1:
            # File upload section
            uploaded_file = st.file_uploader(
                "Choose a file (Text, PDF, Image, Audio, or Video)", 
                type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi"]
            )
            
            if uploaded_file:
                st.sidebar.write(f"Selected file: {uploaded_file.name}")
                file_stats = f"File size: {uploaded_file.size / 1024:.2f} KB"
                st.sidebar.write(file_stats)
            
            doc_title = st.text_input("Document Title", key="doc_title")
            
            if st.button("Process and Add Document"):
                if uploaded_file is not None and doc_title:
                    progress_placeholder = st.empty()
                    with st.spinner("Processing document..."):
                        try:
                            progress_placeholder.info("Step 1/2: Processing document...")
                            file_data = rag_app.process_file(uploaded_file, uploaded_file.name)
                            
                            progress_placeholder.info("Step 2/2: Adding to database...")
                            if rag_app.add_document(doc_title, file_data):
                                progress_placeholder.success("✅ Document successfully processed and added!")
                            else:
                                progress_placeholder.error("❌ Failed to add document to database.")
                        except Exception as e:
                            progress_placeholder.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please provide both a file and a title.")
        
        with tab2:
            # Display statistics
            try:
                result = rag_app.weaviate_client.query.aggregate(rag_app.class_name).with_meta_count().do()
                doc_count = result["data"]["Aggregate"][rag_app.class_name][0]["meta"]["count"]
                st.metric(label="📊 Total Documents", value=doc_count)
            except Exception as e:
                st.warning("Could not fetch document statistics")
        
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
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit, OpenAI, and Weaviate</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API keys and Weaviate URL.")

if __name__ == "__main__":
    main()