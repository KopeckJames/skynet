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
import openai
import json
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

# Optional: Set Tesseract path for Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        """Summarize the given text using GPT-3.5-turbo"""
        try:
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

    def analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of text using GPT API"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of the following text and return a JSON with positive, negative, and neutral scores that sum to 1.0"},
                    {"role": "user", "content": content}
                ],
                max_tokens=100
            )
            sentiment = json.loads(response.choices[0].message.content)
            return sentiment
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def generate_tags(self, content: str) -> List[str]:
        """Generate relevant tags for the content"""
        try:
            # First, create a more structured prompt
            system_prompt = """You are a tag generation system. Generate 5-7 relevant tags for the content.
            Return your response in this exact JSON format: {"tags": ["tag1", "tag2", "tag3"]}
            Do not include any other text or explanation in your response."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content to tag: {content[:1000]}..."}  # Limit content length
                ],
                max_tokens=100,
                temperature=0.3  # Lower temperature for more consistent formatting
            )
            
            # Get the response text and clean it
            response_text = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            print(f"Raw tag response: {response_text}")
            
            try:
                # Parse the JSON response
                json_response = json.loads(response_text)
                tags = json_response.get('tags', [])
                
                # Ensure we have valid tags
                if not tags or not isinstance(tags, list):
                    raise ValueError("Invalid tags format")
                    
                # Clean and validate tags
                tags = [str(tag).strip() for tag in tags if tag and isinstance(tag, (str, int, float))]
                
                return tags
            except json.JSONDecodeError:
                # Fallback: try to extract tags from malformed response
                import re
                tags_match = re.findall(r'["\'](.*?)["\']', response_text)
                if tags_match:
                    return [tag.strip() for tag in tags_match if tag.strip()]
                return []
                
        except Exception as e:
            print(f"Tag generation error: {str(e)}")  # Detailed error logging
            st.error(f"Error generating tags: {str(e)}")
            return []

    def extract_main_topics(self, content: str) -> List[Dict[str, str]]:
        """Extract main topics with brief descriptions"""
        try:
            # Create a structured prompt
            system_prompt = """You are a topic extraction system. Extract 3-5 main topics from the content with brief descriptions.
            Return your response in this exact JSON format:
            {
                "topics": [
                    {"topic": "First Topic", "description": "Description of first topic"},
                    {"topic": "Second Topic", "description": "Description of second topic"}
                ]
            }
            Do not include any other text or explanation in your response."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content to analyze: {content[:1500]}..."}  # Limit content length
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Get the response text and clean it
            response_text = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            print(f"Raw topics response: {response_text}")
            
            try:
                # Parse the JSON response
                json_response = json.loads(response_text)
                topics = json_response.get('topics', [])
                
                # Validate topics structure
                if not topics or not isinstance(topics, list):
                    raise ValueError("Invalid topics format")
                
                # Clean and validate each topic
                validated_topics = []
                for topic in topics:
                    if isinstance(topic, dict) and 'topic' in topic and 'description' in topic:
                        validated_topics.append({
                            'topic': str(topic['topic']).strip(),
                            'description': str(topic['description']).strip()
                        })
                
                return validated_topics
            except json.JSONDecodeError:
                # Fallback: try to extract topics from malformed response
                import re
                topics = []
                # Look for patterns like "Topic: Description" or "1. Topic - Description"
                topic_matches = re.findall(r'(?:^|\n)(?:\d+\.\s*)?([^:\n-]+)[:\-]([^\n]+)', response_text)
                for topic, description in topic_matches:
                    topics.append({
                        'topic': topic.strip(),
                        'description': description.strip()
                    })
                return topics
                
        except Exception as e:
            print(f"Topic extraction error: {str(e)}")  # Detailed error logging
            st.error(f"Error extracting topics: {str(e)}")
            return []

    # Add this helper function to the DocumentProcessor class
    def _format_text_for_analysis(self, text: str, max_length: int = 1500) -> str:
        """Helper function to format text for analysis"""
        if not text:
            return ""
        
        # Clean the text
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
    
    
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

    def process_image(self, file_path: str) -> Dict[str, str]:
        """Process image files using OCR and GPT-4-Vision"""
        try:
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
            
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

    def process_video(self, file_path: str) -> Dict[str, str]:
        """Process video files - extract audio for transcription and frames for analysis"""
        try:
            video = VideoFileClip(file_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                video.audio.write_audiofile(temp_audio.name)
                transcription = self.process_audio(temp_audio.name)
                os.unlink(temp_audio.name)
            
            cap = cv2.VideoCapture(file_path)
            frames = []
            frame_descriptions = []
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // 5)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0 and len(frames) < 5:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
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

    def process_powerpoint(self, file_path: str) -> str:
        """Process PowerPoint files"""
        try:
            prs = Presentation(file_path)
            text_content = []
            
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_content.append(shape.text)
            
            return "\n".join(text_content)
        except Exception as e:
            st.error(f"Error processing PowerPoint: {str(e)}")
            return ""

    def process_excel(self, file_path: str) -> str:
        """Process Excel files"""
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            st.error(f"Error processing Excel: {str(e)}")
            return ""

    def process_csv(self, file_path: str) -> str:
        """Process CSV files"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return ""

    def process_webpage(self, url: str) -> str:
        """Extract text from a webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if response.encoding is None:
                response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return text
            
        except requests.RequestException as e:
            st.error(f"Error fetching webpage: {str(e)}")
            return ""
        except Exception as e:
            st.error(f"Error processing webpage: {str(e)}")
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
        elif ext in ['.pptx']:
            return 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        elif ext in ['.xlsx']:
            return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif ext in ['.csv']:
            return 'text/csv'
        else:
            return 'application/octet-stream'

class RAGApplication:
    def __init__(self, openai_api_key: str, weaviate_url: str, weaviate_api_key: str):
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.processor = DocumentProcessor(self.openai_api_key)
        
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
        """Create Weaviate schema if it doesn't exist"""
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
            elif file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
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

    def add_document(self, title: str, file_data: Dict[str, str]) -> bool:
        """Add a document to the Weaviate database"""
        try:
            content_chunks = self._chunk_text(file_data["content"])
            st.write(f"Splitting document into {len(content_chunks)} chunks")
            
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
        """Generate a response using GPT-3.5-turbo with retrieved context"""
        try:
            context_parts = []
            total_length = 0
            max_context_length = 4000
            
            for doc in context_docs:
                doc_text = f"Title: {doc['title']}\nContent: {doc['content']}\nType: {doc['file_type']}"
                if total_length + len(doc_text) > max_context_length:
                    break
                context_parts.append(doc_text)
                total_length += len(doc_text)
            
            context = "\n\n".join(context_parts)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
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

    def chat_with_context(self, messages: List[Dict], query: str) -> str:
        """Enhanced chat function that maintains context and searches relevant documents"""
        try:
            results = self.search_documents(query)
            context = "\n\n".join([f"Document: {doc['title']}\nContent: {doc['content']}" for doc in results])
            
            conversation = [
                {"role": "system", "content": f"""You are a helpful assistant with access to the following document context:
                
                {context}
                
                Use this context to provide informed answers while maintaining a natural conversation."""}
            ]
            
            for message in messages[-5:]:
                conversation.append({"role": message["role"], "content": message["content"]})
            
            conversation.append({"role": "user", "content": query})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in chat response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    def get_document_statistics(self) -> Dict[str, int]:
        """Get statistics about stored documents"""
        try:
            result = (
                self.weaviate_client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .with_fields("file_type {type: value}")
                .do()
            )
            
            stats = {
                "total_documents": result["data"]["Aggregate"][self.class_name][0]["meta"]["count"],
                "file_types": {}
            }
            
            type_result = (
                self.weaviate_client.query
                .get(self.class_name, ["file_type"])
                .do()
            )
            
            if type_result["data"]["Get"][self.class_name]:
                for doc in type_result["data"]["Get"][self.class_name]:
                    file_type = doc["file_type"]
                    stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                    
            return stats
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")
            return {"total_documents": 0, "file_types": {}}

    def semantic_clustering(self, num_clusters: int = 5) -> Dict[int, List[Dict]]:
        """Perform semantic clustering on documents"""
        try:
            result = (
                self.weaviate_client.query
                .get(self.class_name, ["title", "content"])
                .do()
            )
            
            documents = result["data"]["Get"][self.class_name]
            
            embeddings = []
            for doc in documents:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=doc["content"][:1000]
                )
                embeddings.append(response.data[0].embedding)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            clustered_docs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_docs:
                    clustered_docs[cluster_id] = []
                clustered_docs[cluster_id].append(documents[i])
            
            return clustered_docs
        except Exception as e:
            st.error(f"Error performing clustering: {str(e)}")
            return {}

    def similar_documents(self, document_id: str, limit: int = 5) -> List[Dict]:
        """Find similar documents to a given document"""
        try:
            result = (
                self.weaviate_client.query
                .get(self.class_name, ["title", "content"])
                .with_near_object({"id": document_id})
                .with_limit(limit)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"]:
                return result["data"]["Get"][self.class_name]
            return []
        except Exception as e:
            st.error(f"Error finding similar documents: {str(e)}")
            return []
def show_document_management(rag_app):
    st.header("📁 Document Management")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file (Text, PDF, Image, Audio, Video, PowerPoint, Excel, CSV)", 
        type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi", "pptx", "xlsx", "csv"]
    )
    
    # URL input for webpage processing
    webpage_url = st.text_input("Enter a webpage URL to process")
    
    doc_title = st.text_input("Document Title", key="doc_title")
    
    if st.button("Process and Add Document"):
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
                        # Process document analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Document Analysis")
                            
                            # Summary
                            with st.expander("📝 Summary", expanded=True):
                                summary = rag_app.processor.summarize_text(file_data['content'])
                                st.write(summary)
                            
                            # Sentiment Analysis
                            with st.expander("😊 Sentiment Analysis"):
                                sentiment = rag_app.processor.analyze_sentiment(file_data['content'])
                                # Create a bar chart for sentiment
                                sentiment_df = pd.DataFrame({
                                    'Sentiment': list(sentiment.keys()),
                                    'Score': list(sentiment.values())
                                })
                                st.bar_chart(sentiment_df.set_index('Sentiment'))
                        
                        with col2:
                            # Tags
                            with st.expander("🏷️ Generated Tags", expanded=True):
                                tags = rag_app.processor.generate_tags(file_data['content'])
                                for tag in tags:
                                    st.button(f"#{tag}", key=tag)
                            
                            # Main Topics
                            with st.expander("📌 Main Topics"):
                                topics = rag_app.processor.extract_main_topics(file_data['content'])
                                for topic in topics:
                                    st.markdown(f"**{topic['topic']}**")
                                    st.write(topic['description'])
                                    st.divider()
                        
                        # Add document to database
                        success = rag_app.add_document(doc_title, file_data)
                        if success:
                            st.success("✅ Document added successfully!")
                        else:
                            st.error("Failed to add document to the database")
                            
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        else:
            st.warning("Please provide both a document and a title")

def show_search_and_chat(rag_app):
    st.header("🔍 Search and Chat")
    
    tab1, tab2 = st.tabs(["🔍 Semantic Search", "💬 Chat Interface"])
    
    with tab1:
        query = st.text_input("Enter your search query:")
        num_results = st.slider("Number of results", 1, 10, 3)
        
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    results = rag_app.search_documents(query, limit=num_results)
                    if results:
                        # Display AI-generated response
                        response = rag_app.generate_response(query, results)
                        st.markdown("### 🤖 AI Response")
                        st.write(response)
                        
                        # Display source documents
                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(results, 1):
                            with st.expander(f"Document {i}: {doc['title']}"):
                                st.write(doc['content'])
                                
                                # Add similar documents
                                st.markdown("#### Similar Documents")
                                similar_docs = rag_app.similar_documents(doc.get('id', ''), limit=3)
                                for sim_doc in similar_docs:
                                    st.markdown(f"- {sim_doc['title']}")
                    else:
                        st.info("No relevant documents found")
            else:
                st.warning("Please enter a search query")
    
    with tab2:
        st.markdown("### 💬 Chat with Your Documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response with context
            with st.chat_message("assistant"):
                response = rag_app.chat_with_context(st.session_state.messages, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def show_analytics_dashboard(rag_app):
    st.header("📊 Analytics Dashboard")
    
    # Get document statistics
    stats = rag_app.get_document_statistics()
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats["total_documents"])
    
    with col2:
        st.metric("Document Types", len(stats["file_types"]))
    
    with col3:
        if stats["file_types"]:
            most_common_type = max(stats["file_types"].items(), key=lambda x: x[1])[0]
            st.metric("Most Common Type", most_common_type)
    
    # Document type distribution
    st.subheader("Document Type Distribution")
    if stats["file_types"]:
        file_type_df = pd.DataFrame(
            list(stats["file_types"].items()),
            columns=['File Type', 'Count']
        )
        st.bar_chart(file_type_df.set_index('File Type'))
    
    # Semantic Clustering
    st.subheader("Document Clusters")
    num_clusters = st.slider("Number of clusters", 2, 10, 5)
    
    if st.button("Generate Clusters"):
        with st.spinner("Performing semantic clustering..."):
            clusters = rag_app.semantic_clustering(num_clusters)
            
            for cluster_id, docs in clusters.items():
                with st.expander(f"Cluster {cluster_id + 1} ({len(docs)} documents)"):
                    for doc in docs:
                        st.markdown(f"- **{doc['title']}**")
                        with st.expander("Preview"):
                            st.write(doc['content'][:200] + "...")

def main():
    st.title("📚 Enhanced Multi-Modal RAG Application")
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = "https://3hbemlvctn2km7f8fyfjcg.c0.us-west3.gcp.weaviate.cloud"  # Replace with your Weaviate URL
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not all([openai_api_key, weaviate_url, weaviate_api_key]):
        st.error("Missing required environment variables. Please check your .env file.")
        st.info("Required variables: OPENAI_API_KEY, WEAVIATE_API_KEY")
        return
    
    try:
        # Initialize RAG application
        rag_app = RAGApplication(openai_api_key, weaviate_url, weaviate_api_key)
        st.success("Successfully connected to Weaviate and OpenAI!")
        
        # Create main navigation
        nav_option = st.sidebar.selectbox(
            "Navigation",
            ["Document Management", "Search & Chat", "Analytics Dashboard"]
        )
        
        if nav_option == "Document Management":
            show_document_management(rag_app)
        elif nav_option == "Search & Chat":
            show_search_and_chat(rag_app)
        else:
            show_analytics_dashboard(rag_app)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API keys and Weaviate URL.")

if __name__ == "__main__":
    main()   