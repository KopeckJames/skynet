# services/rag_service.py
from typing import Dict, List, Any, Optional
import openai
import tempfile
import os
from pathlib import Path
import logging
from datetime import datetime

from ..database.weaviate_client import WeaviateClient
from ..processors.document_processor import DocumentProcessor
from ..processors.file_processor import FileProcessor
from ..processors.media_processor import MediaProcessor
from ..processors.web_processor import WebProcessor
from ..config import Config

class RAGService:
    """Main service for handling document processing and retrieval"""

    def __init__(self, openai_api_key: str, weaviate_url: str, weaviate_api_key: str):
        """Initialize the RAG service with necessary components"""
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize database client
        self.weaviate_client = WeaviateClient(weaviate_url, weaviate_api_key, openai_api_key)
        
        # Initialize processors
        self.document_processor = DocumentProcessor(openai_api_key)
        self.file_processor = FileProcessor()
        self.media_processor = MediaProcessor(openai_api_key)
        self.web_processor = WebProcessor()
        
        logging.info("Initialized RAG service")

    def process_file(self, file: Any, filename: str) -> Dict[str, Any]:
        """Process uploaded file based on its type"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file.getvalue())
                file_path = temp_file.name

            file_type = self.file_processor.get_file_type(filename)
            
            # Process based on file type
            if file_type.startswith('image/'):
                result = self.media_processor.process({
                    "type": file_type,
                    "path": file_path
                })
            elif file_type.startswith('audio/'):
                result = self.media_processor.process({
                    "type": file_type,
                    "path": file_path
                })
            elif file_type.startswith('video/'):
                result = self.media_processor.process({
                    "type": file_type,
                    "path": file_path
                })
            else:
                result = self.file_processor.process({
                    "type": file_type,
                    "path": file_path
                })

            os.unlink(file_path)  # Clean up temporary file
            return result

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            raise

    def add_document(self, title: str, content: Dict[str, Any]) -> bool:
        """Add a processed document to the database"""
        try:
            # Process the content with document processor
            processed_content = self.document_processor.process(content["content"])
            
            # Prepare document properties
            document = {
                "title": title,
                "content": processed_content["content"],
                "summary": processed_content["summary"],
                "file_type": content["file_type"],
                "metadata": content["metadata"],
                "tags": processed_content["tags"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Add to database
            result = self.weaviate_client.add_document(document)
            return bool(result)
            
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            raise

    def process_webpage(self, url: str) -> Dict[str, Any]:
        """Process webpage content"""
        try:
            result = self.web_processor.process({"url": url})
            
            # Process the extracted content
            processed_content = self.document_processor.process(result["content"])
            result.update(processed_content)
            
            return result
        except Exception as e:
            logging.error(f"Error processing webpage: {e}")
            raise

    def search_documents(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        try:
            return self.weaviate_client.search_documents(query, limit)
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise

    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """Generate a response using GPT-3.5-turbo with retrieved context"""
        try:
            # Prepare context from documents
            context_parts = []
            total_length = 0
            max_context_length = Config.MAX_CHUNK_SIZE
            
            for doc in context_docs:
                doc_text = f"Title: {doc['title']}\nContent: {doc['content']}\nType: {doc['file_type']}"
                if total_length + len(doc_text) > max_context_length:
                    break
                context_parts.append(doc_text)
                total_length += len(doc_text)
            
            context = "\n\n".join(context_parts)
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on the provided context. "
                                 "Always cite the specific documents you're drawing information from."
                    },
                    {
                        "role": "user", 
                        "content": f"""Given the following context and question, provide a detailed answer. 
                        If the context doesn't contain relevant information, say so.

                        Context:
                        {context}

                        Question: {query}

                        Answer:"""
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise

    def chat_with_context(self, messages: List[Dict], query: str) -> str:
        """Enhanced chat function that maintains context and searches relevant documents"""
        try:
            # Search for relevant documents
            results = self.search_documents(query)
            context = "\n\n".join([
                f"Document: {doc['title']}\nContent: {doc['content']}" 
                for doc in results
            ])
            
            # Prepare conversation history
            conversation = [
                {
                    "role": "system", 
                    "content": f"""You are a helpful assistant with access to the following document context:
                    
                    {context}
                    
                    Use this context to provide informed answers while maintaining a natural conversation.
                    Always cite specific documents when using information from them."""
                }
            ]
            
            # Add recent message history (limited to last 5 messages)
            for message in messages[-5:]:
                conversation.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            
            # Add current query
            conversation.append({"role": "user", "content": query})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=conversation,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error in chat response: {e}")
            raise

    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            stats = {
                "total_documents": self.weaviate_client.get_document_count(),
                "file_types": self.weaviate_client.get_document_types(),
                "last_added": None,
                "total_unique_tags": 0
            }
            
            # Get most recent document
            recent_docs = self.weaviate_client.search_documents("", limit=1)
            if recent_docs:
                stats["last_added"] = recent_docs[0].get("created_at")
            
            # Get unique tags
            tags_result = self.weaviate_client.search_by_filters({}, limit=1000)
            unique_tags = set()
            for doc in tags_result:
                if doc.get("tags"):
                    unique_tags.update(doc["tags"])
            stats["total_unique_tags"] = len(unique_tags)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            raise

    def semantic_clustering(self, num_clusters: int = 5) -> Dict[int, List[Dict]]:
        """Perform semantic clustering on documents"""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Get all documents
            documents = self.weaviate_client.search_documents("", limit=1000)
            
            if not documents:
                return {}
            
            # Generate embeddings
            embeddings = []
            for doc in documents:
                response = self.openai_client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=doc["content"][:1000]  # Limit content length
                )
                embeddings.append(response.data[0].embedding)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(documents)), random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Group documents by cluster
            clustered_docs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_docs:
                    clustered_docs[cluster_id] = []
                clustered_docs[cluster_id].append(documents[i])
            
            return clustered_docs
            
        except Exception as e:
            logging.error(f"Error performing clustering: {e}")
            raise

    def get_document_timeline(self) -> List[Dict[str, Any]]:
        """Get document addition timeline"""
        try:
            documents = self.weaviate_client.search_documents("", limit=1000)
            timeline = []
            
            for doc in documents:
                if doc.get("created_at"):
                    timeline.append({
                        "date": doc["created_at"],
                        "title": doc["title"],
                        "type": doc["file_type"]
                    })
            
            # Sort by date
            timeline.sort(key=lambda x: x["date"])
            return timeline
            
        except Exception as e:
            logging.error(f"Error getting document timeline: {e}")
            raise

    def get_topic_distribution(self) -> Dict[str, int]:
        """Analyze topic distribution across documents"""
        try:
            documents = self.weaviate_client.search_documents("", limit=1000)
            topics = {}
            
            for doc in documents:
                # Process document to extract topics
                processed = self.document_processor.extract_main_topics(doc["content"])
                for topic in processed:
                    topic_name = topic["topic"]
                    topics[topic_name] = topics.get(topic_name, 0) + 1
            
            return topics
            
        except Exception as e:
            logging.error(f"Error getting topic distribution: {e}")
            raise

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of all documents"""
        try:
            documents = self.weaviate_client.search_documents("", limit=10000)
            
            if not documents:
                return False
            
            import json
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_documents": len(documents),
                "documents": documents
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            raise

    def restore_database(self, backup_path: str) -> bool:
        """Restore documents from a backup"""
        try:
            import json
            
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Clear existing documents
            self.weaviate_client.clear_all_documents()
            
            # Restore documents
            documents = backup_data.get("documents", [])
            if documents:
                self.weaviate_client.batch_add_documents(documents)
            
            return True
            
        except Exception as e:
            logging.error(f"Error restoring backup: {e}")
            raise
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            return self.weaviate_client.get_document_statistics()
        except Exception as e:
            logging.error(f"Error getting statistics: {str(e)}")
            return {
                "total_documents": 0,
                "file_types": {},
                "last_added": None,
                "total_unique_tags": 0
            }

    def get_document_timeline(self) -> List[Dict[str, Any]]:
        """Get document addition timeline"""
        try:
            documents = self.weaviate_client.search_documents("", limit=1000)
            if not documents:
                return []
                
            timeline = []
            for doc in documents:
                if doc.get("created_at"):
                    timeline.append({
                        "date": doc["created_at"],
                        "title": doc.get("title", "Untitled"),
                        "type": doc.get("file_type", "unknown")
                    })
            
            # Sort by date
            timeline.sort(key=lambda x: x["date"])
            return timeline
            
        except Exception as e:
            logging.error(f"Error getting document timeline: {str(e)}")
            return []

    def get_topic_distribution(self) -> Dict[str, int]:
        """Analyze topic distribution across documents"""
        try:
            documents = self.weaviate_client.search_documents("", limit=1000)
            if not documents:
                return {}
                
            topics = {}
            for doc in documents:
                # Process document to extract topics
                if doc.get("content"):
                    processed = self.document_processor.extract_main_topics(doc["content"])
                    for topic in processed:
                        topic_name = topic["topic"]
                        topics[topic_name] = topics.get(topic_name, 0) + 1
            
            return topics
            
        except Exception as e:
            logging.error(f"Error getting topic distribution: {str(e)}")
            return {}   