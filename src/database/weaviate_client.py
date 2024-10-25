# src/database/weaviate_client.py

import weaviate
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import hashlib
from ..utils.date_utils import format_datetime

class WeaviateClient:
    """Client for interacting with Weaviate vector database"""

    def __init__(self, url: str, api_key: str, openai_api_key: str):
        """
        Initialize Weaviate client with authentication
        
        Args:
            url (str): Weaviate cluster URL
            api_key (str): Weaviate API key
            openai_api_key (str): OpenAI API key for embeddings
        """
        self.url = url
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        
        # Initialize Weaviate client
        auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": openai_api_key
            }
        )
        
        # Set class name for documents
        self.class_name = "Document"
        
        # Initialize schema
        self._create_schema()
        
        logging.info(f"Initialized Weaviate client at {url}")

    def _create_schema(self):
        """Create or update Weaviate schema"""
        schema_config = {
            "class": self.class_name,
            "description": "A document class for storing processed content",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The main content of the document",
                    "vectorizer": "text2vec-openai"
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title",
                    "vectorizer": "text2vec-openai"
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "description": "Document summary",
                    "vectorizer": "text2vec-openai"
                },
                {
                    "name": "file_type",
                    "dataType": ["string"],
                    "description": "Type of the document"
                },
                {
                    "name": "metadata_str",
                    "dataType": ["string"],
                    "description": "Additional metadata about the document as JSON string"
                },
                {
                    "name": "tags",
                    "dataType": ["string[]"],
                    "description": "Associated tags"
                },
                {
                    "name": "created_at",
                    "dataType": ["date"],
                    "description": "Document creation timestamp"
                }
            ]
        }
        
        try:
            # Check if schema exists
            existing_schema = self.client.schema.get()
            existing_classes = [c['class'] for c in existing_schema.get('classes', [])]
            
            if self.class_name in existing_classes:
                # Delete existing schema
                self.client.schema.delete_class(self.class_name)
            
            # Create new schema
            self.client.schema.create_class(schema_config)
            logging.info(f"Created schema for class {self.class_name}")
                
        except Exception as e:
            logging.error(f"Error with schema creation/update: {e}")
            raise

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 7000) -> List[str]:
        """
        Split text into chunks that fit within token limits
        
        Args:
            text (str): Text to chunk
            max_tokens (int): Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        # Split into sentences first
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Handle case where single sentence is too long
                    chunks.append(sentence[:max_chars])
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def add_document(self, properties: Dict[str, Any]) -> Optional[str]:
        """Add a document to the database"""
        try:
            # Ensure required fields are present
            required_fields = ['title', 'content']
            for field in required_fields:
                if field not in properties:
                    raise ValueError(f"Missing required field: {field}")

            # Convert metadata to string if it exists
            if 'metadata' in properties:
                properties['metadata_str'] = json.dumps(properties['metadata'])
                del properties['metadata']

            # Format date properly
            if 'created_at' in properties:
                if isinstance(properties['created_at'], str):
                    dt = datetime.fromisoformat(properties['created_at'].replace('Z', '+00:00'))
                    properties['created_at'] = format_datetime(dt)
                elif isinstance(properties['created_at'], datetime):
                    properties['created_at'] = format_datetime(properties['created_at'])
            else:
                properties['created_at'] = format_datetime(datetime.utcnow())

            # Ensure tags is a list
            if 'tags' in properties and not isinstance(properties['tags'], list):
                properties['tags'] = [properties['tags']]
            elif 'tags' not in properties:
                properties['tags'] = []

            # Check content length and chunk if necessary
            original_content = properties['content']
            content_chunks = self.chunk_text(original_content)
            
            if len(content_chunks) == 1:
                # Single chunk, proceed normally
                return self._create_document_object(properties)
            else:
                # Multiple chunks, create separate documents
                chunk_ids = []
                base_title = properties['title']
                
                for i, chunk in enumerate(content_chunks, 1):
                    chunk_properties = properties.copy()
                    chunk_properties['title'] = f"{base_title} (Part {i}/{len(content_chunks)})"
                    chunk_properties['content'] = chunk
                    
                    # Add chunk metadata
                    chunk_metadata = json.loads(chunk_properties.get('metadata_str', '{}'))
                    chunk_metadata.update({
                        'chunk_number': i,
                        'total_chunks': len(content_chunks),
                        'original_title': base_title,
                        'is_chunk': True,
                        'document_hash': hashlib.md5(original_content.encode()).hexdigest()
                    })
                    chunk_properties['metadata_str'] = json.dumps(chunk_metadata)
                    
                    chunk_id = self._create_document_object(chunk_properties)
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                
                logging.info(f"Added document in {len(chunk_ids)} chunks: {base_title}")
                return chunk_ids[0] if chunk_ids else None

        except Exception as e:
            logging.error(f"Error adding document: {e}")
            raise

    def _create_document_object(self, properties: Dict[str, Any]) -> Optional[str]:
        """Create a single document object in Weaviate"""
        try:
            result = self.client.data_object.create(
                class_name=self.class_name,
                data_object=properties
            )
            logging.info(f"Added document: {properties.get('title')}")
            return result
        except Exception as e:
            logging.error(f"Error creating document object: {e}")
            raise

    def search_documents(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for documents using vector similarity"""
        try:
            result = (
                self.client.query
                .get(self.class_name, [
                    "title", "content", "summary", "file_type", 
                    "metadata_str", "tags", "created_at"
                ])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                documents = result["data"]["Get"][self.class_name]
                return self._process_chunked_documents(documents)
            return []
            
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise

    def _process_chunked_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process and combine chunked documents"""
        try:
            # Convert metadata_str to dictionary and group chunks
            doc_chunks = {}
            standalone_docs = []
            
            for doc in documents:
                if 'metadata_str' in doc:
                    try:
                        metadata = json.loads(doc['metadata_str'])
                        doc['metadata'] = metadata
                        del doc['metadata_str']
                        
                        if metadata.get('is_chunk'):
                            doc_hash = metadata.get('document_hash')
                            if doc_hash:
                                if doc_hash not in doc_chunks:
                                    doc_chunks[doc_hash] = []
                                doc_chunks[doc_hash].append(doc)
                        else:
                            standalone_docs.append(doc)
                    except:
                        doc['metadata'] = {}
                        standalone_docs.append(doc)
                else:
                    standalone_docs.append(doc)
            
            # Combine chunks and add standalone documents
            processed_docs = []
            
            # Process chunked documents
            for chunks in doc_chunks.values():
                # Sort chunks by chunk number
                chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_number', 0))
                
                # Combine chunks
                combined_doc = chunks[0].copy()
                combined_doc['content'] = ' '.join(chunk['content'] for chunk in chunks)
                combined_doc['title'] = chunks[0].get('metadata', {}).get('original_title', combined_doc['title'])
                
                # Update metadata
                metadata = combined_doc.get('metadata', {})
                metadata['total_chunks'] = len(chunks)
                metadata['is_combined'] = True
                combined_doc['metadata'] = metadata
                
                processed_docs.append(combined_doc)
            
            # Add standalone documents
            processed_docs.extend(standalone_docs)
            
            return processed_docs
            
        except Exception as e:
            logging.error(f"Error processing chunked documents: {e}")
            return documents

    def search_by_filters(self, filters: Dict[str, Any], limit: int = 1000) -> List[Dict]:
        """Search documents with specific filters"""
        try:
            query = self.client.query.get(
                self.class_name,
                ["title", "content", "summary", "file_type", "metadata_str", "tags", "created_at"]
            ).with_limit(limit)

            if filters:
                where_filter = {
                    "operator": "And",
                    "operands": []
                }
                
                for field, value in filters.items():
                    if isinstance(value, list):
                        or_filter = {
                            "operator": "Or",
                            "operands": [
                                {
                                    "path": [field],
                                    "operator": "Equal",
                                    "valueString": str(v)
                                } for v in value
                            ]
                        }
                        where_filter["operands"].append(or_filter)
                    else:
                        where_filter["operands"].append({
                            "path": [field],
                            "operator": "Equal",
                            "valueString": str(value)
                        })
                
                if where_filter["operands"]:
                    query = query.with_where(where_filter)

            result = query.do()
            
            if result and "data" in result and "Get" in result["data"]:
                documents = result["data"]["Get"][self.class_name]
                return self._process_chunked_documents(documents)
            return []
            
        except Exception as e:
            logging.error(f"Error searching with filters: {e}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            return result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            
        except Exception as e:
            logging.error(f"Error getting document count: {e}")
            return 0

    def get_document_types(self) -> Dict[str, int]:
        """Get distribution of document types"""
        try:
            result = (
                self.client.query
                .get(self.class_name, ["file_type"])
                .do()
            )
            
            type_counts = {}
            if result and "data" in result and "Get" in result["data"]:
                documents = result["data"]["Get"][self.class_name]
                for doc in documents:
                    file_type = doc.get("file_type", "unknown")
                    type_counts[file_type] = type_counts.get(file_type, 0) + 1
                    
            return type_counts
            
        except Exception as e:
            logging.error(f"Error getting document types: {e}")
            return {}

    def get_document_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored documents"""
        try:
            stats = {
                "total_documents": self.get_document_count(),
                "file_types": self.get_document_types(),
                "last_added": None,
                "total_unique_tags": 0
            }
            
            # Get most recent document
            recent_docs = self.search_documents("", limit=1)
            if recent_docs:
                stats["last_added"] = recent_docs[0].get("created_at")
            
            # Get unique tags
            all_docs = self.search_documents("", limit=1000)
            unique_tags = set()
            for doc in all_docs:
                if doc.get("tags"):
                    if isinstance(doc["tags"], list):
                        unique_tags.update(doc["tags"])
                    elif isinstance(doc["tags"], str):
                        unique_tags.add(doc["tags"])
            
            stats["total_unique_tags"] = len(unique_tags)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            return {
                "total_documents": 0,
                "file_types": {},
                "last_added": None,
                "total_unique_tags": 0
            }
    # src/database/weaviate_client.py (continued)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID
        
        Args:
            document_id (str): ID of document to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if document is a chunk
            doc = self.get_document_by_id(document_id)
            if doc and doc.get('metadata', {}).get('is_chunk'):
                # Delete all related chunks
                doc_hash = doc['metadata'].get('document_hash')
                if doc_hash:
                    related_docs = self.search_by_filters({
                        'metadata_str': f'.*"document_hash":"{doc_hash}".*'
                    })
                    for related_doc in related_docs:
                        self.client.data_object.delete(
                            class_name=self.class_name,
                            uuid=related_doc.get('id')
                        )
                    return True
            
            # Delete single document
            self.client.data_object.delete(
                class_name=self.class_name,
                uuid=document_id
            )
            logging.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            return False

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID
        
        Args:
            document_id (str): Document ID
            
        Returns:
            Optional[Dict[str, Any]]: Document data if found
        """
        try:
            result = (
                self.client.query
                .get(self.class_name, [
                    "title", "content", "summary", "file_type", 
                    "metadata_str", "tags", "created_at"
                ])
                .with_id(document_id)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"]:
                documents = result["data"]["Get"][self.class_name]
                if documents:
                    processed_docs = self._process_chunked_documents(documents)
                    return processed_docs[0] if processed_docs else None
            return None
            
        except Exception as e:
            logging.error(f"Error getting document by ID: {e}")
            return None

    def update_document(self, document_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update document properties
        
        Args:
            document_id (str): Document ID
            properties (Dict[str, Any]): Updated properties
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get existing document
            existing_doc = self.get_document_by_id(document_id)
            if not existing_doc:
                return False

            # Handle metadata
            if 'metadata' in properties:
                properties['metadata_str'] = json.dumps(properties['metadata'])
                del properties['metadata']

            # Format date if present
            if 'created_at' in properties:
                if isinstance(properties['created_at'], str):
                    dt = datetime.fromisoformat(properties['created_at'].replace('Z', '+00:00'))
                    properties['created_at'] = format_datetime(dt)
                elif isinstance(properties['created_at'], datetime):
                    properties['created_at'] = format_datetime(properties['created_at'])

            # Handle tags
            if 'tags' in properties and not isinstance(properties['tags'], list):
                properties['tags'] = [properties['tags']]

            # Update document
            self.client.data_object.update(
                class_name=self.class_name,
                uuid=document_id,
                data_object=properties
            )
            
            logging.info(f"Updated document: {document_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error updating document: {e}")
            return False

    def get_similar_documents(self, document_id: str, limit: int = 5) -> List[Dict]:
        """
        Find documents similar to the given document
        
        Args:
            document_id (str): Reference document ID
            limit (int): Maximum number of similar documents to return
            
        Returns:
            List[Dict]: List of similar documents
        """
        try:
            result = (
                self.client.query
                .get(self.class_name, [
                    "title", "content", "summary", "file_type", 
                    "metadata_str", "tags", "created_at"
                ])
                .with_near_object({"id": document_id})
                .with_limit(limit)
                .do()
            )
            
            if result and "data" in result and "Get" in result["data"]:
                documents = result["data"]["Get"][self.class_name]
                return self._process_chunked_documents(documents)
            return []
            
        except Exception as e:
            logging.error(f"Error finding similar documents: {e}")
            return []

    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents in batch
        
        Args:
            documents (List[Dict[str, Any]]): List of document properties
            
        Returns:
            List[str]: List of created document IDs
        """
        try:
            document_ids = []
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for doc in documents:
                    # Process each document similar to add_document
                    if 'metadata' in doc:
                        doc['metadata_str'] = json.dumps(doc['metadata'])
                        del doc['metadata']
                    
                    if 'created_at' in doc:
                        if isinstance(doc['created_at'], str):
                            dt = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
                            doc['created_at'] = format_datetime(dt)
                        elif isinstance(doc['created_at'], datetime):
                            doc['created_at'] = format_datetime(doc['created_at'])
                    else:
                        doc['created_at'] = format_datetime(datetime.utcnow())
                    
                    if 'tags' in doc and not isinstance(doc['tags'], list):
                        doc['tags'] = [doc['tags']]
                    
                    # Handle content chunking
                    content_chunks = self.chunk_text(doc['content'])
                    base_title = doc['title']
                    
                    for i, chunk in enumerate(content_chunks, 1):
                        chunk_properties = doc.copy()
                        if len(content_chunks) > 1:
                            chunk_properties['title'] = f"{base_title} (Part {i}/{len(content_chunks)})"
                            
                            # Add chunk metadata
                            chunk_metadata = json.loads(chunk_properties.get('metadata_str', '{}'))
                            chunk_metadata.update({
                                'chunk_number': i,
                                'total_chunks': len(content_chunks),
                                'original_title': base_title,
                                'is_chunk': True,
                                'document_hash': hashlib.md5(doc['content'].encode()).hexdigest()
                            })
                            chunk_properties['metadata_str'] = json.dumps(chunk_metadata)
                        
                        chunk_properties['content'] = chunk
                        doc_id = batch.add_data_object(
                            data_object=chunk_properties,
                            class_name=self.class_name
                        )
                        document_ids.append(doc_id)
            
            logging.info(f"Added {len(documents)} documents in batch")
            return document_ids
            
        except Exception as e:
            logging.error(f"Error in batch document addition: {e}")
            raise

    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.schema.delete_class(self.class_name)
            self._create_schema()
            logging.info("Cleared all documents and recreated schema")
            return True
            
        except Exception as e:
            logging.error(f"Error clearing documents: {e}")
            return False