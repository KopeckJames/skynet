# models/document.py

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
from pathlib import Path
import uuid

@dataclass
class Document:
    """
    Document model representing a processed document in the system.
    Includes content, metadata, and analysis results.
    """
    
    # Required fields
    title: str
    content: str
    file_type: str
    
    # Optional fields with defaults
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    sentiment: Dict[str, float] = field(default_factory=dict)
    topics: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    
    # System fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: str = field(init=False)
    vector_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Generate content hash
        self.content_hash = self._generate_hash()
        
        # Ensure metadata contains basic information
        self.metadata.update({
            "file_type": self.file_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_hash": self.content_hash
        })

    def _generate_hash(self) -> str:
        """Generate a unique hash based on the document content"""
        content_string = f"{self.title}{self.content}{self.file_type}"
        return hashlib.sha256(content_string.encode()).hexdigest()

    def update(self, **kwargs) -> None:
        """
        Update document attributes and metadata
        
        Args:
            **kwargs: Attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.utcnow()
        self.metadata["updated_at"] = self.updated_at.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary format
        
        Returns:
            Dict[str, Any]: Document as dictionary
        """
        doc_dict = asdict(self)
        
        # Convert datetime objects to ISO format strings
        doc_dict["created_at"] = self.created_at.isoformat()
        doc_dict["updated_at"] = self.updated_at.isoformat()
        
        return doc_dict

    def to_json(self) -> str:
        """
        Convert document to JSON string
        
        Returns:
            str: JSON representation of document
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create document instance from dictionary
        
        Args:
            data: Dictionary containing document data
            
        Returns:
            Document: New document instance
        """
        # Convert ISO format strings back to datetime objects
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """
        Create document instance from JSON string
        
        Args:
            json_str: JSON string containing document data
            
        Returns:
            Document: New document instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, file_path: str, content: str, **kwargs) -> 'Document':
        """
        Create document instance from file
        
        Args:
            file_path: Path to the source file
            content: Extracted content from the file
            **kwargs: Additional attributes
            
        Returns:
            Document: New document instance
        """
        path = Path(file_path)
        file_type = cls._get_file_type(path)
        
        # Create basic metadata
        metadata = {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "file_type": file_type,
            "file_extension": path.suffix,
        }
        
        # Create document instance
        return cls(
            title=path.stem,
            content=content,
            file_type=file_type,
            metadata=metadata,
            **kwargs
        )

    @staticmethod
    def _get_file_type(path: Path) -> str:
        """
        Determine file type based on extension
        
        Args:
            path: Path object
            
        Returns:
            str: MIME type
        """
        mime_types = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.csv': 'text/csv',
            '.md': 'text/markdown'
        }
        return mime_types.get(path.suffix.lower(), 'application/octet-stream')

    def validate(self) -> bool:
        """
        Validate document data
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.title or not self.title.strip():
            return False
        
        if not self.content or not self.content.strip():
            return False
        
        if not self.file_type or not self.file_type.strip():
            return False
        
        return True

    def calculate_size(self) -> int:
        """
        Calculate total size of document in bytes
        
        Returns:
            int: Size in bytes
        """
        return len(json.dumps(self.to_dict()).encode())

    def summarize(self) -> Dict[str, Any]:
        """
        Get document summary information
        
        Returns:
            Dict[str, Any]: Summary of document
        """
        return {
            "id": self.id,
            "title": self.title,
            "file_type": self.file_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_length": len(self.content),
            "has_summary": bool(self.summary),
            "num_tags": len(self.tags),
            "num_topics": len(self.topics),
            "num_entities": sum(len(entities) for entities in self.entities.values()),
            "num_keywords": len(self.keywords)
        }

    def merge(self, other: 'Document') -> None:
        """
        Merge another document's data into this document
        
        Args:
            other: Another document instance
        """
        self.tags = list(set(self.tags + other.tags))
        self.topics.extend(other.topics)
        
        # Merge entities
        for entity_type, entities in other.entities.items():
            if entity_type in self.entities:
                self.entities[entity_type] = list(set(self.entities[entity_type] + entities))
            else:
                self.entities[entity_type] = entities
        
        self.keywords = list(set(self.keywords + other.keywords))
        self.metadata.update(other.metadata)
        self.updated_at = datetime.utcnow()

    def __str__(self) -> str:
        """String representation of document"""
        return f"Document(id={self.id}, title={self.title}, type={self.file_type})"

    def __len__(self) -> int:
        """Get content length"""
        return len(self.content)