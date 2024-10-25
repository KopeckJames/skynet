# processors/base_processor.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from datetime import datetime
import hashlib
import json
from pathlib import Path

class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

class BaseProcessor(ABC):
    """
    Abstract base class for all processors in the system.
    Defines common interface and utility methods for processing different types of content.
    """

    def __init__(self):
        """Initialize the base processor"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @abstractmethod
    def process(self, content: Any) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by all processor classes.
        
        Args:
            content: The content to process (type varies by processor)
            
        Returns:
            Dict[str, Any]: Processed content and metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        pass

    def validate_content(self, content: Any) -> bool:
        """
        Validate content before processing.
        
        Args:
            content: Content to validate
            
        Returns:
            bool: True if content is valid, False otherwise
        """
        if content is None:
            self.logger.warning("Received None content")
            return False
            
        if isinstance(content, (str, bytes)) and not content:
            self.logger.warning("Received empty content")
            return False
            
        return True

    def generate_metadata(self, content: Any) -> Dict[str, Any]:
        """
        Generate standard metadata for processed content.
        
        Args:
            content: Content to generate metadata for
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        try:
            metadata = {
                "processor": self.__class__.__name__,
                "processed_at": datetime.utcnow().isoformat(),
                "content_hash": self._generate_content_hash(content),
                "content_type": self._detect_content_type(content),
                "size": self._get_content_size(content)
            }
            
            return metadata
        except Exception as e:
            self.logger.error(f"Error generating metadata: {str(e)}")
            return {}

    def _generate_content_hash(self, content: Any) -> str:
        """
        Generate a hash of the content for tracking and verification.
        
        Args:
            content: Content to hash
            
        Returns:
            str: Content hash
        """
        try:
            if isinstance(content, str):
                content_bytes = content.encode()
            elif isinstance(content, bytes):
                content_bytes = content
            elif isinstance(content, (dict, list)):
                content_bytes = json.dumps(content, sort_keys=True).encode()
            elif hasattr(content, 'read') and callable(content.read):
                content_bytes = content.read()
                content.seek(0)  # Reset file pointer
            else:
                content_bytes = str(content).encode()
            
            return hashlib.sha256(content_bytes).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating content hash: {str(e)}")
            return ""

    def _detect_content_type(self, content: Any) -> str:
        """
        Detect the type of content being processed.
        
        Args:
            content: Content to analyze
            
        Returns:
            str: Detected content type
        """
        if isinstance(content, str):
            return "text/plain"
        elif isinstance(content, bytes):
            return "application/octet-stream"
        elif isinstance(content, dict):
            return "application/json"
        elif isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            return self._get_mime_type(str(content))
        else:
            return "application/unknown"

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get MIME type for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: MIME type
        """
        extension = Path(file_path).suffix.lower()
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
        return mime_types.get(extension, 'application/octet-stream')

    def _get_content_size(self, content: Any) -> int:
        """
        Calculate the size of the content in bytes.
        
        Args:
            content: Content to measure
            
        Returns:
            int: Content size in bytes
        """
        try:
            if isinstance(content, (str, bytes)):
                return len(content)
            elif isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
                return Path(content).stat().st_size
            elif isinstance(content, (dict, list)):
                return len(json.dumps(content).encode())
            elif hasattr(content, 'seek') and hasattr(content, 'tell'):
                current_pos = content.tell()
                content.seek(0, 2)  # Seek to end
                size = content.tell()
                content.seek(current_pos)  # Restore position
                return size
            else:
                return len(str(content).encode())
        except Exception as e:
            self.logger.error(f"Error calculating content size: {str(e)}")
            return 0

    def handle_error(self, error: Exception, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle processing errors in a standardized way.
        
        Args:
            error: The exception that occurred
            content_type: Type of content being processed when error occurred
            
        Returns:
            Dict[str, Any]: Error information
        """
        error_info = {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "processor": self.__class__.__name__
        }
        
        if content_type:
            error_info["content_type"] = content_type
            
        self.logger.error(f"Processing error: {str(error)}", exc_info=True)
        return error_info

    def process_with_error_handling(self, content: Any) -> Dict[str, Any]:
        """
        Wrapper method to process content with standardized error handling.
        
        Args:
            content: Content to process
            
        Returns:
            Dict[str, Any]: Processing results or error information
        """
        try:
            if not self.validate_content(content):
                raise ProcessingError("Invalid content")
                
            metadata = self.generate_metadata(content)
            
            start_time = datetime.utcnow()
            results = self.process(content)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            results.update({
                "metadata": metadata,
                "processing_time": processing_time,
                "success": True
            })
            
            return results
            
        except Exception as e:
            return self.handle_error(e, self._detect_content_type(content))

    def cleanup(self) -> None:
        """
        Cleanup any resources used by the processor.
        Should be called when the processor is no longer needed.
        """
        pass

    def __enter__(self):
        """Support for using processors in 'with' statements"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting 'with' block"""
        self.cleanup()