# utils/text_utils.py

from typing import List, Dict, Any, Optional
import re
from collections import Counter
import string
from datetime import datetime
import hashlib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK data: {e}")

class TextUtils:
    """Utility class for text processing operations"""
    
    def __init__(self):
        """Initialize text processing tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
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

    def clean_text(self, text: str, remove_urls: bool = True, 
                  remove_emails: bool = True, remove_numbers: bool = False) -> str:
        """
        Clean and normalize text
        
        Args:
            text (str): Input text to clean
            remove_urls (bool): Whether to remove URLs
            remove_emails (bool): Whether to remove email addresses
            remove_numbers (bool): Whether to remove numbers
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                         '', text)
        
        # Remove email addresses
        if remove_emails:
            text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        try:
            return sent_tokenize(text)
        except Exception as e:
            logging.error(f"Error extracting sentences: {e}")
            return [text]

    def extract_keywords(self, text: str, min_word_length: int = 3, 
                        max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text (str): Input text
            min_word_length (int): Minimum length of words to consider
            max_keywords (int): Maximum number of keywords to return
            
        Returns:
            List[str]: List of keywords
        """
        # Clean and tokenize text
        clean_text = self.clean_text(text)
        words = word_tokenize(clean_text)
        
        # Filter words
        filtered_words = [
            self.lemmatizer.lemmatize(word.lower())
            for word in words
            if len(word) >= min_word_length
            and word.lower() not in self.stop_words
            and word.isalnum()
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        return [word for word, _ in word_freq.most_common(max_keywords)]

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Clean and tokenize texts
        set1 = set(word_tokenize(self.clean_text(text1)))
        set2 = set(word_tokenize(self.clean_text(text2)))
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def generate_text_hash(self, text: str) -> str:
        """
        Generate a hash for the text content
        
        Args:
            text (str): Input text
            
        Returns:
            str: Hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text using regex patterns
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of found dates
        """
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}',  # 1 Jan 2024
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        
        return dates

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using NLTK
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        try:
            # Tokenize and tag parts of speech
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract named entities
            named_entities = nltk.chunk.ne_chunk(pos_tags)
            
            # Organize entities by type
            entities = {}
            for entity in named_entities:
                if hasattr(entity, 'label'):
                    entity_type = entity.label()
                    entity_text = ' '.join([child[0] for child in entity])
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(entity_text)
            
            return entities
            
        except Exception as e:
            logging.error(f"Error extracting entities: {e}")
            return {}

    def generate_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Generate n-grams from text
        
        Args:
            text (str): Input text
            n (int): Size of n-grams to generate
            
        Returns:
            List[str]: List of n-grams
        """
        words = word_tokenize(text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Dictionary of readability metrics
        """
        try:
            sentences = self.extract_sentences(text)
            words = word_tokenize(text)
            
            # Calculate basic metrics
            num_sentences = len(sentences)
            num_words = len(words)
            num_characters = len(text)
            
            # Average sentence length
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            
            # Average word length
            avg_word_length = num_characters / num_words if num_words > 0 else 0
            
            # Calculate Flesch Reading Ease score
            flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length
            
            return {
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2),
                "flesch_score": round(flesch_score, 2)
            }
            
        except Exception as e:
            logging.error(f"Error calculating readability score: {e}")
            return {}

    def extract_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Extract various statistics from text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Dictionary of text statistics
        """
        try:
            words = word_tokenize(text)
            sentences = self.extract_sentences(text)
            
            # Calculate statistics
            stats = {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "unique_words": len(set(words)),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0
            }
            
            # Add readability metrics
            stats.update(self.calculate_readability_score(text))
            
            # Round floating point values
            for key, value in stats.items():
                if isinstance(value, float):
                    stats[key] = round(value, 2)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating text statistics: {e}")
            return {}

    @staticmethod
    def is_valid_text(text: Optional[str]) -> bool:
        """
        Check if text is valid for processing
        
        Args:
            text (Optional[str]): Input text
            
        Returns:
            bool: Whether text is valid
        """
        if text is None:
            return False
        if not isinstance(text, str):
            return False
        if not text.strip():
            return False
        return True

    @staticmethod
    def truncate_text(text: str, max_length: int = 100, 
                     add_ellipsis: bool = True) -> str:
        """
        Truncate text to specified length
        
        Args:
            text (str): Input text
            max_length (int): Maximum length
            add_ellipsis (bool): Whether to add ellipsis
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
            
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + '...' if add_ellipsis else truncated

    @staticmethod
    def remove_duplicate_whitespace(text: str) -> str:
        """
        Remove duplicate whitespace from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        return ' '.join(text.split())

    @staticmethod
    def sanitize_filename(text: str) -> str:
        """
        Convert text to safe filename
        
        Args:
            text (str): Input text
            
        Returns:
            str: Safe filename
        """
        # Remove invalid filename characters
        filename = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_'))
        # Replace spaces with underscores and convert to lowercase
        return filename.replace(' ', '_').lower()