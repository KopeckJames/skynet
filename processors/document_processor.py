# processors/document_processor.py
from typing import Dict, List, Any
import spacy
from keybert import KeyBERT
import openai
import re
import logging
from .base_processor import BaseProcessor
from ..config import Config

class DocumentProcessor(BaseProcessor):
    """Processor for handling document content analysis"""

    def __init__(self, openai_api_key: str):
        """Initialize the document processor"""
        super().__init__()
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.nlp = spacy.load("en_core_web_sm")
        self.kw_model = KeyBERT()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text

    def process(self, content: str) -> Dict[str, Any]:
        """
        Process document content and perform analysis
        
        Args:
            content (str): Document content to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not self.validate_content(content):
            raise ValueError("Invalid content provided")

        cleaned_content = self.clean_text(content)
        
        try:
            return {
                "content": cleaned_content,
                "summary": self.summarize_text(cleaned_content),
                "sentiment": self.analyze_sentiment(cleaned_content),
                "tags": self.generate_tags(cleaned_content),
                "topics": self.extract_main_topics(cleaned_content),
                "entities": self.extract_entities(cleaned_content),
                "keywords": self.extract_keywords(cleaned_content)
            }
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise

    def summarize_text(self, content: str) -> str:
        """Generate a summary of the text content"""
        try:
            max_chunk_size = Config.MAX_CHUNK_SIZE
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            summaries = []
            
            for chunk in chunks:
                response = self.openai_client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": f"Summarize the following text:\n{chunk}"}
                    ],
                    max_tokens=150
                )
                summaries.append(response.choices[0].message.content.strip())
            
            return " ".join(summaries)
        except Exception as e:
            logging.error(f"Error summarizing text: {str(e)}")
            return ""

    def analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of the text content"""
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the sentiment of the following text and return a JSON with positive, negative, and neutral scores that sum to 1.0"
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=100
            )
            sentiment_text = response.choices[0].message.content.strip()
            
            # Extract values using regex
            sentiment = {}
            for match in re.finditer(r'"(\w+)":\s*(0\.\d+)', sentiment_text):
                sentiment[match.group(1)] = float(match.group(2))
            
            # Ensure all components are present
            default_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            default_sentiment.update(sentiment)
            return default_sentiment
            
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {str(e)}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def generate_tags(self, content: str) -> List[str]:
        """Generate relevant tags for the content"""
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 5-7 relevant tags for the content. Return as a comma-separated list."
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=100
            )
            
            tags_text = response.choices[0].message.content.strip()
            tags = [tag.strip() for tag in tags_text.split(',')]
            return [tag for tag in tags if tag]  # Remove any empty tags
            
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            return []

    def extract_main_topics(self, content: str) -> List[Dict[str, str]]:
        """Extract main topics with descriptions"""
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 3-5 main topics with brief descriptions. Format as: Topic: Description"
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=500
            )
            
            topics_text = response.choices[0].message.content.strip()
            topics = []
            
            # Parse topics and descriptions
            for line in topics_text.split('\n'):
                if ':' in line:
                    topic, description = line.split(':', 1)
                    topics.append({
                        "topic": topic.strip(),
                        "description": description.strip()
                    })
            
            return topics
            
        except Exception as e:
            logging.error(f"Error extracting topics: {str(e)}")
            return []

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
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
            logging.error(f"Error extracting entities: {str(e)}")
            return {}

    def extract_keywords(self, content: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords using KeyBERT"""
        try:
            keywords = self.kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=num_keywords
            )
            return [kw[0] for kw in keywords]
            
        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            return []

    def validate_content(self, content: Any) -> bool:
        """Validate content before processing"""
        if not super().validate_content(content):
            return False
            
        # Additional document-specific validation
        if not isinstance(content, str):
            return False
            
        if len(content.strip()) == 0:
            return False
            
        return True