# processors/web_processor.py
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import json
from .base_processor import BaseProcessor

class WebProcessor(BaseProcessor):
    """Processor for handling web content"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()

    def process(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process web content based on URL"""
        if not self.validate_content(content):
            raise ValueError("Invalid content provided")

        url = content.get("url")
        if not url:
            raise ValueError("URL is required")

        try:
            webpage_content = self.process_webpage(url)
            metadata = self.extract_metadata(url, webpage_content.get("html", ""))
            
            return {
                "content": webpage_content.get("text", ""),
                "html": webpage_content.get("html", ""),
                "metadata": metadata,
                "links": webpage_content.get("links", []),
                "type": "web_content"
            }
        except Exception as e:
            raise Exception(f"Error processing web content: {str(e)}")

    def process_webpage(self, url: str) -> Dict[str, Any]:
        """Extract content from a webpage"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            if response.encoding is None:
                response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "iframe", "nav", "footer"]):
                element.decompose()
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract links
            links = self._extract_links(soup, url)
            
            return {
                "text": main_content,
                "html": str(soup),
                "links": links
            }
        except requests.RequestException as e:
            raise Exception(f"Error fetching webpage: {str(e)}")

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from webpage"""
        # Try to find main content area
        main_content = None
        content_tags = ["main", "article", "div[role='main']", "#content", ".content"]
        
        for tag in content_tags:
            main_content = soup.select_one(tag)
            if main_content:
                break
        
        if not main_content:
            main_content = soup
        
        # Get text with preserved structure
        lines = []
        for element in main_content.stripped_strings:
            line = element.strip()
            if line:
                lines.append(line)
        
        return "\n".join(lines)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract and normalize links from webpage"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True)
            
            # Normalize URL
            full_url = urljoin(base_url, href)
            
            # Only include http(s) links
            if full_url.startswith(('http://', 'https://')):
                links.append({
                    'url': full_url,
                    'text': text if text else full_url
                })
        
        return links

    def extract_metadata(self, url: str, html: str) -> Dict[str, Any]:
        """Extract metadata from webpage"""
        soup = BeautifulSoup(html, 'html.parser')
        parsed_url = urlparse(url)
        
        metadata = {
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'title': self._get_title(soup),
            'description': self._get_meta_description(soup),
            'keywords': self._get_meta_keywords(soup),
            'author': self._get_meta_author(soup),
            'published_date': self._get_published_date(soup),
            'og_data': self._get_og_data(soup),
            'schema_data': self._get_schema_data(soup)
        }
        
        return {k: v for k, v in metadata.items() if v}  # Remove None values

    def _get_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None

    def _get_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description"""
        meta_desc = soup.find('meta', {'name': 'description'})
        return meta_desc.get('content', None) if meta_desc else None

    def _get_meta_keywords(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta keywords"""
        meta_keywords = soup.find('meta', {'name': 'keywords'})
        return meta_keywords.get('content', None) if meta_keywords else None

    def _get_meta_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information"""
        meta_author = soup.find('meta', {'name': 'author'})
        return meta_author.get('content', None) if meta_author else None

    def _get_published_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract published date"""
        pub_date = (
            soup.find('meta', {'property': 'article:published_time'}) or
            soup.find('meta', {'name': 'publication-date'}) or
            soup.find('time', {'datetime': True})
        )
        return pub_date.get('content', pub_date.get('datetime', None)) if pub_date else None

    def _get_og_data(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract OpenGraph data"""
        og_data = {}
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            key = tag.get('property', '')[3:]  # Remove 'og:' prefix
            og_data[key] = tag.get('content', '')
        return og_data

    def _get_schema_data(self, soup: BeautifulSoup) -> list:
        """Extract Schema.org structured data"""
        schema_data = []
        schema_tags = soup.find_all('script', type='application/ld+json')
        for tag in schema_tags:
            try:
                data = json.loads(tag.string)
                schema_data.append(data)
            except (json.JSONDecodeError, AttributeError):
                continue
        return schema_data