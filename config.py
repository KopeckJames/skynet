import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WEAVIATE_URL = "https://3hbemlvctn2km7f8fyfjcg.c0.us-west3.gcp.weaviate.cloud"
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    # Additional configuration
      # Chunking settings
    MAX_CHUNK_TOKENS = 7000  # Slightly below the 8192 limit to be safe
    CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks
    
    # Vector settings
    VECTOR_DIMENSIONS = 1536  # OpenAI ada-002 embedding dimensions
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    @staticmethod
    def validate():
        required_vars = ["OPENAI_API_KEY", "WEAVIATE_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Validate Weaviate URL
        if not Config.WEAVIATE_URL or not Config.WEAVIATE_URL.startswith("http"):
            raise ValueError("Invalid Weaviate URL. Please update the URL in config.py")