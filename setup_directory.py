# setup_directory.py
import os
from pathlib import Path

def setup_project_structure():
    """Create project directory structure and __init__ files"""
    # Define the directory structure
    directories = [
        'src',
        'src/processors',
        'src/models',
        'src/database',
        'src/services',
        'src/utils',
        'src/ui',
    ]
    
    # Create directories and __init__ files
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Initialize package\n')
    
    # Create minimal config.py if it doesn't exist
    if not os.path.exists('src/config.py'):
        with open('src/config.py', 'w') as f:
            f.write('''import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WEAVIATE_URL = "https://your-weaviate-cluster-url.weaviate.cloud"
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    @staticmethod
    def validate():
        required_vars = ["OPENAI_API_KEY", "WEAVIATE_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
''')
    
    print("Project structure created successfully!")
    print("\nDirectory structure:")
    os.system('tree' if os.name != 'nt' else 'dir /s /b')

if __name__ == "__main__":
    setup_project_structure()