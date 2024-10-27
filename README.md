# RAG Chatbot with Document Processing and API/URL Data Fetching

This is a Streamlit-based application that allows users to:
- Upload documents (PDF, DOCX, TXT).
- Extract text from webpages and store it.
- Fetch data from external APIs.
- Store and manage documents in a Weaviate database.
- Query the stored data using a chat interface with context-aware responses powered by GPT.

## Features
- **Document Uploads**: Supports PDF, DOCX, and TXT files, with previews before storage.
- **API Data Fetching**: Fetches and stores content from external APIs.
- **URL Content Extraction**: Extracts text content from any given webpage URL.
- **Chat Interface**: Uses OpenAI’s GPT-3.5 to answer questions based on stored content.
- **Document Management**: View stored documents and delete any unwanted entries.

## Prerequisites
- Python 3.8+
- [Weaviate](https://weaviate.io/) instance set up and running (locally or on the cloud)
- OpenAI API key

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

pip install -r requirements.txt

# Add API Keys
Create a .streamlit/secrets.toml file in the root of the project to store your OpenAI and Weaviate API credentials:
# .streamlit/secrets.toml
[secrets]
```OPENAI_API_KEY = "your-openai-api-key"
WEAVIATE_URL = "your-weaviate-instance-url"
WEAVIATE_API_KEY = "your-weaviate-api-key"```
#Replace your-openai-api-key, your-weaviate-instance-url, and your-weaviate-api-key with your actual credentials.

Start the Streamlit app using the following command: streamlit run app.py

##Installing Tesseract
# Tesseract Installation on Windows
Download Tesseract:

Visit the official Tesseract GitHub page: https://github.com/UB-Mannheim/tesseract/wiki
Download the latest executable installer for Windows.
Install Tesseract:

Run the downloaded installer.
Follow the installation instructions and choose a directory where you want Tesseract to be installed.
Add Tesseract to PATH:

Open the Start Menu and search for Environment Variables.
Click on Edit the system environment variables.
In the System Properties window, click on Environment Variables.
Under System Variables, select Path and click Edit.
Click New and add the path to the tesseract.exe file (e.g., C:\Program Files\Tesseract-OCR\).
Click OK to save.
Verify Installation:

Open a command prompt (cmd) and type:
bash
Copy code
tesseract --version
You should see the Tesseract version displayed.



#Tesseract Installation on macOS
Install using Homebrew:

If you don't have Homebrew installed, install it using the following command in Terminal:
bash
Copy code
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Once Homebrew is installed, run:
bash
Copy code
brew install tesseract
Verify Installation:

Check if Tesseract is correctly installed by running:
bash
Copy code
tesseract --version
This should display the installed version of Tesseract.


##Installing Poppler
Poppler is a PDF rendering library, which includes pdftoppm and pdftocairo utilities that are often used for converting PDFs to images.

#Poppler Installation on Windows
Download Poppler:

Visit https://github.com/oschwartz10612/poppler-windows/releases/
Download the latest Poppler binary for Windows (usually a .zip file).
Install Poppler:

Extract the contents of the zip file to a folder, e.g., C:\poppler-23.05.0\.
Add Poppler to PATH:

Open the Start Menu and search for Environment Variables.
Click on Edit the system environment variables.
In the System Properties window, click on Environment Variables.
Under System Variables, select Path and click Edit.
Click New and add the path to the bin folder inside the extracted Poppler folder (e.g., C:\poppler-23.05.0\Library\bin\).
Click OK to save.
Verify Installation:

Open a command prompt (cmd) and type:
bash
Copy code
pdftoppm -v
This should display the version of pdftoppm.

##Poppler Installation on macOS
Install using Homebrew:

If you don't have Homebrew installed, install it using the following command in Terminal:
bash
Copy code
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Once Homebrew is installed, run:
bash
Copy code
brew install poppler
Verify Installation:

Check if Poppler is correctly installed by running:
bash
Copy code
pdftoppm -v
This should display the installed version of pdftoppm.
Summary of Commands
Operation	Windows	macOS
Install Tesseract	Download from GitHub	brew install tesseract
Add Tesseract to PATH	Add C:\Program Files\Tesseract-OCR\ to PATH	Not needed (Homebrew handles this)
Verify Tesseract	tesseract --version	tesseract --version
Install Poppler	Download from GitHub	brew install poppler
Add Poppler to PATH	Add C:\path\to\poppler\bin\ to PATH	Not needed (Homebrew handles this)
Verify Poppler	pdftoppm -v	pdftoppm -v


##Note for Python Usage
Python Libraries: After installing Tesseract and Poppler, you may also need to install Python libraries like pytesseract and pdf2image to interact with these tools in your Python code:

bash
Copy code
pip install pytesseract pdf2image
Configure pytesseract:

In your Python code, you may need to specify the Tesseract executable path if it isn't automatically detected:

python
Copy code
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed
Following these instructions, you should have both Tesseract and Poppler installed and ready to use on both Windows and macOS systems. This setup will allow your Python application to perform OCR on images using Tesseract and convert PDFs to images using Poppler.
