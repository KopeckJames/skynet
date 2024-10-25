# run_app.py
import streamlit as st
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Verify environment variables
required_vars = ['OPENAI_API_KEY', 'WEAVIATE_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.info("Please set these variables in your .env file")
    st.stop()

def main():
    st.title("📚 RAG Application")
    
    try:
        from src.config import Config
        from src.services.rag_service import RAGService
        
        # Initialize RAG service
        if 'rag_service' not in st.session_state:
            with st.spinner("Initializing RAG service..."):
                st.session_state.rag_service = RAGService(
                    Config.OPENAI_API_KEY,
                    Config.WEAVIATE_URL,
                    Config.WEAVIATE_API_KEY
                )
            st.success("Successfully connected to services!")
        
        # Create main navigation
        nav_option = st.sidebar.selectbox(
            "Navigation",
            ["Document Management", "Search & Chat", "Analytics Dashboard"]
        )
        
        # Show appropriate interface based on selection
        if nav_option == "Document Management":
            from src.ui.document_management import show_document_management
            show_document_management(st.session_state.rag_service)
        elif nav_option == "Search & Chat":
            from src.ui.search_chat import show_search_and_chat
            show_search_and_chat(st.session_state.rag_service)
        else:
            from src.ui.analytics_dashboard import show_analytics_dashboard
            show_analytics_dashboard(st.session_state.rag_service)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your configuration and try again.")
        logging.exception("Application error")

if __name__ == "__main__":
    main()