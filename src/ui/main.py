# ui/main.py
import streamlit as st
from ..config import Config
from ..services.rag_service import RAGService
from .document_management import show_document_management
from .search_chat import show_search_and_chat
from .analytics_dashboard import show_analytics_dashboard

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Enhanced RAG Application",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

def show_sidebar():
    """Display sidebar navigation"""
    with st.sidebar:
        st.title("📚 RAG Application")
        
        # Navigation
        nav_option = st.radio(
            "Navigation",
            ["Document Management", "Search & Chat", "Analytics Dashboard"]
        )
        
        # Display some basic stats if we have them
        if hasattr(st.session_state, 'rag_service'):
            try:
                stats = st.session_state.rag_service.get_document_statistics()
                st.divider()
                st.subheader("Quick Stats")
                st.metric("Total Documents", stats["total_documents"])
                st.metric("Document Types", len(stats["file_types"]))
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        return nav_option

def main():
    """Main application entry point"""
    setup_page_config()
    initialize_session_state()
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize RAG service if not already done
        if 'rag_service' not in st.session_state:
            st.session_state.rag_service = RAGService(
                Config.OPENAI_API_KEY,
                Config.WEAVIATE_URL,
                Config.WEAVIATE_API_KEY
            )
        
        # Show sidebar navigation
        nav_option = show_sidebar()
        
        # Main content area
        if nav_option == "Document Management":
            show_document_management(st.session_state.rag_service)
        elif nav_option == "Search & Chat":
            show_search_and_chat(st.session_state.rag_service)
        else:
            show_analytics_dashboard(st.session_state.rag_service)
            
    except ValueError as e:
        st.error("Configuration Error")
        st.error(str(e))
        st.info("Please check your .env file and ensure all required variables are set.")
        
    except Exception as e:
        st.error("Application Error")
        st.error(str(e))
        st.info("Please check the logs for more details.")
        
if __name__ == "__main__":
    main()