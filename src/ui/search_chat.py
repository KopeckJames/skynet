# ui/search_chat.py
import streamlit as st
import pandas as pd
from typing import List, Dict
from ..services.rag_service import RAGService

def display_chat_message(message: Dict[str, str], is_user: bool):
    """Display a chat message with appropriate styling"""
    with st.chat_message("user" if is_user else "assistant"):
        st.markdown(message["content"])

def display_search_results(results: List[Dict], query: str, response: str):
    """Display search results and AI response"""
    # Display AI response
    st.markdown("### 🤖 AI Response")
    st.write(response)
    
    # Display source documents
    st.markdown("### 📚 Source Documents")
    for i, doc in enumerate(results, 1):
        with st.expander(f"Document {i}: {doc['title']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Content:**")
                st.write(doc['content'])
            
            with col2:
                st.markdown("**Details:**")
                st.write(f"Type: {doc.get('file_type', 'Unknown')}")
                st.write(f"Added: {doc.get('created_at', 'Unknown')}")
                
                if doc.get('tags'):
                    st.markdown("**Tags:**")
                    st.write(", ".join(doc['tags']))
                
                # Show relevance score if available
                if 'certainty' in doc.get('_additional', {}):
                    relevance = doc['_additional']['certainty'] * 100
                    st.metric("Relevance", f"{relevance:.1f}%")

def show_semantic_search(rag_service: RAGService):
    """Display semantic search interface"""
    st.markdown("### 🔍 Semantic Search")
    
    # Search input
    query = st.text_input("Enter your search query:")
    num_results = st.slider("Number of results", 1, 10, 3)
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                try:
                    # Get search results
                    results = rag_service.search_documents(query, limit=num_results)
                    
                    if results:
                        # Generate AI response
                        response = rag_service.generate_response(query, results)
                        
                        # Display results
                        display_search_results(results, query, response)
                    else:
                        st.info("No relevant documents found")
                        
                except Exception as e:
                    st.error(f"Error performing search: {str(e)}")
        else:
            st.warning("Please enter a search query")

def show_chat_interface(rag_service: RAGService):
    """Display chat interface"""
    st.markdown("### 💬 Chat with Your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, message["role"] == "user")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message({"content": prompt}, True)
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = rag_service.chat_with_context(
                    st.session_state.messages,
                    prompt
                )
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                display_chat_message({"content": response}, False)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

def show_search_and_chat(rag_service: RAGService):
    """Main search and chat interface"""
    st.header("🔍 Search and Chat")
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["🔍 Semantic Search", "💬 Chat Interface"])
    
    with tab1:
        show_semantic_search(rag_service)
    
    with tab2:
        show_chat_interface(rag_service)

    # Add helpful tips in sidebar
    with st.sidebar:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use **Semantic Search** for specific queries about your documents
        - Use **Chat Interface** for more interactive conversations
        - You can reference previous messages in the chat
        - The system will find relevant documents automatically
        """)