# ui/document_management.py
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import tempfile
import time
from ..services.rag_service import RAGService

def show_upload_section(rag_service: RAGService):
    """Display file upload section"""
    st.subheader("📤 Upload Documents")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload (Text, PDF, Image, Audio, Video, PowerPoint, Excel, CSV)",
        type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", 
              "avi", "pptx", "xlsx", "csv", "doc", "docx"],
        accept_multiple_files=True
    )
    
    # Web URL input
    webpage_url = st.text_input(
        "Or enter a webpage URL to process",
        placeholder="https://example.com/article"
    )
    
    # Document title
    doc_title = st.text_input(
        "Document Title",
        placeholder="Enter a title for your document(s)",
        key="doc_title"
    )
    
    # Process button
    if st.button("Process and Add Document(s)"):
        if (uploaded_files or webpage_url) and doc_title:
            process_documents(rag_service, uploaded_files, webpage_url, doc_title)
        else:
            st.warning("Please provide both a document/URL and a title")

def process_documents(
    rag_service: RAGService,
    uploaded_files: list,
    webpage_url: str,
    base_title: str
):
    """Process and add documents to the database"""
    with st.spinner("Processing document(s)..."):
        try:
            # Process uploaded files
            if uploaded_files:
                for idx, file in enumerate(uploaded_files):
                    # Generate unique title for multiple files
                    title = f"{base_title} {idx+1}" if len(uploaded_files) > 1 else base_title
                    
                    with st.expander(f"Processing: {file.name}", expanded=True):
                        # Process the file
                        file_data = rag_service.process_file(file, file.name)
                        
                        if file_data:
                            # Show analysis results
                            display_analysis_results(file_data)
                            
                            # Add to database
                            success = rag_service.add_document(title, file_data)
                            if success:
                                st.success(f"✅ Successfully added: {title}")
                            else:
                                st.error(f"Failed to add document: {title}")
            
            # Process webpage
            if webpage_url:
                with st.expander(f"Processing webpage: {webpage_url}", expanded=True):
                    webpage_data = rag_service.process_webpage(webpage_url)
                    if webpage_data:
                        # Show analysis results
                        display_analysis_results(webpage_data)
                        
                        # Add to database
                        success = rag_service.add_document(base_title, webpage_data)
                        if success:
                            st.success(f"✅ Successfully added webpage: {base_title}")
                        else:
                            st.error("Failed to add webpage")
            
            st.session_state.processing_complete = True
            
        except Exception as e:
            st.error(f"Error processing document(s): {str(e)}")

def display_analysis_results(file_data: Dict[str, Any]):
    """Display document analysis results"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary
        if "summary" in file_data:
            with st.expander("📝 Summary", expanded=True):
                st.write(file_data["summary"])
        
        # Sentiment Analysis
        if "sentiment" in file_data:
            with st.expander("😊 Sentiment Analysis"):
                sentiment_df = pd.DataFrame({
                    'Sentiment': list(file_data["sentiment"].keys()),
                    'Score': list(file_data["sentiment"].values())
                })
                st.bar_chart(sentiment_df.set_index('Sentiment'))
    
    with col2:
        # Tags
        if "tags" in file_data:
            with st.expander("🏷️ Generated Tags", expanded=True):
                st.write("Click to copy:")
                for tag in file_data["tags"]:
                    st.code(f"#{tag}", language=None)
        
        # Main Topics
        if "topics" in file_data:
            with st.expander("📌 Main Topics"):
                for topic in file_data["topics"]:
                    st.markdown(f"**{topic['topic']}**")
                    st.write(topic['description'])
                    st.divider()

def show_document_list(rag_service: RAGService):
    """Display list of stored documents"""
    st.subheader("📚 Document Library")
    
    # Search and filters
    col1, col2, col3 = st.columns(3)
    with col1:
        search_query = st.text_input("🔍 Search documents", key="doc_search")
    with col2:
        file_types = rag_service.get_document_statistics()["file_types"]
        selected_type = st.selectbox(
            "Filter by type",
            ["All"] + list(file_types.keys())
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First", "Title A-Z", "Title Z-A"]
        )
    
    # Get documents based on filters
    try:
        if search_query:
            documents = rag_service.search_documents(search_query, limit=100)
        else:
            filters = {}
            if selected_type != "All":
                filters["file_type"] = selected_type
            documents = rag_service.weaviate_client.search_by_filters(filters, limit=100)
        
        # Sort documents
        if sort_by == "Newest First":
            documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "Oldest First":
            documents.sort(key=lambda x: x.get("created_at", ""))
        elif sort_by == "Title A-Z":
            documents.sort(key=lambda x: x.get("title", "").lower())
        elif sort_by == "Title Z-A":
            documents.sort(key=lambda x: x.get("title", "").lower(), reverse=True)
        
        # Display documents
        for doc in documents:
            with st.expander(f"📄 {doc['title']}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write("**Summary:**")
                    st.write(doc.get("summary", "No summary available"))
                
                with col2:
                    st.write("**Details:**")
                    st.write(f"Type: {doc.get('file_type', 'Unknown')}")
                    st.write(f"Added: {doc.get('created_at', 'Unknown')}")
                    if doc.get("tags"):
                        st.write("Tags:", ", ".join(doc["tags"]))
                
                with col3:
                    st.write("**Actions:**")
                    if st.button("🗑️ Delete", key=f"delete_{doc.get('id')}"):
                        if st.warning("Are you sure?"):
                            success = rag_service.weaviate_client.delete_document(doc.get('id'))
                            if success:
                                st.success("Document deleted")
                                time.sleep(1)
                                st.rerun()
                    
                    if st.button("📊 Similar", key=f"similar_{doc.get('id')}"):
                        similar_docs = rag_service.weaviate_client.get_similar_documents(doc.get('id'))
                        st.write("Similar documents:")
                        for similar in similar_docs:
                            st.write(f"- {similar['title']}")
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

def show_document_management(rag_service: RAGService):
    """Main document management interface"""
    st.header("📁 Document Management")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📤 Upload", "📚 Library"])
    
    with tab1:
        show_upload_section(rag_service)
    
    with tab2:
        show_document_list(rag_service)