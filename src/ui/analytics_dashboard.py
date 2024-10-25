# ui/analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ..services.rag_service import RAGService

def show_key_metrics(rag_service: RAGService):
    """Display key metrics at the top of the dashboard"""
    try:
        stats = rag_service.get_document_statistics()
        
        # Create three columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                stats["total_documents"],
                help="Total number of documents in the database"
            )
        
        with col2:
            st.metric(
                "Document Types",
                len(stats["file_types"]),
                help="Number of different document types"
            )
        
        with col3:
            st.metric(
                "Unique Tags",
                stats["total_unique_tags"],
                help="Total number of unique tags across all documents"
            )
        
        with col4:
            if stats.get("last_added"):
                last_added = datetime.fromisoformat(stats["last_added"])
                days_ago = (datetime.utcnow() - last_added).days
                st.metric(
                    "Last Addition",
                    f"{days_ago}d ago",
                    help="Time since the last document was added"
                )
    
    except Exception as e:
        st.error(f"Error loading key metrics: {str(e)}")

# ui/analytics_dashboard.py (continued)

def show_document_distribution(rag_service: RAGService):
    """Display document type distribution chart"""
    try:
        stats = rag_service.get_document_statistics()
        
        if stats["file_types"]:
            # Create distribution DataFrame
            df = pd.DataFrame(
                list(stats["file_types"].items()),
                columns=['File Type', 'Count']
            )
            
            # Create pie chart
            fig = px.pie(
                df,
                values='Count',
                names='File Type',
                title='Document Type Distribution',
                hole=0.4
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents found in the database")
            
    except Exception as e:
        st.error(f"Error showing document distribution: {str(e)}")

def show_timeline_analysis(rag_service: RAGService):
    """Display document addition timeline"""
    try:
        timeline = rag_service.get_document_timeline()
        
        if timeline:
            # Convert to DataFrame
            df = pd.DataFrame(timeline)
            df['date'] = pd.to_datetime(df['date'])
            
            # Group by date and count
            daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            
            # Create line chart
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title='Document Additions Over Time',
                labels={'date': 'Date', 'count': 'Documents Added'}
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent additions
            st.subheader("Recent Additions")
            recent_docs = df.sort_values('date', ascending=False).head(5)
            for _, doc in recent_docs.iterrows():
                st.write(f"📄 {doc['title']} ({doc['type']}) - {doc['date'].strftime('%Y-%m-%d')}")
        else:
            st.info("No timeline data available")
            
    except Exception as e:
        st.error(f"Error showing timeline analysis: {str(e)}")

def show_topic_analysis(rag_service: RAGService):
    """Display topic distribution and analysis"""
    try:
        topics = rag_service.get_topic_distribution()
        
        if topics:
            # Convert to DataFrame and sort by count
            df = pd.DataFrame(
                list(topics.items()),
                columns=['Topic', 'Count']
            ).sort_values('Count', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                df,
                x='Count',
                y='Topic',
                orientation='h',
                title='Topic Distribution',
                labels={'Count': 'Number of Documents', 'Topic': ''}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available")
            
    except Exception as e:
        st.error(f"Error showing topic analysis: {str(e)}")

def show_semantic_clusters(rag_service: RAGService):
    """Display semantic clustering analysis"""
    try:
        num_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=5,
            help="Adjust the number of semantic clusters"
        )
        
        if st.button("Generate Clusters"):
            with st.spinner("Performing semantic clustering..."):
                clusters = rag_service.semantic_clustering(num_clusters)
                
                if clusters:
                    # Display clusters
                    for cluster_id, docs in clusters.items():
                        with st.expander(f"Cluster {cluster_id + 1} ({len(docs)} documents)"):
                            # Try to identify cluster theme
                            if docs:
                                titles = [doc['title'] for doc in docs]
                                titles_text = " ".join(titles)
                                
                                # Extract common themes
                                themes = rag_service.document_processor.extract_keywords(
                                    titles_text,
                                    num_keywords=3
                                )
                                if themes:
                                    st.markdown(f"**Common themes:** {', '.join(themes)}")
                            
                            # List documents in cluster
                            for doc in docs:
                                st.markdown(f"- **{doc['title']}**")
                                if doc.get('summary'):
                                    with st.expander("Summary"):
                                        st.write(doc['summary'])
                else:
                    st.info("Not enough documents for clustering")
                    
    except Exception as e:
        st.error(f"Error performing semantic clustering: {str(e)}")

def show_tag_analysis(rag_service: RAGService):
    """Display tag analysis and co-occurrence"""
    try:
        # Get documents with their tags
        documents = rag_service.weaviate_client.search_documents("", limit=1000)
        
        if documents:
            # Collect all tags and their frequencies
            tag_freq = {}
            tag_cooccurrence = {}
            
            for doc in documents:
                tags = doc.get('tags', [])
                for tag in tags:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1
                    
                    # Calculate tag co-occurrences
                    for other_tag in tags:
                        if tag != other_tag:
                            if tag not in tag_cooccurrence:
                                tag_cooccurrence[tag] = {}
                            tag_cooccurrence[tag][other_tag] = tag_cooccurrence[tag].get(other_tag, 0) + 1
            
            # Create tag cloud data
            tag_df = pd.DataFrame(
                list(tag_freq.items()),
                columns=['Tag', 'Count']
            ).sort_values('Count', ascending=False)
            
            # Display top tags
            st.subheader("Most Common Tags")
            fig = px.bar(
                tag_df.head(15),
                x='Tag',
                y='Count',
                title='Top 15 Tags'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display tag co-occurrence matrix
            st.subheader("Tag Co-occurrence")
            top_tags = tag_df.head(10)['Tag'].tolist()
            cooccurrence_matrix = []
            
            for tag1 in top_tags:
                row = []
                for tag2 in top_tags:
                    count = tag_cooccurrence.get(tag1, {}).get(tag2, 0)
                    row.append(count)
                cooccurrence_matrix.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=cooccurrence_matrix,
                x=top_tags,
                y=top_tags,
                colorscale='Viridis'
            ))
            fig.update_layout(
                title='Tag Co-occurrence Matrix',
                xaxis_title='Tags',
                yaxis_title='Tags'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tags found in documents")
            
    except Exception as e:
        st.error(f"Error analyzing tags: {str(e)}")

def show_analytics_dashboard(rag_service: RAGService):
    """Main analytics dashboard interface"""
    st.header("📊 Analytics Dashboard")
    
    try:
        # Get statistics
        stats = rag_service.get_document_statistics()
        
        # Show key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", stats.get("total_documents", 0))
        
        with col2:
            st.metric("Document Types", len(stats.get("file_types", {})))
        
        with col3:
            st.metric("Unique Tags", stats.get("total_unique_tags", 0))
        
        # Show document distribution
        st.subheader("Document Type Distribution")
        file_types = stats.get("file_types", {})
        if file_types:
            df = pd.DataFrame(list(file_types.items()), columns=['Type', 'Count'])
            st.bar_chart(df.set_index('Type'))
        else:
            st.info("No documents found in the database")
        
        # Show timeline
        st.subheader("Document Timeline")
        timeline = rag_service.get_document_timeline()
        if timeline:
            df = pd.DataFrame(timeline)
            df['date'] = pd.to_datetime(df['date'])
            st.line_chart(df.set_index('date').resample('D').size())
        else:
            st.info("No timeline data available")
        
        # Show topic distribution
        st.subheader("Topic Distribution")
        topics = rag_service.get_topic_distribution()
        if topics:
            df = pd.DataFrame(list(topics.items()), columns=['Topic', 'Count'])
            st.bar_chart(df.set_index('Topic'))
        else:
            st.info("No topic data available")
            
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
        st.info("Please check the application logs for more details")