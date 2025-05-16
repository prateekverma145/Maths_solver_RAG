"""
Streamlit web application for JEE Bench Q&A
"""
import logging
import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pandas as pd
from streamlit_chat import message
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_jee_dataset, create_dataframe, filter_questions, get_question_by_index
from src.setup import verify_api_keys, setup_vector_store
from src.vector_store import collection_exists_with_data, get_vector_store
from src.workflow import run_workflow
from config.config import SUBJECTS, QUESTION_TYPES, QDRANT_COLLECTION, QDRANT_MODE, QDRANT_PERSIST_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamlit_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="JEE Bench Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'api_keys_verified' not in st.session_state:
    st.session_state.api_keys_verified = False
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
# Add feedback state variables
if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False
if 'regeneration_used' not in st.session_state:
    st.session_state.regeneration_used = False
if 'last_message_id' not in st.session_state:
    st.session_state.last_message_id = 0

def check_api_keys() -> Dict[str, bool]:
    """
    Check if API keys are available
    
    Returns:
        Dictionary with API key verification results
    """
    results = verify_api_keys()
    st.session_state.api_keys_verified = all(results.values())
    return results

def initialize_data() -> None:
    """
    Initialize dataset and dataframe
    """
    try:
        with st.spinner("Loading dataset..."):
            # Load dataset
            dataset = load_jee_dataset()
            st.session_state.dataset = dataset
            
            # Create dataframe
            df = create_dataframe(dataset)
            st.session_state.dataframe = df
            st.session_state.filtered_df = df
            
            st.success(f"Dataset loaded with {len(dataset)} questions")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

def initialize_vector_store() -> None:
    """
    Initialize vector store once
    """
    try:
        if st.session_state.vector_store is None:
            with st.spinner("Initializing vector store..."):
                vector_store = get_vector_store()
                st.session_state.vector_store = vector_store
                st.session_state.vector_store_ready = True
                st.success("Vector store initialized successfully")
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")

def display_question_stats() -> None:
    """
    Display statistics about the questions
    """
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        
        # Get counts by subject and type
        subject_counts = df['subject'].value_counts()
        type_counts = df['type'].value_counts()
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Plot subject distribution
        with col1:
            st.subheader("Subject Distribution")
            fig1, ax1 = plt.subplots()
            ax1.pie(subject_counts, labels=subject_counts.index, autopct='%1.1f%%')
            ax1.axis('equal')
            st.pyplot(fig1)
        
        # Plot question type distribution
        with col2:
            st.subheader("Question Type Distribution")
            fig2, ax2 = plt.subplots()
            ax2.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
            ax2.axis('equal')
            st.pyplot(fig2)
        
        # Display counts in a table
        st.subheader("Question Counts")
        counts_df = pd.DataFrame({
            'Subject': subject_counts.index,
            'Count': subject_counts.values
        })
        st.write(counts_df)
        
        types_df = pd.DataFrame({
            'Type': type_counts.index,
            'Count': type_counts.values
        })
        st.write(types_df)

def filter_and_display_questions() -> None:
    """
    Filter and display questions based on selection
    """
    if st.session_state.dataframe is not None:
        st.subheader("Filter Questions")
        
        # Create filters
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.selectbox("Subject", ["All"] + SUBJECTS)
        
        with col2:
            q_type = st.selectbox("Question Type", ["All"] + QUESTION_TYPES)
        
        # Apply filters
        filtered_df = st.session_state.dataframe
        
        if subject != "All":
            filtered_df = filtered_df[filtered_df['subject'] == subject]
        
        if q_type != "All":
            filtered_df = filtered_df[filtered_df['type'] == q_type]
        
        st.session_state.filtered_df = filtered_df
        
        # Display filtered questions
        st.subheader(f"Questions ({len(filtered_df)} results)")
        
        if not filtered_df.empty:
            # Create a selection box
            selected_index = st.selectbox(
                "Select a question",
                filtered_df['index'].tolist(),
                format_func=lambda x: f"Q{x}: {filtered_df[filtered_df['index'] == x]['description'].iloc[0]}"
            )
            
            # Get selected question
            selected_question = get_question_by_index(st.session_state.dataset, selected_index)
            
            if selected_question:
                st.markdown("---")
                st.markdown(f"**Question {selected_index}** ({selected_question.get('subject', '')}, {selected_question.get('type', '')})")
                st.markdown(selected_question.get('question', ''))
                
                if st.button("Answer this question"):
                    st.session_state.current_question = selected_question.get('question', '')
                    answer_current_question()
        else:
            st.info("No questions match the selected filters.")

def display_chat_interface() -> None:
    """
    Display chat interface for question answering
    """
    st.subheader("Ask a custom JEE question")
    
    # Input field
    question = st.text_area("Enter your question here", height=150)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Submit Question"):
            if question:
                st.session_state.current_question = question
                answer_current_question()
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("Reset Conversation"):
            # Clear chat history and reset feedback state
            st.session_state.messages = []
            st.session_state.current_question = None
            st.session_state.current_answer = None
            st.session_state.awaiting_feedback = False
            st.session_state.regeneration_used = False
            st.success("Conversation history has been reset!")
            st.rerun()

def answer_current_question() -> None:
    """
    Answer the current question using the workflow
    """
    if st.session_state.current_question:
        with st.spinner("Generating answer... This may take a while."):
            try:
                # Run workflow with vector store from session state
                result = run_workflow(
                    question=st.session_state.current_question,
                    vector_store=st.session_state.vector_store
                )
                
                # Store answer
                st.session_state.current_answer = result.get('answer', '')
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": st.session_state.current_question,
                    "id": st.session_state.last_message_id
                })
                st.session_state.last_message_id += 1
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": st.session_state.current_answer,
                    "id": st.session_state.last_message_id,
                    "feedback": None  # None means no feedback yet
                })
                st.session_state.last_message_id += 1
                
                # Set awaiting feedback to True
                st.session_state.awaiting_feedback = True
                # Reset regeneration flag when generating a new answer
                st.session_state.regeneration_used = False
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                logger.error(f"Error generating answer: {e}")

def handle_feedback(feedback: str, message_id: int) -> None:
    """
    Handle user feedback on a generated answer
    
    Args:
        feedback: The feedback (positive or negative)
        message_id: The ID of the message receiving feedback
    """
    # Find the message by ID
    for i, msg in enumerate(st.session_state.messages):
        if msg.get("id") == message_id:
            # Update the message with feedback
            st.session_state.messages[i]["feedback"] = feedback
            
            # If negative feedback and regeneration not used yet, regenerate
            if feedback == "negative" and not st.session_state.regeneration_used:
                # Get the corresponding question (should be the previous message)
                if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                    question = st.session_state.messages[i-1]["content"]
                    
                    # Set current question to regenerate
                    st.session_state.current_question = question
                    
                    # Mark regeneration as used
                    st.session_state.regeneration_used = True
                    
                    # Generate a new answer
                    with st.spinner("Regenerating answer... This may take a while."):
                        try:
                            # Run workflow with vector store from session state
                            result = run_workflow(
                                question=question,
                                vector_store=st.session_state.vector_store
                            )
                            
                            # Store answer
                            new_answer = result.get('answer', '')
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"**Regenerated answer:**\n\n{new_answer}",
                                "id": st.session_state.last_message_id,
                                "feedback": None,
                                "is_regenerated": True
                            })
                            st.session_state.last_message_id += 1
                            
                        except Exception as e:
                            st.error(f"Error regenerating answer: {e}")
                            logger.error(f"Error regenerating answer: {e}")
            
            # No longer awaiting feedback
            st.session_state.awaiting_feedback = False
            break

def display_chat_history() -> None:
    """
    Display chat history with feedback options
    """
    if st.session_state.messages:
        st.subheader("Chat History")
        
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"msg_{i}")
            else:
                # if "$" in msg["content"] or "\\(" in msg["content"]:
                #     st.markdown(msg["content"], unsafe_allow_html=True)
                # else:
                #     message(msg["content"], is_user=False, key=f"msg_{i}")
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # If this is the last assistant message and feedback is None, show feedback buttons
                is_last_message = i == len(st.session_state.messages) - 1
                needs_feedback = msg.get("feedback") is None
                is_regenerated = msg.get("is_regenerated", False)
                
                if is_last_message and needs_feedback and st.session_state.awaiting_feedback and not is_regenerated:
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("👍 Helpful", key=f"positive_{i}"):
                            handle_feedback("positive", msg.get("id"))
                            st.rerun()
                    with col2:
                        if st.button("👎 Not Helpful", key=f"negative_{i}"):
                            handle_feedback("negative", msg.get("id"))
                            st.rerun()
                    with col3:
                        st.markdown("*Please provide feedback on this response*")
                
                # If regenerated answer has no feedback yet, show feedback buttons
                elif is_last_message and needs_feedback and is_regenerated:
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("👍 Better", key=f"positive_{i}"):
                            handle_feedback("positive", msg.get("id"))
                            st.rerun()
                    with col2:
                        if st.button("👎 Still Not Helpful", key=f"negative_{i}"):
                            handle_feedback("negative", msg.get("id"))
                            st.rerun()
                    with col3:
                        st.markdown("*Please provide feedback on the regenerated answer*")
                
                # If feedback already received, show it
                elif msg.get("feedback") is not None:
                    if msg.get("feedback") == "positive":
                        st.success("✓ Marked as helpful")
                    else:
                        st.error("✗ Marked as not helpful")

def sidebar_setup() -> None:
    """
    Sidebar for setup steps
    """
    st.sidebar.title("Setup")
    
    # Step 1: Verify API Keys
    st.sidebar.subheader("1. Verify API Keys")
    if st.sidebar.button("Verify API Keys"):
        results = check_api_keys()
        
        # Display results
        for key, available in results.items():
            if available:
                st.sidebar.success(f"{key.capitalize()} API key verified")
            else:
                st.sidebar.error(f"{key.capitalize()} API key missing")
    
    # Step 2: Load Dataset
    st.sidebar.subheader("2. Load Dataset")
    if st.sidebar.button("Load Dataset"):
        initialize_data()
    
    # Step 3: Initialize Vector Store
    st.sidebar.subheader("3. Initialize Vector Store")
    if st.sidebar.button("Initialize Vector Store"):
        initialize_vector_store()
    
    # Display session state for debugging
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write({
            "API Keys Verified": st.session_state.api_keys_verified,
            "Vector Store Ready": st.session_state.vector_store_ready,
            "Dataset Loaded": st.session_state.dataset is not None,
            "Awaiting Feedback": st.session_state.awaiting_feedback,
            "Regeneration Used": st.session_state.regeneration_used,
            "Message Count": len(st.session_state.messages),
        })

def main() -> None:
    """
    Main function for Streamlit app
    """
    st.title("JEE Bench Q&A System")
    
    # Setup sidebar
    sidebar_setup()
    
    # Main content
    if not st.session_state.api_keys_verified:
        st.warning("Please verify API keys in the sidebar.")
    elif st.session_state.dataset is None:
        st.warning("Please load the dataset in the sidebar.")
    elif not st.session_state.vector_store_ready:
        st.warning("Please initialize the vector store in the sidebar.")
    else:
        # Create tabs
        tab1, tab2 = st.tabs(["Question Browser", "Custom Questions"])
        
        with tab1:
            filter_and_display_questions()
        
        with tab2:
            display_chat_interface()
            display_chat_history()

if __name__ == "__main__":
    main() 