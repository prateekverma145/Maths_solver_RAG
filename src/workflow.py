import logging
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated, cast

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from src.data_loader import get_few_shot_example, load_few_shot_examples
from src.tavily_search import search_web, format_search_results
from src.vector_store import semantic_search
from src.groq_llm import init_groq_llm, answer_question

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define state type
class QuestionState(TypedDict):
    question: str
    subject: Optional[str]
    question_type: Optional[str]
    search_results: Optional[str]
    similar_questions: Optional[str]
    few_shot_example: Optional[Dict[str, str]]
    answer: Optional[str]

def process_similar_questions(results: List[Tuple[Document, float]]) -> str:
    """
    Process similar questions from vector search
    
    Args:
        results: List of (Document, score) tuples
        
    Returns:
        Formatted string of similar questions
    """
    if not results:
        return "No similar questions found."
    
    formatted_text = "### Similar Questions from Dataset:\n\n"
    
    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content
        metadata = doc.metadata
        
        formatted_text += f"**Similar Question {i} (Similarity: {score:.2f}):**\n"
        formatted_text += f"{content}\n"
        formatted_text += f"Subject: {metadata.get('subject', 'Unknown')}, "
        formatted_text += f"Type: {metadata.get('type', 'Unknown')}\n\n"
    
    return formatted_text

def retrieve_subject_and_type(state: QuestionState) -> QuestionState:
    """
    Retrieve subject and question type using LLM
    
    Args:
        state: Current state
        
    Returns:
        Updated state with subject and question type
    """
    try:
        logger.info("Retrieving subject and question type")
        
        # Get Groq LLM
        llm = init_groq_llm(temperature=0.1)
        
        # Ask LLM to categorize the question
        prompt = f"""
        respond only if it is maths related problem if not the do not give response 
Please identify the subject and type of this JEE problem. Respond ONLY in JSON format with 'subject' and 'type' fields.
Valid subjects: phy (Physics), chem (Chemistry), math (Mathematics)
Valid types: MCQ, MCQ(multiple), Integer, Numeric

Problem:
{state['question']}
"""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        try:
            import json
            classification = json.loads(response.content)
            subject = classification.get("subject")
            question_type = classification.get("type")
            
            logger.info(f"Classified as subject: {subject}, type: {question_type}")
            
            # Update state
            return {
                **state,
                "subject": subject,
                "question_type": question_type
            }
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {
                **state,
                "subject": None,
                "question_type": None
            }
    
    except Exception as e:
        logger.error(f"Error retrieving subject and type: {e}")
        return {
            **state,
            "subject": None,
            "question_type": None
        }

def web_search(state: QuestionState) -> QuestionState:
    """
    Perform web search for relevant information
    
    Args:
        state: Current state
        
    Returns:
        Updated state with search results
    """
    try:
        logger.info("Performing web search")
        
        # Perform web search
        results = search_web(state['question'])
        
        # Format search results
        formatted_results = format_search_results(results)
        
        # Update state
        return {
            **state,
            "search_results": formatted_results
        }
    
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return {
            **state,
            "search_results": "No relevant information found from web search."
        }

def vector_search(state: QuestionState, vector_store: Any) -> QuestionState:
    """
    Perform vector search for similar questions
    
    Args:
        state: Current state
        vector_store: Vector store instance
        
    Returns:
        Updated state with similar questions
    """
    try:
        logger.info("Performing vector search")
        
        # Prepare filter if subject is available
        filter_criteria = {}
        if state.get("subject"):
            filter_criteria["subject"] = state["subject"]
        
        # Perform semantic search
        results = semantic_search(state['question'], vector_store, k=3, filter_criteria=filter_criteria)
        
        # Process similar questions
        similar_questions = process_similar_questions(results)
        
        # Update state
        return {
            **state,
            "similar_questions": similar_questions
        }
    
    except Exception as e:
        logger.error(f"Error performing vector search: {e}")
        return {
            **state,
            "similar_questions": "No similar questions found."
        }

def get_few_shot(state: QuestionState) -> QuestionState:
    """
    Get few-shot example for the question
    
    Args:
        state: Current state
        
    Returns:
        Updated state with few-shot example
    """
    try:
        logger.info("Getting few-shot example")
        
        # Get subject and question type
        subject = state.get("subject")
        question_type = state.get("question_type")
        
        # Skip if subject or question type is not available
        if not subject or not question_type:
            logger.warning("Subject or question type not available for few-shot example")
            return {
                **state,
                "few_shot_example": None
            }
        
        # Get few-shot examples
        examples = load_few_shot_examples()
        
        # Get few-shot example
        example = get_few_shot_example(examples, subject, question_type)
        
        # Update state
        return {
            **state,
            "few_shot_example": example
        }
    
    except Exception as e:
        logger.error(f"Error getting few-shot example: {e}")
        return {
            **state,
            "few_shot_example": None
        }

def generate_answer(state: QuestionState) -> QuestionState:
    """
    Generate answer for the question
    
    Args:
        state: Current state
        
    Returns:
        Updated state with answer
    """
    try:
        logger.info("Generating answer")
        
        # Get Groq LLM
        llm = init_groq_llm()
        
        # Create a structured prompt for scientific/mathematical content
        prompt = f"""
You are an expert mathematics professor whose sole purpose is to answer the student’s question with a clear, step-by-step solution. When given a JEE-level or math problem:
you should only answer to maths related problems,others problems should not be entertained.
Explain the solution in Markdown with LaTeX math expressions using dollar signs,Use **LaTeX** math expressions inside `$...$` for inline math.
Use multiple lines, bold text, and bullet points for clarity.
any other question than maths must not be answered.
1.Read the problem carefully.
2.State which concepts and formulas you will use.
3.Break the solution into numbered steps, showing each calculation in detail.
4.At each step, explain your reasoning and choices.
5.Conclude with the final result in the requested format—nothing more.
6.Do not offer unrelated commentary. Focus exclusively on solving the math problem.

Question: {state['question']}

Additional Context:
- Search Results: {state.get('search_results', '')}
- Similar Questions: {state.get('similar_questions', '')}
- Few-shot Example: {state.get('few_shot_example', '')}

Please provide your solution in a clear, structured format.
"""
        
        # Generate answer
        answer = llm.invoke([{"role": "user", "content": prompt}])
        
        # Update state
        return {
            **state,
            "answer": answer.content
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            **state,
            "answer": "Error generating answer."
        }

def create_workflow(vector_store: Any) -> StateGraph:
    """
    Create the workflow graph
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        StateGraph instance
    """
    # Create graph
    workflow = StateGraph(QuestionState)
    
    # Add nodes
    workflow.add_node("retrieve_subject_and_type", retrieve_subject_and_type)
    workflow.add_node("web_search", web_search)
    workflow.add_node("vector_search", lambda state: vector_search(state, vector_store))
    workflow.add_node("get_few_shot", get_few_shot)
    workflow.add_node("generate_answer", generate_answer)
    
    # Add edges
    workflow.add_edge("retrieve_subject_and_type", "web_search")
    workflow.add_edge("web_search", "vector_search")
    workflow.add_edge("vector_search", "get_few_shot")
    workflow.add_edge("get_few_shot", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve_subject_and_type")
    
    # Compile the workflow
    return workflow.compile()

def run_workflow(question: str, vector_store: Any) -> Dict[str, Any]:
    """
    Run the workflow
    
    Args:
        question: Question to answer
        vector_store: Vector store instance
        
    Returns:
        Dictionary containing the answer and related information
    """
    try:
        # Create and compile workflow
        app = create_workflow(vector_store)
        
        # Create initial state
        initial_state = {
            "question": question,
            "subject": None,
            "question_type": None,
            "search_results": None,
            "similar_questions": None,
            "few_shot_example": None,
            "answer": None
        }
        
        # Run workflow
        logger.info(f"Running workflow for question: {question[:50]}...")
        result = app.invoke(initial_state)
        
        return {
            "question": question,
            "subject": result.get("subject"),
            "question_type": result.get("question_type"),
            "search_results": result.get("search_results"),
            "similar_questions": result.get("similar_questions"),
            "few_shot_example": result.get("few_shot_example"),
            "answer": result.get("answer")
        }
    
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        return {
            "question": question,
            "answer": f"Error running workflow: {str(e)}"
        } 