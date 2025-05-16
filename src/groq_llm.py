"""
Groq LLM module for question answering
"""
import logging
from typing import Dict, List, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config.config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_groq_llm(temperature: float = 0.2, max_tokens: int = 2048) -> ChatGroq:
    """
    Initialize the Groq LLM
    
    Args:
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        
    Returns:
        ChatGroq instance
    """
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Successfully initialized Groq LLM with model: {GROQ_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Error initializing Groq LLM: {e}")
        raise

def create_few_shot_prompt(question: str, 
                          search_results: str, 
                          similar_questions: str,
                          few_shot_example: Dict[str, str]) -> List[SystemMessage | HumanMessage]:
    """
    Create a prompt for the Groq LLM with few-shot examples
    
    Args:
        question: Question to answer
        search_results: Search results from web search
        similar_questions: Similar questions from vector search
        few_shot_example: Few-shot example with problem and solution
        
    Returns:
        List of chat messages
    """
    # System prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Few-shot example
    if few_shot_example and few_shot_example.get("problem") and few_shot_example.get("solution"):
        example_text = f"""
Here is an example of how to solve a similar problem:

PROBLEM:
{few_shot_example['problem']}

SOLUTION:
{few_shot_example['solution']}

Now, solve the current problem using a similar approach:
"""
    else:
        example_text = ""
    
    # Current problem with search results and similar questions
    current_problem = f"""
{example_text}

CURRENT PROBLEM:
{question}

{search_results}

{similar_questions}

Please provide a detailed solution step by step, showing all your work and reasoning. End your solution with the final answer clearly marked.
"""
    
    messages.append(HumanMessage(content=current_problem))
    
    return messages

def answer_question(llm: ChatGroq, 
                   question: str, 
                   search_results: str = "", 
                   similar_questions: str = "",
                   few_shot_example: Optional[Dict[str, str]] = None) -> str:
    """
    Answer a question using the Groq LLM
    
    Args:
        llm: ChatGroq instance
        question: Question to answer
        search_results: Search results from web search
        similar_questions: Similar questions from vector search
        few_shot_example: Few-shot example with problem and solution
        
    Returns:
        Answer from the LLM
    """
    try:
        # Create prompt
        messages = create_few_shot_prompt(
            question=question,
            search_results=search_results,
            similar_questions=similar_questions,
            few_shot_example=few_shot_example if few_shot_example else {}
        )
        
        # Get answer
        response = llm.invoke(messages)
        answer = response.content
        
        logger.info(f"Successfully generated answer with {len(answer)} characters")
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}" 