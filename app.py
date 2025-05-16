"""
JEE Bench Q&A - Main Application

This application uses Tavily, Qdrant, LangChain LangGraph, and Groq to answer JEE bench questions.
"""
import argparse
import json
import logging
import sys
from typing import Dict, Any, Optional

from config.config import QDRANT_MODE, QDRANT_PERSIST_DIR, USE_CHUNKING, CHUNK_SIZE, CHUNK_OVERLAP
from src.data_loader import load_jee_dataset, get_question_by_index
from src.setup import setup_application
from src.workflow import run_workflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def setup(force_reindex: bool = False) -> Dict[str, Any]:
    """
    Set up the application
    
    Args:
        force_reindex: Force rebuilding the vector index even if it exists
        
    Returns:
        Setup results
    """
    return setup_application(force_reindex)

def answer_question(question: str) -> Dict[str, Any]:
    """
    Answer a question
    
    Args:
        question: Question to answer
        
    Returns:
        Answer and related information
    """
    return run_workflow(question)

def answer_from_dataset(index: int) -> Optional[Dict[str, Any]]:
    """
    Answer a question from the dataset by index
    
    Args:
        index: Question index in the dataset
        
    Returns:
        Answer and related information
    """
    # Load dataset
    dataset = load_jee_dataset()
    
    # Get question
    question_data = get_question_by_index(dataset, index)
    
    if not question_data:
        logger.error(f"Question with index {index} not found")
        return None
    
    # Extract question
    question = question_data.get('question', '')
    
    # Answer question
    result = answer_question(question)
    
    # Add gold answer
    result['gold_answer'] = question_data.get('gold', '')
    
    return result

def print_formatted_setup_results(results: Dict[str, Any]) -> None:
    """
    Print formatted setup results
    
    Args:
        results: Setup results
    """
    print("\n=== Setup Results ===\n")
    
    # Print API key status
    print("API Keys:")
    for key, available in results.get("api_keys", {}).items():
        status = "✓" if available else "✗"
        print(f"  {key.capitalize()}: {status}")
    
    # Print configuration details
    print("\nConfiguration:")
    print(f"  Qdrant Mode: {QDRANT_MODE}")
    if QDRANT_MODE == "in-memory" and QDRANT_PERSIST_DIR:
        print(f"  Persistence Directory: {QDRANT_PERSIST_DIR}")
    print(f"  Text Chunking: {'Enabled' if USE_CHUNKING else 'Disabled'}")
    if USE_CHUNKING:
        print(f"  Chunk Size: {CHUNK_SIZE}")
        print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    
    # Print vector store setup status
    print("\nVector Store Setup:")
    vector_store_setup = results.get("vector_store_setup", False)
    vector_store_result = results.get("vector_store_result", {})
    
    if vector_store_setup:
        print(f"  Status: ✓ Success")
        print(f"  Message: {vector_store_result.get('message', '')}")
        print(f"  Dataset Size: {vector_store_result.get('dataset_size', 'Unknown')} questions")
    else:
        print(f"  Status: ✗ Failed")
        print(f"  Message: {vector_store_result.get('message', 'Unknown error')}")
        
        if "error" in vector_store_result:
            print(f"  Error: {vector_store_result['error']}")
        
        if "help" in vector_store_result:
            print(f"\nHelp: {vector_store_result['help']}")
    
    print("\nSee app.log for more details.\n")

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description='JEE Bench Q&A Application')
    parser.add_argument('--setup', action='store_true', help='Set up the application')
    parser.add_argument('--force-reindex', action='store_true', help='Force rebuilding the vector index even if it exists')
    parser.add_argument('--question', type=str, help='Question to answer')
    parser.add_argument('--index', type=int, help='Answer question from dataset by index')
    parser.add_argument('--output', type=str, help='Output file for answer')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_results = setup(args.force_reindex)
        if args.json:
            print(json.dumps(setup_results, indent=2))
        else:
            print_formatted_setup_results(setup_results)
        return
    
    result = None
    
    if args.question:
        logger.info(f"Answering question: {args.question[:50]}...")
        result = answer_question(args.question)
    
    elif args.index is not None:
        logger.info(f"Answering question from dataset with index: {args.index}")
        result = answer_from_dataset(args.index)
    
    else:
        parser.print_help()
        return
    
    if result:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Answer saved to {args.output}")
        else:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("\n" + "="*80)
                print("QUESTION:")
                print(result.get('question', ''))
                print("\n" + "="*80)
                print("ANSWER:")
                print(result.get('answer', ''))
                
                if 'gold_answer' in result:
                    print("\n" + "="*80)
                    print(f"GOLD ANSWER: {result.get('gold_answer', '')}")

if __name__ == '__main__':
    main()