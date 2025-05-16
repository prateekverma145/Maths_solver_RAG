"""
Data loader module for JEE bench dataset
"""
import json
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
from config.config import DATA_PATH, FEW_SHOT_EXAMPLES_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jee_dataset(data_path: str = DATA_PATH) -> List[Dict[str, Any]]:
    """
    Load the JEE bench dataset from the JSON file
    
    Args:
        data_path: Path to the dataset.json file
        
    Returns:
        List of questions with their metadata
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Successfully loaded dataset with {len(dataset)} questions")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def load_few_shot_examples(examples_path: str = FEW_SHOT_EXAMPLES_PATH) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load the few-shot examples from the JSON file
    
    Args:
        examples_path: Path to the few_shot_examples.json file
        
    Returns:
        Dictionary of few-shot examples organized by subject and question type
    """
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        logger.info(f"Successfully loaded few-shot examples")
        return examples
    except Exception as e:
        logger.error(f"Error loading few-shot examples: {e}")
        raise

def create_dataframe(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the dataset list to a pandas DataFrame for easier manipulation
    
    Args:
        dataset: List of questions with their metadata
        
    Returns:
        Pandas DataFrame containing the dataset
    """
    df = pd.DataFrame(dataset)
    return df

def filter_questions(df: pd.DataFrame, 
                    subject: Optional[str] = None, 
                    question_type: Optional[str] = None) -> pd.DataFrame:
    """
    Filter questions by subject and/or question type
    
    Args:
        df: DataFrame containing the JEE bench data
        subject: Subject filter (phy, chem, math)
        question_type: Question type filter (MCQ, MCQ(multiple), Integer, Numeric)
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if subject:
        filtered_df = filtered_df[filtered_df['subject'] == subject]
    
    if question_type:
        filtered_df = filtered_df[filtered_df['type'] == question_type]
    
    return filtered_df

def get_question_by_index(dataset: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific question by its index
    
    Args:
        dataset: List of questions
        index: Question index to retrieve
        
    Returns:
        Question dictionary if found, None otherwise
    """
    for question in dataset:
        if question.get('index') == index:
            return question
    return None

def get_few_shot_example(examples: Dict[str, Dict[str, Dict[str, str]]], 
                        subject: str, 
                        question_type: str) -> Dict[str, str]:
    """
    Get a few-shot example for a specific subject and question type
    
    Args:
        examples: Few-shot examples dictionary
        subject: Subject (phy, chem, math)
        question_type: Question type (MCQ, MCQ(multiple), Integer, Numeric)
        
    Returns:
        Few-shot example with problem and solution
    """
    try:
        return examples[subject][question_type]
    except KeyError:
        logger.warning(f"No few-shot example found for {subject}/{question_type}")
        return {"problem": "", "solution": ""} 