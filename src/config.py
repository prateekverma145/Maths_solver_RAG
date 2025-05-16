
"""
Configuration settings for the JEE Bench QA system
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # For embeddings

# Model Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Default to llama3
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huggingface-local")  # Default to local HuggingFace embeddings

# Vector DB Configuration
QDRANT_MODE = os.getenv("QDRANT_MODE", "in-memory")  # Options: "in-memory", "local", "cloud"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # For Qdrant Cloud
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jee_bench_data")
QDRANT_PERSIST_DIR = os.getenv("QDRANT_PERSIST_DIR", "qdrant_data")  # Directory for persisting in-memory data

# Chunking Configuration
USE_CHUNKING = os.getenv("USE_CHUNKING", "True")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Application Settings
DATA_PATH = "data/dataset.json"
FEW_SHOT_EXAMPLES_PATH = "data/few_shot_examples.json"
VECTOR_DIMENSION = 768  # HuggingFace all-mpnet-base-v2 embedding dimension

# Subject Categories
SUBJECTS = ["phy", "chem", "math"]

# Question Types
QUESTION_TYPES = ["MCQ", "MCQ(multiple)", "Integer", "Numeric"]

# Prompt Templates
SYSTEM_PROMPT = """You are a highly knowledgeable AI assistant specialized in solving JEE problems. 
Your task is to provide accurate solutions with detailed explanations for the given JEE problem.
Follow these guidelines:
1. Read the problem carefully
2. Identify the relevant concepts and formulas
3. Show all steps of your calculation clearly
4. Explain your reasoning at each step
5. Provide the final answer in the format requested
Be precise, clear, and pedagogical in your explanations."""

# Search settings
TAVILY_SEARCH_DEPTH = "advanced"
MAX_SEARCH_RESULTS = 5

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true" 