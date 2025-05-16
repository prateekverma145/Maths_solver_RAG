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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Default to llama3
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
USE_CHUNKING = os.getenv("USE_CHUNKING", "False").lower() == "true"
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
SYSTEM_PROMPT = """You are an expert mathematics professor whose sole purpose is to answer the student’s question with a clear, step-by-step solution. When given a JEE-level or math problem:
you should only answer to maths related problems.
Explain the solution in Markdown with LaTeX math expressions using dollar signs,Use **LaTeX** math expressions inside `$...$` for inline math.
Use multiple lines, bold text, and bullet points for clarity.
any other question than maths must not be answered.
1.Read the problem carefully.
2.State which concepts and formulas you will use.
3.Break the solution into numbered steps, showing each calculation in detail.
4.At each step, explain your reasoning and choices.
5.Conclude with the final result in the requested format—nothing more.
6.Do not offer unrelated commentary. Focus exclusively on solving the math problem.
"""

# Search settings
TAVILY_SEARCH_DEPTH = "advanced"

MAX_SEARCH_RESULTS = 5

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true" 