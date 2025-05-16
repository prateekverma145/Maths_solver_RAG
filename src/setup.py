"""
Setup module for initializing the application
"""
import logging
import os
from typing import Dict, List, Any

from config.config import QDRANT_MODE, QDRANT_PERSIST_DIR, QDRANT_COLLECTION
from config.config import GROQ_API_KEY, TAVILY_API_KEY, HUGGINGFACE_API_KEY, EMBEDDING_MODEL, QDRANT_API_KEY
from src.data_loader import load_jee_dataset
from src.vector_store import index_dataset, collection_exists_with_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_vector_store(force_reindex: bool = False) -> Dict[str, Any]:
    """
    Set up the vector store with the JEE bench dataset
    
    Args:
        force_reindex: Force rebuilding the vector index even if it exists
        
    Returns:
        Dictionary with setup results
    """
    try:
        logger.info("Setting up vector store")
        
        # Prepare persistence directory if needed
        if QDRANT_MODE == "in-memory" and QDRANT_PERSIST_DIR:
            try:
                os.makedirs(QDRANT_PERSIST_DIR, exist_ok=True)
                logger.info(f"Created persistence directory at {QDRANT_PERSIST_DIR}")
            except Exception as e:
                logger.warning(f"Could not create persistence directory: {e}")
        
        # Check if collection already exists with data (unless force_reindex is True)
        if not force_reindex:
            has_existing_data = collection_exists_with_data(QDRANT_COLLECTION)
            if has_existing_data:
                logger.info(f"Collection {QDRANT_COLLECTION} already exists with data, checking if it's usable")
                # Verify that we can actually use the vector store
                try:
                    # Import here to avoid circular imports
                    from src.vector_store import get_vector_store, semantic_search
                    vector_store = get_vector_store()
                    # Try a simple search to verify everything works
                    test_results = semantic_search("Test query", vector_store, k=1)
                    logger.info("Successfully verified existing vector store")
                    
                    # Get dataset size for reporting
                    dataset = load_jee_dataset()
                    return {
                        "success": True,
                        "message": f"Using existing vector store (mode: {QDRANT_MODE})",
                        "dataset_size": len(dataset)
                    }
                except Exception as e:
                    logger.warning(f"Existing vector store found but not usable: {e}")
                    if not force_reindex:
                        logger.info("Forcing reindex due to unusable existing vector store")
                        force_reindex = True
            
        if force_reindex:
            logger.info("Force reindex requested, rebuilding vector store")
        
        # Load dataset
        dataset = load_jee_dataset()
        logger.info(f"Loaded dataset with {len(dataset)} questions")
        
        # Try to index dataset
        try:
            vector_store = index_dataset(dataset)
            logger.info("Vector store setup complete")
            return {
                "success": True,
                "message": f"Vector store setup complete (mode: {QDRANT_MODE})",
                "dataset_size": len(dataset)
            }
        except Exception as e:
            if QDRANT_MODE == "local" and "refused" in str(e).lower():
                logger.error(f"Error connecting to Qdrant: {e}")
                return {
                    "success": False,
                    "message": "Failed to connect to local Qdrant server. Make sure Qdrant is running.",
                    "error": str(e),
                    "help": "See README.md for instructions on installing and running Qdrant."
                }
            elif QDRANT_MODE == "cloud" and ("forbidden" in str(e).lower() or "unauthorized" in str(e).lower()):
                logger.error(f"Error connecting to Qdrant Cloud: {e}")
                return {
                    "success": False,
                    "message": "Failed to connect to Qdrant Cloud. Check your API key and URL.",
                    "error": str(e),
                    "help": "Ensure your QDRANT_URL and QDRANT_API_KEY are set correctly in .env file."
                }
            else:
                logger.error(f"Error indexing dataset: {e}")
                return {
                    "success": False,
                    "message": f"Failed to index dataset in {QDRANT_MODE} mode",
                    "error": str(e)
                }
    
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        return {
            "success": False,
            "message": "Error setting up vector store",
            "error": str(e)
        }

def verify_api_keys() -> Dict[str, bool]:
    """
    Verify that API keys are available
    
    Returns:
        Dictionary with API key verification results
    """
  
    
    results = {
        "groq": GROQ_API_KEY != "",
        "tavily": TAVILY_API_KEY != "",
    }
    
    # Only verify HuggingFace API key if using the API version
    if EMBEDDING_MODEL == "huggingface-api":
        results["huggingface"] = HUGGINGFACE_API_KEY != ""
    
    # Only verify Qdrant API key if using cloud mode
    if QDRANT_MODE == "cloud":
        results["qdrant"] = QDRANT_API_KEY != ""
    
    for key, available in results.items():
        if available:
            logger.info(f"{key.capitalize()} API key is available")
        else:
            logger.warning(f"{key.capitalize()} API key is not available")
    
    return results

def setup_application(force_reindex: bool = False) -> Dict[str, Any]:
    """
    Set up the application
    
    Args:
        force_reindex: Force rebuilding the vector index even if it exists
        
    Returns:
        Dictionary with setup results
    """
    try:
        logger.info("Setting up application")
        
        # Verify API keys
        api_keys = verify_api_keys()
        
        result = {
            "api_keys": api_keys,
            "vector_store_setup": False
        }
        
        # Set up vector store if necessary API keys are available
        required_keys_present = True
        
        # Check if required API keys are present based on configuration
        if QDRANT_MODE == "cloud" and not api_keys.get("qdrant", False):
            logger.warning("Qdrant Cloud API key is missing but cloud mode is selected")
            required_keys_present = False
        
        if EMBEDDING_MODEL == "huggingface-api" and not api_keys.get("huggingface", False):
            logger.warning("HuggingFace API key is missing but API embedding model is selected")
            required_keys_present = False
            
        if not api_keys.get("groq", False) or not api_keys.get("tavily", False):
            logger.warning("Core API keys (Groq or Tavily) are missing")
            required_keys_present = False
        
        if required_keys_present:
            logger.info(f"All required API keys are available for {QDRANT_MODE} mode, setting up vector store")
            vector_store_result = setup_vector_store(force_reindex)
            result["vector_store_result"] = vector_store_result
            result["vector_store_setup"] = vector_store_result.get("success", False)
        else:
            logger.warning("Some required API keys are missing, skipping vector store setup")
            result["vector_store_result"] = {
                "success": False,
                "message": "Missing required API keys for the selected configuration"
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error setting up application: {e}")
        return {
            "api_keys": {},
            "vector_store_setup": False,
            "error": str(e)
        } 