"""
Vector database operations using Qdrant
"""
import traceback
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
# from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config.config import (
    QDRANT_HOST, 
    QDRANT_PORT, 
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION, 
    QDRANT_MODE,
    QDRANT_PERSIST_DIR,
    VECTOR_DIMENSION,
    HUGGINGFACE_API_KEY,
    EMBEDDING_MODEL,
    USE_CHUNKING,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Get the embedding model based on config
    
    Returns:
        Embedding model instance
    """
    if EMBEDDING_MODEL == "huggingface-api":
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACE_API_KEY, 
            model_name="thenlper/gte-large"
        )
    elif EMBEDDING_MODEL == "huggingface-local":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        logger.warning(f"Unsupported embedding model: {EMBEDDING_MODEL}, using HuggingFace embeddings")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def init_qdrant_client() -> QdrantClient:
    """
    Initialize the Qdrant client based on configuration.
    Supports:
      - in-memory with optional local persistence
      - Qdrant Cloud
      - standalone local server
    """
    try:
        if QDRANT_MODE == "in-memory":
            if QDRANT_PERSIST_DIR:
                try:
                    # Try in-memory with persistence first
                    client = QdrantClient(path=QDRANT_PERSIST_DIR)
                    logger.info(f"🗄️  In-memory Qdrant with persistence at {QDRANT_PERSIST_DIR}")
                    return client
                except RuntimeError as e:
                    if "already accessed by another instance" in str(e):
                        # If locked (e.g., by Streamlit hot-reload), fall back to HTTP mode
                        logger.info("Storage folder is locked, falling back to HTTP mode")
                        client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
                        logger.info(f"📡 Connected to local Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
                        return client
                    else:
                        raise
            else:
                client = QdrantClient()  # defaults to pure in-memory
                logger.info("🗄️  Pure in-memory Qdrant (no persistence)")
                return client
        
        elif QDRANT_MODE == "cloud" and QDRANT_URL and QDRANT_API_KEY:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info(f"☁️  Connected to Qdrant Cloud at {QDRANT_URL}")
            return client
        
        else:
            client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
            logger.info(f"📡 Connected to local Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            return client

    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        raise

def create_collection(client: QdrantClient, collection_name: str = QDRANT_COLLECTION):
    """
    Create a new collection in Qdrant if it doesn't exist
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
    """
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        print(f"Collection names: {collection_names}")
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise

def chunk_document(document: Document) -> List[Document]:
    """
    Chunk a document into smaller pieces if chunking is enabled
    
    Args:
        document: Document to chunk
        
    Returns:
        List of chunked documents or original document in a list
    """
    if not USE_CHUNKING:
        return [document]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    docs = text_splitter.split_documents([document])
    logger.info(f"Split document into {len(docs)} chunks")
    return docs

def convert_to_documents(dataset: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert the JEE bench dataset to LangChain Document objects with optional chunking
    
    Args:
        dataset: List of questions with their metadata
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for item in dataset:
        content = f"Question: {item['question']}\nGold Answer: {item['gold']}"
        metadata = {
            "index": item.get("index"),
            "description": item.get("description", ""),
            "subject": item.get("subject", ""),
            "type": item.get("type", ""),
        }
        
        doc = Document(page_content=content, metadata=metadata)
        
        # Apply chunking if enabled
        chunked_docs = chunk_document(doc)
        documents.extend(chunked_docs)
    
    logger.info(f"Created {len(documents)} documents" + 
                f" (with chunking enabled: {USE_CHUNKING})" if USE_CHUNKING else "")
    return documents

def index_dataset(dataset: List[Dict[str, Any]], 
                 collection_name: str = QDRANT_COLLECTION) -> Qdrant:
    """
    Index the dataset into Qdrant
    
    Args:
        dataset: List of questions with their metadata
        collection_name: Name of the collection to use
        
    Returns:
        Qdrant vector store instance
    """
    try:
        # Convert dataset to documents
        documents = convert_to_documents(dataset)
        print("Documents converted to LangChain documents")
        
        # Get embedding model
        embeddings = get_embedding_model()
        print("Embedding model created")
        
        # Test embeddings
        try:
            vectors = embeddings.embed_documents(["This is a test."])
            print(f"Embedding test successful, vector dimension: {len(vectors[0])}")
        except Exception as e:
            logger.error(f"Error testing embeddings: {e}")
            raise
        
        # Special case for in-memory mode to avoid connection issues
        if QDRANT_MODE == "in-memory":
            try:
                # Initialize Qdrant client and create collection
                if QDRANT_PERSIST_DIR:
                    # Create directory if needed
                    os.makedirs(QDRANT_PERSIST_DIR, exist_ok=True)
                    
                    # Create vector store with persistence
                    vector_store = Qdrant.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        collection_name=collection_name,
                        path=QDRANT_PERSIST_DIR
                    )
                    print(f"Vector store created in in-memory mode with persistence at {QDRANT_PERSIST_DIR}")
                else:
                    # Create vector store in memory without persistence
                    vector_store = Qdrant.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        collection_name=collection_name,
                        location=":memory:"
                    )
                    print("Vector store created in in-memory mode without persistence")
                
                logger.info(f"Successfully indexed {len(documents)} documents in collection {collection_name}")
                return vector_store
            except Exception as e:
                logger.error(f"Error creating in-memory vector store: {e}")
                raise
        
        # For other modes, use the standard approach
        # Initialize Qdrant client and create collection
        client = init_qdrant_client()
        print("Qdrant client initialized")
        create_collection(client, collection_name)
        print("Collection created")
        
        # Create vector store based on Qdrant mode
        if QDRANT_MODE == "cloud" and QDRANT_URL and QDRANT_API_KEY:
            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
        else:
            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                url=f"http://{QDRANT_HOST}:{QDRANT_PORT}"
            )
            
        print("Vector store created")
        logger.info(f"Successfully indexed {len(documents)} documents in collection {collection_name}")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error indexing dataset: {e}")
        logger.error(traceback.format_exc())
        raise

def get_vector_store(collection_name: str = QDRANT_COLLECTION) -> Qdrant:
    """
    Get an existing Qdrant vector store instance
    
    Args:
        collection_name: Name of the collection to use
        
    Returns:
        Qdrant vector store instance
    """
    try:
        # Get embedding model
        embeddings = get_embedding_model()
        
        # Log embedding model information for debugging
        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        
        # Test that embeddings work
        try:
            test_embedding = embeddings.embed_query("Test query")
            logger.info(f"Embedding test successful, vector dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Error testing embeddings: {e}")
            raise

        def try_connect_http():
            """Try to connect to Qdrant HTTP server"""
            try:
                client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
                # Test connection
                client.get_collections()
                return client
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant HTTP server: {e}")
                return None
        
        # Get vector store based on Qdrant mode
        if QDRANT_MODE == "in-memory":
            if QDRANT_PERSIST_DIR:
                try:
                    # Try in-memory with persistence first
                    client = QdrantClient(path=QDRANT_PERSIST_DIR)
                    vector_store = Qdrant(
                        client=client,
                        collection_name=collection_name,
                        embeddings=embeddings
                    )
                    logger.info("Successfully connected using in-memory with persistence")
                    return vector_store
                except RuntimeError as e:
                    if "already accessed by another instance" in str(e):
                        # If locked (e.g., by Streamlit hot-reload), try HTTP mode
                        logger.info("Storage folder is locked, trying HTTP mode")
                        http_client = try_connect_http()
                        if http_client:
                            vector_store = Qdrant(
                                client=http_client,
                                collection_name=collection_name,
                                embeddings=embeddings
                            )
                            logger.info("Successfully connected using HTTP mode")
                            return vector_store
                        else:
                            # If HTTP mode fails, try pure in-memory as last resort
                            logger.info("HTTP mode failed, falling back to pure in-memory mode")
                            client = QdrantClient()
                            vector_store = Qdrant(
                                client=client,
                                collection_name=collection_name,
                                embeddings=embeddings
                            )
                            return vector_store
                    else:
                        raise
            else:
                # Pure in-memory mode
                client = QdrantClient()
                vector_store = Qdrant(
                    client=client,
                    collection_name=collection_name,
                    embeddings=embeddings
                )
                return vector_store
        elif QDRANT_MODE == "cloud" and QDRANT_URL and QDRANT_API_KEY:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            vector_store = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
            return vector_store
        else:
            # Try HTTP mode
            http_client = try_connect_http()
            if http_client:
                vector_store = Qdrant(
                    client=http_client,
                    collection_name=collection_name,
                    embeddings=embeddings
                )
                return vector_store
            else:
                # Fall back to in-memory if HTTP fails
                logger.info("HTTP mode failed, falling back to in-memory mode")
                client = QdrantClient()
                vector_store = Qdrant(
                    client=client,
                    collection_name=collection_name,
                    embeddings=embeddings
                )
                return vector_store
    
    except Exception as e:
        logger.error(f"Error getting vector store: {e}")
        logger.error(traceback.format_exc())
        raise

def semantic_search(
    query: str,
    vector_store: Qdrant,
    k: int = 5,
    use_mmr: bool = False,
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.5,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search on the vector store, optionally using MMR.

    Args:
        query: Search query
        vector_store: Qdrant vector store instance
        k: Number of results to return (after reranking/MMR)
        use_mmr: If True, perform MMR; otherwise simple similarity search
        fetch_k: (MMR only) Number of top docs to fetch before MMR. 
                 Must be >= k. Defaults to max(2*k, 20).
        lambda_mult: (MMR only) trade-off parameter in [0,1]; 
                     higher = more diversity, lower = more relevance.
        filter_criteria: Optional filter criteria dict

    Returns:
        List of (Document, score) tuples
    """
    try:
        # build Qdrant filter if requested
        qfilter = None
        if filter_criteria:
            qfilter = {
                "must": [
                    {"key": key, "match": {"value": value}}
                    for key, value in filter_criteria.items()
                ]
            }

        if use_mmr:
            # default fetch_k if not provided
            if fetch_k is None:
                fetch_k = max(2 * k, 20)
            logger.info(f"Running MMR search: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}")
            results = vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=qfilter
            )
            # Convert results to (Document, score) format
            return [(doc, 1.0) for doc in results]  # MMR doesn't provide scores
        else:
            logger.info(f"Running standard search: k={k}")
            results = vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=qfilter
            )
            return results

        logger.info(f"Found {len(results)} results for query: {query}")

    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise
# def collection_exists_with_data(collection_name: str = QDRANT_COLLECTION) -> bool:
#     """
#     Check if a collection exists and contains data
    
#     Args:
#         collection_name: Name of the collection to check
        
#     Returns:
#         True if collection exists and contains data, False otherwise
#     """
def collection_exists_with_data(collection_name: str = QDRANT_COLLECTION) -> bool:
    """
    Check if a collection exists and contains data
    
    Args:
        collection_name: Name of the collection to check
        
    Returns:
        True if collection exists and contains data, False otherwise
    """
    try:
        # Check if the directory exists for in-memory mode with persistence
        if QDRANT_MODE == "in-memory" and QDRANT_PERSIST_DIR:
            collection_path = os.path.join(QDRANT_PERSIST_DIR, "collections", collection_name)
            if os.path.exists(collection_path):
                logger.info(f"Found persistent collection at {collection_path}")
                return True
        
        # For other modes, check using the client
        client = init_qdrant_client()
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Collection {collection_name} does not exist")
            return False
        
        # Check if collection has points
        collection_info = client.get_collection(collection_name=collection_name)
        point_count = collection_info.vectors_count
        
        logger.info(f"Collection {collection_name} contains {point_count} points")
        
        # Return True if collection has points
        return point_count > 0
    
    except Exception as e:
        logger.warning(f"Error checking collection: {e}")
        return False