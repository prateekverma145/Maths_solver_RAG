"""
Example script demonstrating how to use HuggingFace embeddings
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import get_embedding_model
from config.config import EMBEDDING_MODEL, HUGGINGFACE_API_KEY
import numpy as np

def main():
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    
    # Get the embedding model
    embedding_model = get_embedding_model()
    
    # Test with a sample text
    text = "What is Planck's constant in quantum mechanics?"
    
    print(f"Generating embeddings for: '{text}'")
    
    # Generate embeddings
    embedding = embedding_model.embed_query(text)
    
    # Print info about the embeddings
    print(f"\nEmbedding dimension: {len(embedding)}")
    print(f"Embedding type: {type(embedding)}")
    print(f"Sample of embedding vector: {embedding[:5]}...")
    
    # Create another embedding to compare
    another_text = "Calculate the value of Planck's constant using photoelectric effect."
    another_embedding = embedding_model.embed_query(another_text)
    
    # Calculate cosine similarity
    dot_product = np.dot(embedding, another_embedding)
    norm_1 = np.linalg.norm(embedding)
    norm_2 = np.linalg.norm(another_embedding)
    similarity = dot_product / (norm_1 * norm_2)
    
    print(f"\nCosine similarity between the two texts: {similarity:.4f}")
    
    # Create a dissimilar text for comparison
    dissimilar_text = "The economic impact of climate change on global agriculture."
    dissimilar_embedding = embedding_model.embed_query(dissimilar_text)
    
    # Calculate cosine similarity with dissimilar text
    dot_product = np.dot(embedding, dissimilar_embedding)
    norm_3 = np.linalg.norm(dissimilar_embedding)
    dissimilar_similarity = dot_product / (norm_1 * norm_3)
    
    print(f"Cosine similarity with dissimilar text: {dissimilar_similarity:.4f}")
    
    print("\nThis demonstrates how semantic similarity is captured by the embeddings.")
    print("Similar queries have higher cosine similarity scores.")
    
    # Print info about the embedding model
    print("\nEmbedding model details:")
    print(f"Model class: {embedding_model.__class__.__name__}")
    
    if EMBEDDING_MODEL == "huggingface-api":
        print(f"Using HuggingFace Inference API with API key: {HUGGINGFACE_API_KEY[:5]}...")
    else:
        print("Using local HuggingFace embeddings (no API key required)")
        print("Default model: sentence-transformers/all-mpnet-base-v2")

if __name__ == "__main__":
    main() 