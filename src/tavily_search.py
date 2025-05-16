"""
Web search using Tavily API
"""
import logging
from typing import List, Dict, Any

from langchain_tavily import TavilySearch
from config.config import TAVILY_API_KEY, TAVILY_SEARCH_DEPTH, MAX_SEARCH_RESULTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_web(query: str, 
              search_depth: str = TAVILY_SEARCH_DEPTH, 
              max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    """
    Search the web for information related to a query using Tavily API
    
    Args:
        query: Search query
        search_depth: Search depth (basic or advanced)
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, content, and URL
    """
    try:
        # Augment query with educational context
        augmented_query = f" {query}"
        
        # Initialize Tavily search
        search = TavilySearch(
            api_key=TAVILY_API_KEY,
            search_depth=search_depth,
            k=max_results,
            include_raw_content=True,
            include_domains=["brilliant.org", "khanacademy.org", "byjus.com", 
                           "vedantu.com", "unacademy.com", "physicswallah.com",
                           "wikipedia.org", "youtube.com", "nptel.ac.in",
                           "toppr.com", "ncert.nic.in"]
        )
        
        # Execute search
        results = search.invoke(augmented_query)
        
        logger.info(f"Found {len(results)} search results for query: {augmented_query}")
        return results
    
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        # Return empty list on error
        return []

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for inclusion in the prompt
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string of search results
    """
    if not results:
        return "No relevant information found from web search."
    
    formatted_text = "### Relevant Information from Web Search:\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        content = result.get("content", "").strip()
        url = result.get("url", "")
        
        formatted_text += f"**{i}. {title}**\n"
        formatted_text += f"{content}\n"
        formatted_text += f"Source: {url}\n\n"
    
    return formatted_text 