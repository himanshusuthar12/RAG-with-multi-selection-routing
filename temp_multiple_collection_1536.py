"""
RAG System with Multi-Collection Routing
=========================================
This module implements a Retrieval Augmented Generation (RAG) system that:
1. Routes queries to appropriate Qdrant collections based on domain classification
2. Searches for similar text in vector databases
3. Generates factual answers using OpenAI
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, ScoredPoint


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Load environment vars
load_dotenv()

# Validate required environment variables
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "qdrantUrl", "qdrantApiKey"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

DOMAIN_TO_COLLECTION = json.loads(os.getenv("DOMAIN_TO_COLLECTION", "{}"))
if not DOMAIN_TO_COLLECTION:
    logger.warning("DOMAIN_TO_COLLECTION is empty. Routing may fail.")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("qdrantUrl"),
    api_key=os.getenv("qdrantApiKey")
)


# Route query to collections
def smart_route_query(query: str) -> List[str]:
    """
    Routes a query to the most relevant Qdrant collection(s) based on domain classification.
    
    Args:
        query: The user's query string
        
    Returns:
        List of collection names that match the query domain
        
    Raises:
        ValueError: If routing fails or no valid collections are found
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Get available domains from configuration
    available_domains = list(DOMAIN_TO_COLLECTION.keys())
    if not available_domains:
        raise ValueError("No domains configured in DOMAIN_TO_COLLECTION")
    
    ROUTER_PROMPT = f"""
    You are a domain classifier.

    Available domains:
    - emails
    - webScrapingData
    - OnlineRetailDataSet

    Task:
    Given a question, determine which single domain from the list is most relevant. 

    Instructions:
    - Return ONLY a JSON array containing ONE domain name that best matches the question.
    - Do not include any explanations, text, or extra formatting.
    - Example format: ["emails"]

    Question: {query}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": "You are a domain classifier. Output JSON only. Return a JSON array with one domain name, e.g., [\"emails\"]"},
                {"role": "user", "content": ROUTER_PROMPT}
            ],
            temperature=0
        )
        
        raw = response.choices[0].message.content.strip()
        logger.info(f"Router response: {raw}")
        
        # Try to parse as JSON object first, then extract array if needed
        try:
            parsed = json.loads(raw)
            # Handle both {"domain": "emails"} and ["emails"] formats
            if isinstance(parsed, dict):
                # If it's a dict, try to find a domain key
                domains = [v for k, v in parsed.items() if v in available_domains]
                if not domains:
                    # Try to get the first value
                    domains = list(parsed.values())[:1] if parsed else []
            elif isinstance(parsed, list):
                domains = parsed
            else:
                raise ValueError(f"Unexpected JSON format: {type(parsed)}")
        except json.JSONDecodeError:
            # Fallback: try to extract domain from text
            domains = [d for d in available_domains if d.lower() in raw.lower()]
            if not domains:
                raise ValueError(f"Could not parse router output as JSON: {raw}")
        
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"Invalid router output format: {raw}")
            
    except Exception as e:
        logger.error(f"Error in smart_route_query: {e}")
        raise ValueError(f"Failed to route query: {str(e)}")

    collections = [
        DOMAIN_TO_COLLECTION[d]
        for d in domains
        if d in DOMAIN_TO_COLLECTION
    ]

    if not collections:
        raise ValueError(f"No valid collection mapped for domains: {domains}")

    logger.info(f"Routed query to collections: {collections}")
    return collections



def search_similar_text(query_text: str, top_k: int = 3, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns Top-K similar records from Qdrant with FULL payload.
    
    Args:
        query_text: The query text to search for
        top_k: Number of top results to return (default: 3)
        collection_name: Optional collection name. If not provided, will be routed automatically.
        
    Returns:
        List of dictionaries containing id, score, and payload for each result
        
    Raises:
        ValueError: If collection doesn't exist or search fails
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    try:
        # Create embedding
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        embedding = embedding_response.data[0].embedding
        logger.info(f"Created embedding of size {len(embedding)}")
        
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise ValueError(f"Failed to create embedding: {str(e)}")

    # Route to collection if not provided
    if collection_name is None:
        collections = smart_route_query(query_text)
        if not collections:
            raise ValueError("No collections found for query")
        collection_name = collections[0]
        logger.info(f"Auto-routed to collection: {collection_name}")

    # Check if collection exists
    if not qdrant.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant")

    try:
        # Use the standard search method
        results: List[ScoredPoint] = qdrant.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
            with_payload=True
        )
        
        final_results = []
        for point in results:
            final_results.append({
                "id": str(point.id),
                "score": float(point.score),
                "payload": point.payload or {}
            })
        
        logger.info(f"Found {len(final_results)} results from collection '{collection_name}'")
        return final_results
        
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        raise ValueError(f"Failed to search collection '{collection_name}': {str(e)}")


def answer_query(query: str, qdrant_results: List[Dict[str, Any]]) -> str:
    """
    Produces final factual answer strictly from Qdrant vector DB results.
    
    Args:
        query: The original user query
        qdrant_results: List of search results from Qdrant
        
    Returns:
        A factual answer string based on the retrieved results
        
    Raises:
        ValueError: If query is empty or no results provided
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not qdrant_results:
        return "I couldn't find any relevant information to answer your question."

    safe_results = safe_json(qdrant_results)

    ANSWER_PROMPT = f"""
    You are a factual RAG answer engine. Your task is to answer the question using ONLY the information provided in the search results.
    
    Important rules:
    - Use only the facts from the provided results
    - Do not make up information or hallucinate
    - If the results don't contain enough information, say so
    - Be concise and accurate
    
    Question: {query}

    Search Results:
    {json.dumps(safe_results, indent=2)}

    Answer:
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": "You are a factual RAG answer engine. Do not hallucinate. Use only the provided information."},
                {"role": "user", "content": ANSWER_PROMPT}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Generated answer successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise ValueError(f"Failed to generate answer: {str(e)}")


def safe_json(obj: Any) -> Any:
    """
    Recursively converts objects to JSON-serializable format.
    Handles Qdrant ScoredPoint objects and other complex types.
    
    Args:
        obj: Object to convert to JSON-serializable format
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, list):
        return [safe_json(o) for o in obj]

    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}

    # Qdrant ScoredPoint
    if hasattr(obj, "payload") and hasattr(obj, "score"):
        return {
            "id": str(getattr(obj, "id", "")),
            "score": float(getattr(obj, "score", 0)),
            "payload": safe_json(getattr(obj, "payload", {}))
        }

    # Any other object → string fallback
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)

    return obj


def create_collection_if_not_exists(collection_name: str, vector_size: int) -> None:
    """
    Creates a Qdrant collection if it doesn't already exist.
    
    Args:
        collection_name: Name of the collection to create
        vector_size: Size of the embedding vectors (e.g., 1536 for text-embedding-3-small)
        
    Raises:
        ValueError: If collection_name is empty or vector_size is invalid
    """
    if not collection_name or not collection_name.strip():
        raise ValueError("Collection name cannot be empty")
    
    if vector_size <= 0:
        raise ValueError("Vector size must be greater than 0")
    
    if not qdrant.collection_exists(collection_name):
        try:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise ValueError(f"Failed to create collection: {str(e)}")
    else:
        logger.info(f"Collection '{collection_name}' already exists")





# Main execution
def main() -> None:
    """
    Main function to run the RAG system interactively.
    """
    try:
        query = input("Enter your query: ").strip()
        
        if not query:
            print("Error: Query cannot be empty.")
            return
        
        print(f"\nProcessing query: {query}\n")
        
        # Route query to appropriate collection(s)
        collections = smart_route_query(query)
        print(f"✓ Routed to collection(s): {collections}\n")
        
        # Search for similar text
        results = search_similar_text(query)
        # print(f"✓ Found {len(results)} relevant result(s)\n")
        
        # Generate answer
        answer = answer_query(query, results)
        print("ANSWER:", answer)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
