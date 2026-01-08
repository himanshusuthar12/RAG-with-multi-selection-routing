"""
FastAPI RAG System with Multi-Collection Routing
================================================
This module implements a FastAPI REST API for the RAG system that:
1. Routes queries to appropriate Qdrant collections based on domain classification
2. Searches for similar text in vector databases
3. Generates factual answers using OpenAI
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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


# FastAPI App Setup
app = FastAPI(
    title="RAG System API",
    description="Retrieval Augmented Generation API with Multi-Collection Routing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The user's query/question", min_length=1)
    top_k: int = Field(default=3, description="Number of top results to retrieve", ge=1, le=20)
    collection_name: Optional[str] = Field(default=None, description="Optional collection name to search in")

class RouteRequest(BaseModel):
    """Request model for route endpoint"""
    query: str = Field(..., description="The query to route", min_length=1)

class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query_text: str = Field(..., description="The text to search for", min_length=1)
    top_k: int = Field(default=3, description="Number of top results to return", ge=1, le=20)
    collection_name: Optional[str] = Field(default=None, description="Optional collection name")

class SearchResult(BaseModel):
    """Model for search result"""
    id: str
    score: float
    payload: Dict[str, Any]

class RouteResponse(BaseModel):
    """Response model for route endpoint"""
    query: str
    collections: List[str]
    domains: List[str]

class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query_text: str
    collection_name: str
    results: List[SearchResult]
    total_results: int

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    collections: List[str]
    results_count: int
    answer: str
    search_results: Optional[List[SearchResult]] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    openai_connected: bool
    qdrant_connected: bool
    collections_configured: int


# Business Logic Functions
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
            model="gpt-4o-mini",
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
            model="gpt-4o-mini",
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



# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query - POST - Ask a question and get an answer",
            "route": "/route - POST - Route a query to collections",
            "search": "/search - POST - Search for similar text",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check OpenAI connection
        openai_connected = True
        try:
            openai_client.models.list()
        except:
            openai_connected = False
        
        # Check Qdrant connection
        qdrant_connected = True
        try:
            qdrant.get_collections()
        except:
            qdrant_connected = False
        
        return HealthResponse(
            status="healthy" if (openai_connected and qdrant_connected) else "degraded",
            openai_connected=openai_connected,
            qdrant_connected=qdrant_connected,
            collections_configured=len(DOMAIN_TO_COLLECTION)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
async def route_query(request: RouteRequest):
    """
    Route a query to the most relevant collection(s).
    Returns the collections and domains that match the query.
    """
    try:
        collections = smart_route_query(request.query)
        
        # Get domains from collections
        domains = [
            domain for domain, collection in DOMAIN_TO_COLLECTION.items()
            if collection in collections
        ]
        
        return RouteResponse(
            query=request.query,
            collections=collections,
            domains=domains
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error routing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to route query: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_text(request: SearchRequest):
    """
    Search for similar text in Qdrant collections.
    Returns top-k similar results with scores and payloads.
    """
    try:
        results = search_similar_text(
            query_text=request.query_text,
            top_k=request.top_k,
            collection_name=request.collection_name
        )
        
        # Get collection name (either provided or auto-routed)
        collection_name = request.collection_name
        if collection_name is None:
            collections = smart_route_query(request.query_text)
            collection_name = collections[0] if collections else "unknown"
        
        search_results = [SearchResult(**result) for result in results]
        
        return SearchResponse(
            query_text=request.query_text,
            collection_name=collection_name,
            results=search_results,
            total_results=len(search_results)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint: Ask a question and get an answer.
    This endpoint:
    1. Routes the query to appropriate collection(s)
    2. Searches for similar text
    3. Generates a factual answer
    
    Set include_results=True in the request to include search results in response.
    """
    try:
        # Route query
        collections = smart_route_query(request.query)
        
        # Search for similar text
        results = search_similar_text(
            query_text=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name
        )
        
        # Generate answer
        answer = answer_query(request.query, results)
        
        # Format search results
        search_results = [SearchResult(**result) for result in results]
        
        return QueryResponse(
            query=request.query,
            collections=collections,
            results_count=len(results),
            answer=answer,
            search_results=search_results
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )



# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
