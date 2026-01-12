import json
import os
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Load environment vars
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
HISTORY_FILE = os.getenv("HISTORY_FILE")

# Parse JSON env vars with error handling
column_names = {}
DOMAIN_TO_NAMESPACE = {}
try:
    column_names_str = os.getenv("column_names", "{}")
    if column_names_str:
        column_names = json.loads(column_names_str)
except json.JSONDecodeError:
    logger.warning("Failed to parse column_names, using empty dict")

try:
    domain_to_namespace_str = os.getenv("DOMAIN_TO_NAMESPACE", "{}")
    if domain_to_namespace_str:
        DOMAIN_TO_NAMESPACE = json.loads(domain_to_namespace_str)
except json.JSONDecodeError:
    logger.warning("Failed to parse DOMAIN_TO_NAMESPACE, using empty dict")

# Constants
EMBED_MODEL = "text-embedding-3-small"
EMBED_SIZE = 1536
ROUTER_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"

# Validate required env vars
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise ValueError(f"Missing required env vars: {', '.join(missing)}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)

# Routing config
available_domains = [
    "emails",
    "webScrapingData",
]

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with Pinecone",
    description="Retrieval Augmented Generation API with multi-domain routing",
    version="1.0.0"
)


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query/question", min_length=1)
    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of search results to retrieve")
    use_history: bool = Field(default=True, description="Whether to use conversation history")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=3, ge=1, le=20, description="Number of search results")
    namespace: Optional[str] = Field(default=None, description="Pinecone namespace (optional, will be auto-routed if not provided)")


class QueryResponse(BaseModel):
    answer: str
    domain: str
    namespace: str
    refined_query: Optional[str] = None
    search_results_count: int
    search_results: List[Dict[str, Any]] = []


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    namespace: str
    query: str


class HistoryResponse(BaseModel):
    user_id: str
    history: List[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    index_exists: bool
    available_domains: List[str]


# Utilities
def safe_json(obj: Any) -> Any:
    if isinstance(obj, list):
        return [safe_json(o) for o in obj]
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def extract_json_from_response(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    
    # Try to extract from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find JSON array or object directly
    json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if json_match:
        return json_match.group(1).strip()
    
    return text


def create_embedding(text: str) -> List[float]:
    if not text.strip():
        raise ValueError("Text cannot be empty")
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def get_index():
    return pc.Index(PINECONE_INDEX_NAME)


# Index management
def ensure_index_exists() -> None:
    # Pinecone v3+: list_indexes() returns IndexModel objects with .name attribute
    existing_indexes = pc.list_indexes()
    existing_names = {idx.name for idx in existing_indexes}
    
    if PINECONE_INDEX_NAME in existing_names:
        return

    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_SIZE,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    # Wait for index to be ready
    for _ in range(30):
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        # Pinecone v3+: status is an object with .ready attribute
        if index_info.status.ready:
            return
        time.sleep(1)
    
    raise TimeoutError(f"Index '{PINECONE_INDEX_NAME}' did not become ready in time")


def smart_route_query(query: str, History: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    prompt = f"""
    You are a domain classifier.
    Available History:
    {json.dumps(History, indent=2)}

    Available domains:
    {json.dumps(available_domains, indent=2)}

    Rules:
    - Return ONLY a JSON array with ONE domain string
    - You MUST select exactly one domain from the available domains
    - If unsure, pick the most likely domain based on the question
    - No explanation, no markdown, just the JSON array

    Question:
    {query}
    """.strip()

    response = openai_client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown formatting. Always return exactly one domain."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = (response.choices[0].message.content or "").strip()

    json_str = extract_json_from_response(raw)
    
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse router response: {raw}")
        # Fallback to default domain
        logger.warning(f"Using default domain: {available_domains[0]}")
        domain = available_domains[0]
        namespace = DOMAIN_TO_NAMESPACE.get(domain, domain)
        return domain, namespace

    # Handle empty array or invalid format - use default
    if not isinstance(parsed, list) or len(parsed) != 1:
        logger.warning(f"Invalid router output: {parsed}. Using default domain: {available_domains[0]}")
        domain = available_domains[0]
        namespace = DOMAIN_TO_NAMESPACE.get(domain, domain)
        return domain, namespace

    domain = parsed[0]
    
    # Handle unknown domain - use default
    if domain not in available_domains:
        logger.warning(f"Unknown domain: {domain}. Using default domain: {available_domains[0]}")
        domain = available_domains[0]

    namespace = DOMAIN_TO_NAMESPACE.get(domain, domain)
    return domain, namespace


# Pinecone search
def search_similar_text(
    query_text: str,
    top_k: int = 3,
    namespace: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:

    ensure_index_exists()
    index = get_index()

    if namespace is None:
        _, namespace = smart_route_query(query_text, history or [])

    # Get column names for this namespace, with fallback to all metadata
    column_name = column_names.get(namespace, None)
    if column_name is None:
        logger.warning(f"No column names configured for namespace '{namespace}', returning all metadata")

    embedding = create_embedding(query_text)

    res = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    results = []
    for m in res.matches or []:
        if column_name:
            # Filter payload to only include specified columns
            filtered_payload = {key: m.metadata.get(key) for key in column_name if m.metadata and key in m.metadata}
        else:
            # Return all metadata if no column names specified
            filtered_payload = m.metadata or {}

        results.append({
            # "id": m.id,
            "score": float(m.score),
            "payload": filtered_payload
        })

    return results


# Answer generation
def answer_query(
    query: str,
    search_results: List[Dict[str, Any]],
) -> str:

    if not search_results:
        return "I couldn't find relevant information to answer your question."

    prompt = f"""
    Answer the question using ONLY the search results.

    Question:
    {query}

    Search Results (JSON):
    {json.dumps(safe_json(search_results), indent=2)}

    Answer:
    """.strip()

    response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": "Use only provided data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return (response.choices[0].message.content or "").strip()

# Query refinement
def refine_query(
    query: str,
    history: List[Dict[str, Any]],
) -> str:

    if not history:
        return query

    prompt = f"""
    Refine the query based on the conversation history.

    Query:
    {query}

    History:
    {json.dumps(history, indent=2)}

    Refined Query:
    """.strip()

    response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": "Use only provided data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return (response.choices[0].message.content or "").strip()


def load_history(user_id: str) -> List[Dict[str, str]]:
    """
    Loads only clean (query, answer) pairs from history file.
    Ignores all other fields if present.
    """
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        user_history = []

        for item in raw:

            if str(item.get("user_id")) == str(user_id):
                query = item.get("query") or item.get("question") or item.get("user")
                answer = item.get("answer") or item.get("response") or item.get("assistant")

                if query and answer:
                    user_history.append({
                        "query": query.strip(),
                        "answer": answer.strip()
                    })

        return user_history

    except Exception as e:
        logger.error(f"Error loading history: {e}")
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
            logger.info("Corrupted history file reset.")
        return []


def save_to_history(user_id, query, answer, refined_query=None):
    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error reading history file: {e}")
            history = []

    history.append({
        "user_id": user_id,
        "query": query.strip(),
        "refined_query": refined_query,
        "answer": answer.strip(),
        "Date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving history: {e}")
        raise


# FastAPI Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone index on startup"""
    try:
        ensure_index_exists()
        logger.info("FastAPI application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {"message": "RAG API with Pinecone", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        existing_indexes = pc.list_indexes()
        existing_names = {idx.name for idx in existing_indexes}
        index_exists = PINECONE_INDEX_NAME in existing_names
        
        return HealthResponse(
            status="healthy",
            index_exists=index_exists,
            available_domains=available_domains
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint that handles the full RAG pipeline:
    1. Routes query to appropriate domain/namespace
    2. Refines query based on history (if enabled)
    3. Searches Pinecone for similar content
    4. Generates answer using search results
    5. Saves to history
    """
    try:
        # Load history if enabled
        history = []
        if request.use_history:
            history = load_history(request.user_id)
            # Use last 3 history items, or all if fewer than 3
            history = history[-3:] if len(history) >= 3 else history

        # Route query to domain/namespace
        domain, namespace = smart_route_query(request.query, history)
        logger.info(f"Routed to domain: {domain}, namespace: {namespace}")

        # Refine query based on history
        refined_query = None
        if request.use_history and history:
            refined_query = refine_query(request.query, history)
            logger.info(f"Refined query: {refined_query}")
        else:
            refined_query = request.query

        # Search for similar content
        results = search_similar_text(
            refined_query, 
            top_k=request.top_k, 
            namespace=namespace,
            history=history
        )
        logger.info(f"Found {len(results)} results")

        # Generate answer
        answer = answer_query(request.query, results)

        # Save to history
        save_to_history(request.user_id, request.query, answer, refined_query)

        return QueryResponse(
            answer=answer,
            domain=domain,
            namespace=namespace,
            refined_query=refined_query,
            search_results_count=len(results),
            search_results=results
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_endpoint(request: SearchRequest):
    """
    Search endpoint for retrieving similar content from Pinecone.
    Returns search results without generating an answer.
    """
    try:
        # Determine namespace
        if request.namespace:
            namespace = request.namespace
        else:
            _, namespace = smart_route_query(request.query, [])

        # Search
        results = search_similar_text(
            request.query,
            top_k=request.top_k,
            namespace=namespace
        )

        return SearchResponse(
            results=results,
            namespace=namespace,
            query=request.query
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during search")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/history/{user_id}", response_model=HistoryResponse, tags=["History"])
async def get_history(user_id: str):
    """
    Get conversation history for a specific user.
    """
    try:
        history = load_history(user_id)
        return HistoryResponse(
            user_id=user_id,
            history=history
        )
    except Exception as e:
        logger.exception("Error loading history")
        raise HTTPException(status_code=500, detail=f"Error loading history: {str(e)}")


@app.delete("/history/{user_id}", tags=["History"])
async def clear_history(user_id: str):
    """
    Clear conversation history for a specific user.
    Note: This removes all entries for the user from the history file.
    """
    try:
        if not os.path.exists(HISTORY_FILE):
            return {"message": "No history file found", "deleted_count": 0}

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Filter out entries for this user
        original_count = len(history)
        history = [item for item in history if str(item.get("user_id")) != str(user_id)]
        deleted_count = original_count - len(history)

        # Save updated history
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return {
            "message": f"History cleared for user {user_id}",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.exception("Error clearing history")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
