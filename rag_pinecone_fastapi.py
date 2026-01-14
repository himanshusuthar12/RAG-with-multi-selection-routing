import json
import os
import logging
import re
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import pyodbc for SQL Server support
try:
    import pyodbc
    SQL_SERVER_AVAILABLE = True
except ImportError:
    SQL_SERVER_AVAILABLE = False
    logger.warning("pyodbc not installed. SQL Server support unavailable. Install with: pip install pyodbc")

# Load environment vars
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.json")

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()  # "sqlite" or "sqlserver"
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")  # For SQLite: path to .db file, For SQL Server: connection string
CONFIG_TABLE_NAME = os.getenv("CONFIG_TABLE_NAME", "aiChatbot001")

# Initialize config variables (will be loaded from database)
available_domains = None
DOMAIN_TO_NAMESPACE = {}
column_names = {}
EMBED_MODEL = None
EMBED_SIZE = None
ROUTER_MODEL = None


# Validate required env vars
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "DB_CONNECTION_STRING"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise ValueError(f"Missing required env vars: {', '.join(missing)}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with Pinecone",
    description="RAG API with multi-domain routing and database configuration",
    version="1.0.0"
)


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query/question")
    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of search results to return")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    domain: str = Field(..., description="Selected domain")
    namespace: str = Field(..., description="Pinecone namespace used")
    refined_query: Optional[str] = Field(None, description="Refined query after context consideration")
    search_results_count: int = Field(..., description="Number of search results found")


class HistoryResponse(BaseModel):
    user_id: str
    history: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    config_loaded: bool
    index_exists: bool
    available_domains: List[str]


class ConfigResponse(BaseModel):
    available_domains: List[str]
    domain_to_namespace: Dict[str, str]
    embed_model: str
    embed_size: int
    router_model: str
    ROUTER_MODEL: str


# Database functions
def get_db_connection():
    """Get database connection based on DB_TYPE."""
    if DB_TYPE == "sqlite":
        if not DB_CONNECTION_STRING:
            raise ValueError("DB_CONNECTION_STRING must be set for SQLite (path to .db file)")
        return sqlite3.connect(DB_CONNECTION_STRING)
    elif DB_TYPE == "sqlserver":
        if not SQL_SERVER_AVAILABLE:
            raise ImportError("pyodbc is required for SQL Server. Install with: pip install pyodbc")
        if not DB_CONNECTION_STRING:
            raise ValueError("DB_CONNECTION_STRING must be set for SQL Server (connection string)")
        return pyodbc.connect(DB_CONNECTION_STRING)
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}. Use 'sqlite' or 'sqlserver'")


def load_config_from_database() -> Tuple[List[str], Dict[str, str], Dict[str, List[str]], str, int, str]:
    """
    Load configuration from database table aiChatbot001.
    Returns: (available_domains, DOMAIN_TO_NAMESPACE, column_names, EMBED_MODEL, EMBED_SIZE, ROUTER_MODEL)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute query to get config
        query = f"SELECT available_domains, DOMAIN_TO_NAMESPACE, column_names, EMBED_MODEL, EMBED_SIZE, ROUTER_MODEL FROM {CONFIG_TABLE_NAME}"
        cursor.execute(query)
        
        row = cursor.fetchone()
        
        # Check if there are multiple rows (warn but use first one)
        remaining_rows = cursor.fetchall()
        if remaining_rows:
            logger.warning(f"Multiple rows found in '{CONFIG_TABLE_NAME}'. Using the first row. Consider filtering by a specific condition.")
        
        cursor.close()
        conn.close()
        
        if not row:
            raise ValueError(f"No configuration found in table '{CONFIG_TABLE_NAME}'. Please insert configuration data.")
        
        # Parse the row data
        available_domains_str = row[0] if row[0] else "[]"
        domain_to_namespace_str = row[1] if row[1] else "{}"
        column_names_str = row[2] if row[2] else "{}"
        embed_model = row[3] if row[3] else "text-embedding-3-small"  # Default fallback
        embed_size = row[4] if row[4] is not None else 1536  # Default fallback
        router_model = row[5] if row[5] else "gpt-4o-mini"  # Default fallback
        
        # Parse JSON strings
        try:
            available_domains = json.loads(available_domains_str) if isinstance(available_domains_str, str) else available_domains_str
            DOMAIN_TO_NAMESPACE = json.loads(domain_to_namespace_str) if isinstance(domain_to_namespace_str, str) else domain_to_namespace_str
            column_names = json.loads(column_names_str) if isinstance(column_names_str, str) else column_names_str
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from database: {e}")
        
        # Ensure available_domains is a list
        if not isinstance(available_domains, list):
            raise ValueError(f"available_domains must be a JSON array, got: {type(available_domains)}")
        
        # Ensure DOMAIN_TO_NAMESPACE is a dict
        if not isinstance(DOMAIN_TO_NAMESPACE, dict):
            raise ValueError(f"DOMAIN_TO_NAMESPACE must be a JSON object, got: {type(DOMAIN_TO_NAMESPACE)}")
        
        # Ensure column_names is a dict
        if not isinstance(column_names, dict):
            raise ValueError(f"column_names must be a JSON object, got: {type(column_names)}")
        
        # Ensure EMBED_SIZE is an integer
        try:
            embed_size = int(embed_size)
        except (ValueError, TypeError):
            logger.warning(f"Invalid EMBED_SIZE value: {embed_size}. Using default: 1536")
            embed_size = 1536
        
        # Ensure EMBED_MODEL and ROUTER_MODEL are strings
        if not isinstance(embed_model, str):
            embed_model = str(embed_model) if embed_model else "text-embedding-3-small"
        if not isinstance(router_model, str):
            router_model = str(router_model) if router_model else "gpt-4o-mini"
        
        logger.info(f"Successfully loaded config from database: {len(available_domains)} domains, EMBED_MODEL={embed_model}, EMBED_SIZE={embed_size}, ROUTER_MODEL={router_model}")
        return available_domains, DOMAIN_TO_NAMESPACE, column_names, embed_model, embed_size, router_model
        
    except Exception as e:
        logger.error(f"Failed to load config from database: {e}")
        raise


# Load configuration from database at startup
@app.on_event("startup")
async def startup_event():
    """Load configuration from database when the app starts."""
    global available_domains, DOMAIN_TO_NAMESPACE, column_names, EMBED_MODEL, EMBED_SIZE, ROUTER_MODEL
    try:
        available_domains, DOMAIN_TO_NAMESPACE, column_names, EMBED_MODEL, EMBED_SIZE, ROUTER_MODEL = load_config_from_database()
        logger.info("Configuration loaded successfully from database")
    except Exception as e:
        logger.error(f"Failed to load configuration from database: {e}")
        raise


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
def search_similar_text(query_text: str, top_k: int = 3, namespace: Optional[str] = None, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_index_exists()
    index = get_index()

    if namespace is None:
        _, namespace = smart_route_query(query_text, [])

    # Get column names for this namespace, with fallback to all metadata
    column_name = column_names.get(domain, None)
    if column_name is None:
        logger.warning(f"No column names configured for namespace '{domain}', returning all metadata")

    embedding = create_embedding(query_text)

    res = index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace=namespace)

    results = []
    for m in res.matches or []:
        if column_name:
            # Filter payload to only include specified columns
            filtered_payload = {key: m.metadata.get(key) for key in column_name if m.metadata and key in m.metadata}
        else:
            # Return all metadata if no column names specified
            filtered_payload = m.metadata or {}

        results.append({
            "score": float(m.score),
            "payload": filtered_payload
        })

    return results


# Answer generation
def answer_query(query: str, search_results: List[Dict[str, Any]]) -> str:
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
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": "Use only provided data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return (response.choices[0].message.content or "").strip()


# Query refinement
def refine_query(query: str, history: List[Dict[str, Any]]) -> str:
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
        model=ROUTER_MODEL,
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
        logger.warning(f"Error loading history: {e}. Resetting history file.")
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []


def save_to_history(user_id, query, answer, refined_query=None, namespace=None):
    history = []

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({
        "user_id": user_id,
        "namespace": namespace,
        "query": query.strip(),
        "refined_query": refined_query,
        "answer": answer.strip(),
        "Date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API with Pinecone",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query (POST)",
            "history": "/history/{user_id} (GET)",
            "health": "/health (GET)",
            "config": "/config (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        index_exists = False
        try:
            existing_indexes = pc.list_indexes()
            existing_names = {idx.name for idx in existing_indexes}
            index_exists = PINECONE_INDEX_NAME in existing_names
        except Exception as e:
            logger.warning(f"Error checking index: {e}")
        
        return HealthResponse(
            status="healthy",
            config_loaded=available_domains is not None,
            index_exists=index_exists,
            available_domains=available_domains or []
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/config", response_model=ConfigResponse, tags=["General"])
async def get_config():
    """Get current configuration."""
    if available_domains is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    return ConfigResponse(
        available_domains=available_domains,
        domain_to_namespace=DOMAIN_TO_NAMESPACE,
        embed_model=EMBED_MODEL,
        embed_size=EMBED_SIZE,
        router_model=ROUTER_MODEL,
        ROUTER_MODEL=ROUTER_MODEL
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Main query endpoint that processes user queries using RAG.
    
    - **query**: User's question/query
    - **user_id**: User identifier for history tracking
    - **top_k**: Number of search results to return (1-10, default: 3)
    """
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        if not request.user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID cannot be empty"
            )
        
        # Load user history
        history = load_history(request.user_id)
        
        # Use last 3 history items, or all if fewer than 3
        recent_history = history[-3:] if len(history) >= 3 else history
        
        # Route query to appropriate domain
        domain, namespace = smart_route_query(request.query, recent_history)
        logger.info(f"Routed to domain: {domain}, namespace: {namespace}")
        
        # Refine query based on history
        refined_query = refine_query(request.query, recent_history)
        logger.info(f"Refined query: {refined_query}")
        
        # Search for similar content
        results = search_similar_text(refined_query, top_k=request.top_k, namespace=namespace, domain=domain)
        logger.info(f"Found {len(results)} results")
        
        # Generate answer
        answer = answer_query(request.query, results)
        
        # Save to history
        save_to_history(request.user_id, request.query, answer, refined_query, namespace)
        
        return QueryResponse(
            answer=answer,
            domain=domain,
            namespace=namespace,
            refined_query=refined_query,
            search_results_count=len(results)
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/history/{user_id}", response_model=HistoryResponse, tags=["RAG"])
async def get_history(user_id: str):
    """Get chat history for a specific user."""
    try:
        history = load_history(user_id)
        
        # Get full history entries (not just query/answer pairs)
        full_history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                
                for item in raw:
                    if str(item.get("user_id")) == str(user_id):
                        full_history.append(item)
            except Exception as e:
                logger.warning(f"Error loading full history: {e}")
        
        return HistoryResponse(
            user_id=user_id,
            history=full_history
        )
    except Exception as e:
        logger.exception("Error loading history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
