import json
import os
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
logger = logging.getLogger(__name__)


# Load environment vars
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise ValueError(f"Missing required env vars: {', '.join(missing)}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
EMBED_SIZE = 1536
ROUTER_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"


# Routing config
available_domains = [
    "emails",
    "webScrapingData",
]

DOMAIN_TO_NAMESPACE = {
    "emails": "emails",
    "webScrapingData": "webScrapingData",
}


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



# Routing (LLM only)
def smart_route_query(query: str) -> Tuple[str, str]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    prompt = f"""
    You are a domain classifier.

    Available domains:
    {json.dumps(available_domains, indent=2)}

    Rules:
    - Return ONLY a JSON array with ONE domain string
    - No explanation, no markdown, just the JSON array

    Question:
    {query}
    """.strip()

    response = openai_client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown formatting."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = (response.choices[0].message.content or "").strip()
    # logger.info(f"Router response: {raw}")

    # Extract JSON, handling potential markdown wrapping
    json_str = extract_json_from_response(raw)
    
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse router response: {raw}")
        raise ValueError(f"Invalid JSON from router: {e}")

    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(f"Invalid router output format: expected list with one element, got {parsed}")

    domain = parsed[0]
    if domain not in available_domains:
        raise ValueError(f"Unknown domain: {domain}. Available: {available_domains}")

    namespace = DOMAIN_TO_NAMESPACE[domain]
    return domain, namespace



# Pinecone search
def search_similar_text(
    query_text: str,
    top_k: int = 3,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:

    ensure_index_exists()
    index = get_index()

    if namespace is None:
        _, namespace = smart_route_query(query_text)

    embedding = create_embedding(query_text)

    res = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    results = []
    for m in res.matches or []:
        results.append({
            "id": m.id,
            "score": float(m.score),
            "payload": m.metadata or {},
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



# Main
def main() -> None:
    try:
        ensure_index_exists()
        while True:
            query = input("Enter your query: ").strip()
            if not query:
                print("Query cannot be empty")
                return

            domain, namespace = smart_route_query(query)
            # logger.info(f"Routed to domain: {domain}, namespace: {namespace}")
            
            results = search_similar_text(query, top_k=3, namespace=namespace)
            # logger.info(f"Found {len(results)} results")
            
            answer = answer_query(query, results)

            print("ANSWER:", answer)
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nCancelled by user, Goodbye!")
    except Exception as e:
        logger.exception("Error")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()