import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer



# Load environment vars
load_dotenv()

DOMAIN_TO_COLLECTION = json.loads(os.getenv("DOMAIN_TO_COLLECTION", "{}"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("qdrantUrl"),
    api_key=os.getenv("qdrantApiKey")
)
e5_model = SentenceTransformer("intfloat/e5-large-v2")


# Route query to collections
def smart_route_query(query: str) -> str:
    """
    Routes a query to the most appropriate domain/collection using OpenAI.
    
    Args:
        query: The user's query string
        
    Returns:
        The domain name that best matches the query
        
    Raises:
        ValueError: If the router output is invalid or domain not found
        Exception: If OpenAI API call fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Get available domains from environment configuration
    available_domains = list(DOMAIN_TO_COLLECTION.keys())
    
    if not available_domains:
        raise ValueError("No domains configured in DOMAIN_TO_COLLECTION")
    
    ROUTER_PROMPT = f"""
    Choose the best matching domain from the following list:

    Valid domains:
    - webScrapingData
    - OnlineRetailDataSet

    Return ONLY a JSON array with ONE domain name.
    Do not include any explanations, text, or extra formatting.

    Question: {query}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed: correct model name
            messages=[
                {"role": "system", "content": "You are a domain classifier. Return JSON only."},
                {"role": "user", "content": ROUTER_PROMPT}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        raw = response.choices[0].message.content.strip()
        # print("Router raw:", raw)
        
    except Exception as e:
        raise Exception(f"Failed to get router response from OpenAI: {str(e)}")

    # Parse and normalize the response
    try:
        data = json.loads(raw)
        # Handle both JSON object and array responses
        if isinstance(data, dict):
            # If it's a dict, try to find a domain key
            domain = None
            for key in ['domain', 'selected_domain', 'result']:
                if key in data and data[key] in DOMAIN_TO_COLLECTION:
                    domain = data[key]
                    break
            if not domain:
                # Try to find any value that matches a domain
                for value in data.values():
                    if isinstance(value, str) and value in DOMAIN_TO_COLLECTION:
                        domain = value
                        break
            if not domain:
                raise ValueError(f"Could not extract domain from response: {data}")
        elif isinstance(data, list):
            domain = str(data[0]).strip() if data else None
            if not domain:
                raise ValueError("Empty domain list in response")
        else:
            domain = str(data).strip()
            
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract domain from raw text
        domain = raw.strip()
        # Remove common JSON artifacts
        domain = domain.strip('[]"\'')
    except Exception as e:
        raise ValueError(f"Failed to parse router response: {str(e)}")

    # Final validation
    if domain not in DOMAIN_TO_COLLECTION:
        raise ValueError(
            f"Router returned invalid domain: '{domain}'. "
            f"Valid domains are: {', '.join(available_domains)}"
        )

    return domain


def search_similar_text(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches for similar text in Qdrant vector database.
    
    Args:
        query_text: The query string to search for
        top_k: Number of top results to return (default: 3)
        
    Returns:
        List of dictionaries containing id, payload, and score for each result
        
    Raises:
        ValueError: If collection doesn't exist or query fails
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    
    # Route query to appropriate domain
    domain = smart_route_query(query_text)
    # print(f"Routed domain: {domain}")

    collection_name = DOMAIN_TO_COLLECTION[domain]
    
    # Check if collection exists
    if not qdrant.collection_exists(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' does not exist in Qdrant. "
            f"Please create it first or check the collection name."
        )

    # Generate embedding using E5 model
    try:
        embedding = e5_model.encode(f"query: {query_text}", normalize_embeddings=True).tolist()
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")

    # Query Qdrant
    try:
        results = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False  # Don't return vectors to save bandwidth
        )
    except Exception as e:
        raise Exception(f"Failed to query Qdrant: {str(e)}")

    # Process results
    final_results = []
    
    # Qdrant returns ScoredPoint objects or tuples depending on version
    for item in results:
        try:
            # Handle ScoredPoint object (newer Qdrant client)
            if hasattr(item, 'id') and hasattr(item, 'payload'):
                point_id = item.id
                payload = item.payload
                score = getattr(item, 'score', None)
            # Handle tuple format (older Qdrant client)
            elif isinstance(item, (tuple, list)):
                if len(item) >= 2:
                    point_id = item[0]
                    payload = item[1]
                    score = item[2] if len(item) > 2 else None
                else:
                    continue
            else:
                # Fallback: try to extract as dict
                point_id = item.get('id') if isinstance(item, dict) else None
                payload = item.get('payload') if isinstance(item, dict) else item
                score = item.get('score') if isinstance(item, dict) else None
                
            if point_id is None:
                continue
                
            final_results.append({
                "id": str(point_id),
                "payload": payload if payload else {},
                "score": float(score) if score is not None else None
            })
        except Exception as e:
            print(f"Warning: Failed to process result item: {str(e)}")
            continue

    return final_results



def answer_query(query: str, qdrant_results: List[Dict[str, Any]]) -> str:
    """
    Produces final factual answer strictly from Qdrant vector DB results.
    
    Args:
        query: The original user query
        qdrant_results: List of search results from Qdrant
        
    Returns:
        The generated answer string
        
    Raises:
        ValueError: If query is empty or no results provided
        Exception: If OpenAI API call fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not qdrant_results:
        return "I couldn't find any relevant information to answer your question."

    safe_results = safe_json(qdrant_results)

    ANSWER_PROMPT = f"""
    You are a factual RAG (Retrieval-Augmented Generation) answer engine.
    Your task is to answer the user's question using ONLY the information provided in the search results.
    
    Important rules:
    - Use only the facts from the provided results
    - Do not make up information that is not in the results
    - If the results don't contain enough information, say so clearly
    - Be concise and accurate
    - Cite specific details from the results when relevant

    Question: {query}

    Search Results:
    {json.dumps(safe_results, indent=2)}

    Provide a clear, factual answer based on the search results above:
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed: correct model name
            messages=[
                {"role": "system", "content": "You are a factual RAG answer engine. Do not hallucinate. Use only the provided information."},
                {"role": "user", "content": ANSWER_PROMPT}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        raise Exception(f"Failed to generate answer from OpenAI: {str(e)}")


def safe_json(obj: Any) -> Any:
    """
    Recursively converts objects to JSON-serializable format.
    Handles Qdrant ScoredPoint objects and other complex types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, list):
        return [safe_json(o) for o in obj]

    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}

    # Qdrant ScoredPoint object
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
        vector_size: Size of the vectors (embedding dimension)
        
    Raises:
        Exception: If collection creation fails
    """
    if not collection_name:
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
            print(f"Created collection: {collection_name}")
        except Exception as e:
            raise Exception(f"Failed to create collection '{collection_name}': {str(e)}")
    else:
        print(f"Collection '{collection_name}' already exists")



# Main execution

if __name__ == "__main__":
    try:
        query = input("Enter your query: ").strip()
        
        if not query:
            print("Error: Query cannot be empty")
            exit(1)
        
        # Route query to appropriate domain
        domain = smart_route_query(query)
        # print(f"Routed to domain: {domain}")
        print(f"Using collection: {DOMAIN_TO_COLLECTION[domain]}")

        # Search for similar text
        results = search_similar_text(query, top_k=3)
        # print(f"Found {len(results)} results\n")

        # Generate answer
        answer = answer_query(query, results)
        print("Answer:", answer)
        print("-"*50)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()































# import json
# import os
# from typing import List
# from openai import OpenAI
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from qdrant_client import QdrantClient

# 
# # Load environment vars
# 
# load_dotenv()

# DOMAIN_TO_COLLECTION = json.loads(os.getenv("DOMAIN_TO_COLLECTION", "{}"))
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# qdrant = QdrantClient(
#     url=os.getenv("qdrantUrl"),
#     api_key=os.getenv("qdrantApiKey")
# )

# 
# # Route query to collections
# 
# def smart_route_query(query: str) -> List[str]:
#     ROUTER_PROMPT = """
#     You are a domain classifier.

#     Available domains:
#     - emails
#     - webScrapingData
#     - OnlineRetailDataSet

#     Task:
#     Given a question, determine which single domain from the list is most relevant. 

#     Instructions:
#     - Return ONLY a JSON array containing ONE domain name that best matches the question.
#     - Do not include any explanations, text, or extra formatting.

#     Question: {query}
#     """

#     response = openai_client.responses.create(
#         model="gpt-4.1-mini",
#         input=[
#             {"role": "system", "content": "You are a domain classifier. Output JSON only."},
#             {"role": "user", "content": ROUTER_PROMPT.format(query=query)}
#         ],
#         temperature=0
#     )

#     raw = response.output_text.strip()

#     try:
#         domains = json.loads(raw)
#         if not isinstance(domains, list):
#             raise ValueError
#     except Exception:
#         raise ValueError(f"Invalid router output: {raw}")

#     collections = [
#         DOMAIN_TO_COLLECTION[d]
#         for d in domains
#         if d in DOMAIN_TO_COLLECTION
#     ]

#     if not collections:
#         raise ValueError(f"No valid domain mapped for: {domains}")

#     return collections



# def search_similar_text(query_text: str, top_k: int = 3):
#     """
#     Returns Top-K similar records from Qdrant with FULL payload
#     """

#     # Create embedding
#     embedding = openai_client.embeddings.create(
#         model="text-embedding-3-small",
#         input=query_text
#     ).data[0].embedding

#     collection_name = smart_route_query(query_text)[0]

#     # Check if collection exists
#     if not qdrant.collection_exists(collection_name):
#         raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant")

#     # FIX 4: Use the standard search method which is more reliable
#     results = qdrant.query_points(
#         collection_name=collection_name,
#         query=embedding,  # ✅ correct parameter
#         limit=top_k,
#         with_payload=True
#     )

#     final_results = []

#     for point_id, payload in results:
#         final_results.append({
#             "id": point_id,
#             "payload": payload
#         })



#     return final_results


# def answer_query(query: str, qdrant_results) -> str:
#     """
#     Produces final factual answer strictly from Qdrant vector DB results.
#     """

#     safe_results = safe_json(qdrant_results)

#     ANSWER_PROMPT = f"""
#     Merge relevant facts from results to answer the question.
#     Use only the given data.

#     Q: {query}

#     Results:
#     {json.dumps(safe_results, indent=2)}

#     Answer:
#     """

#     response = openai_client.responses.create(
#         model="gpt-4.1-mini",
#         input=[
#             {"role": "system", "content": "You are a factual RAG answer engine. Do not hallucinate."},
#             {"role": "user", "content": ANSWER_PROMPT}
#         ],
#         temperature=0
#     )

#     return response.output_text.strip()


# def safe_json(obj):
#     if isinstance(obj, list):
#         return [safe_json(o) for o in obj]

#     if isinstance(obj, dict):
#         return {k: safe_json(v) for k, v in obj.items()}

#     # Qdrant ScoredPoint
#     if hasattr(obj, "payload") and hasattr(obj, "score"):
#         return {
#             "id": str(getattr(obj, "id", "")),
#             "score": float(getattr(obj, "score", 0)),
#             "payload": safe_json(getattr(obj, "payload", {}))
#         }

#     # Any other object → string fallback
#     if not isinstance(obj, (str, int, float, bool, type(None))):
#         return str(obj)

#     return obj

# from qdrant_client.models import VectorParams, Distance

# def create_collection_if_not_exists(collection_name, vector_size):
#     if not qdrant.collection_exists(collection_name):
#         qdrant.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(
#                 size=vector_size,
#                 distance=Distance.COSINE
#             )
#         )




# 
# # Run
# 
# if __name__ == "__main__":
#     query = input("Enter your query: ")

#     collections = smart_route_query(query)
#     print(f"Routed to collections: {collections}")

#     results = search_similar_text(query)
#     print(answer_query(query, results))
