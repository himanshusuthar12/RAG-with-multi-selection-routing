# RAG System FastAPI

A FastAPI-based REST API for the Retrieval Augmented Generation (RAG) system with multi-collection routing.

## Features

- 🔍 **Query Routing**: Automatically routes queries to the most relevant Qdrant collection
- 🔎 **Vector Search**: Search for similar text in vector databases
- 💬 **Answer Generation**: Generate factual answers using OpenAI GPT models
- 📊 **Health Monitoring**: Health check endpoint for system status
- 📝 **Auto Documentation**: Interactive API documentation at `/docs`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
qdrantUrl=your_qdrant_url
qdrantApiKey=your_qdrant_api_key
DOMAIN_TO_COLLECTION={"emails": "collection1", "webScrapingData": "collection2", "OnlineRetailDataSet": "collection3"}
```

## Running the API

### Option 1: Direct Python
```bash
python rag_api.py
```

### Option 2: Uvicorn (Recommended)
```bash
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check the health status of the API and connections.

**Response:**
```json
{
  "status": "healthy",
  "openai_connected": true,
  "qdrant_connected": true,
  "collections_configured": 3
}
```

### 2. Route Query
**POST** `/route`

Route a query to determine which collection(s) it should use.

**Request:**
```json
{
  "query": "What are the latest emails?"
}
```

**Response:**
```json
{
  "query": "What are the latest emails?",
  "collections": ["emails_collection"],
  "domains": ["emails"]
}
```

### 3. Search
**POST** `/search`

Search for similar text in Qdrant collections.

**Request:**
```json
{
  "query_text": "What are the latest emails?",
  "top_k": 3,
  "collection_name": null
}
```

**Response:**
```json
{
  "query_text": "What are the latest emails?",
  "collection_name": "emails_collection",
  "results": [
    {
      "id": "123",
      "score": 0.95,
      "payload": {...}
    }
  ],
  "total_results": 3
}
```

### 4. Query (Main Endpoint)
**POST** `/query`

Main endpoint that routes, searches, and generates an answer.

**Request:**
```json
{
  "query": "What are the latest emails?",
  "top_k": 3,
  "collection_name": null
}
```

**Response:**
```json
{
  "query": "What are the latest emails?",
  "collections": ["emails_collection"],
  "results_count": 3,
  "answer": "Based on the search results...",
  "search_results": [
    {
      "id": "123",
      "score": 0.95,
      "payload": {...}
    }
  ]
}
```

## Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest emails?", "top_k": 3}'
```

### Using Python requests

```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the latest emails?",
        "top_k": 3
    }
)

result = response.json()
print(result["answer"])
```

### Using the example script

```bash
python api_example.py
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (health check failures)

Error responses include a `detail` field with error information:
```json
{
  "detail": "Query cannot be empty"
}
```

## CORS

The API includes CORS middleware configured to allow all origins. For production, update the `allow_origins` in `rag_api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Update this
    ...
)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `qdrantUrl` | Qdrant server URL | Yes |
| `qdrantApiKey` | Qdrant API key | Yes |
| `DOMAIN_TO_COLLECTION` | JSON mapping of domains to collections | Yes |

## Notes

- The API uses `gpt-4o-mini` for routing and answer generation
- Embeddings are created using `text-embedding-3-small`
- Default `top_k` is 3, maximum is 20
- All queries are automatically routed to the most relevant collection
