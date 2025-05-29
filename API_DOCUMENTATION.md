# 🚀 NL2SQL API Documentation

## Overview

The Schema-Aware NL2SQL API provides RESTful endpoints to convert natural language questions into SQL queries and execute them against connected databases.

## Base URL

```
http://localhost:8000  # Local development
https://your-domain.com  # Production
```

## Authentication

All endpoints (except `/` and `/health`) require API key authentication using Bearer token:

```bash
Authorization: Bearer your-api-key-here
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check API health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "database_connected": true,
  "model_loaded": true,
  "uptime": 3600.5
}
```

### 2. Connect to Database

**POST** `/connect`

Connect to a database.

**Request Body:**
```json
{
  "db_type": "sqlite",
  "db_path": "/path/to/database.db"
}
```

**PostgreSQL Example:**
```json
{
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "user": "username",
  "password": "password"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully connected to database",
  "database_type": "sqlite",
  "tables_count": 5
}
```

### 3. Process Natural Language Query

**POST** `/query`

Convert natural language to SQL and optionally execute it.

**Request Body:**
```json
{
  "query": "Show all books by J.K. Rowling",
  "execute": true,
  "include_schema": false
}
```

**Response:**
```json
{
  "success": true,
  "generated_sql": "SELECT * FROM books WHERE author = 'J.K. Rowling';",
  "results": [
    {
      "id": 7,
      "title": "Harry Potter and the Sorcerer's Stone",
      "author": "J.K. Rowling",
      "genre": "Fantasy",
      "publication_year": 1997,
      "price": 15.99,
      "rating": 4.7
    }
  ],
  "confidence_score": 0.85,
  "processing_time": 1.23,
  "row_count": 1
}
```

### 4. Batch Query Processing

**POST** `/batch-query`

Process multiple natural language queries in one request.

**Request Body:**
```json
{
  "queries": [
    "Show all books",
    "What is the average book rating?",
    "How many authors are there?"
  ],
  "execute": true
}
```

**Response:**
```json
{
  "success": true,
  "total_queries": 3,
  "results": [
    {
      "query": "Show all books",
      "success": true,
      "generated_sql": "SELECT * FROM books;",
      "results": [...],
      "confidence_score": 0.9,
      "processing_time": 0.8,
      "row_count": 10
    },
    ...
  ]
}
```

### 5. Execute SQL Directly

**POST** `/execute-sql`

Execute SQL queries directly.

**Request Body:**
```json
{
  "sql": "SELECT COUNT(*) FROM books WHERE rating > 4.0;",
  "validate_only": false
}
```

**Response:**
```json
{
  "success": true,
  "sql": "SELECT COUNT(*) FROM books WHERE rating > 4.0;",
  "results": [{"count": 8}],
  "row_count": 1,
  "columns": ["count"]
}
```

### 6. Get Database Schema

**GET** `/schema?table_name=books`

Get database schema information.

**Response:**
```json
{
  "success": true,
  "database_type": "sqlite",
  "tables": ["books", "authors"],
  "total_tables": 2,
  "schema_details": {
    "database_type": "sqlite",
    "tables": {...},
    "relationships": [...]
  }
}
```

### 7. Get Query History

**GET** `/history?limit=10`

Get recent query history.

**Response:**
```json
{
  "success": true,
  "history": [
    {
      "query": "Show all books",
      "sql": "SELECT * FROM books;",
      "success": true,
      "timestamp": 1703123456.789,
      "row_count": 10
    }
  ],
  "count": 1
}
```

### 8. Get Statistics

**GET** `/statistics`

Get usage statistics.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_queries": 150,
    "successful_queries": 142,
    "failed_queries": 8,
    "success_rate": 0.947,
    "average_confidence": 0.84,
    "api_requests": 200,
    "uptime": 7200.5
  }
}
```

### 9. Load Different Model

**POST** `/load-model`

Load a different NL2SQL model.

**Request Body:**
```json
{
  "model_name": "tscholak/t5-base-spider"
}
```

### 10. Disconnect Database

**DELETE** `/disconnect`

Disconnect from the current database.

**Response:**
```json
{
  "success": true,
  "message": "Disconnected from database"
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error description",
  "status_code": 400,
  "timestamp": 1703123456.789
}
```

Common HTTP status codes:
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing API key)
- `403` - Forbidden (invalid API key)
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable (agent not initialized)

## Usage Examples

### Python Client

```python
import requests

API_BASE = "http://localhost:8000"
API_KEY = "your-api-key-here"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Connect to database
response = requests.post(f"{API_BASE}/connect", 
    json={"db_type": "sqlite", "db_path": "sample.db"},
    headers=headers
)

# Query database
response = requests.post(f"{API_BASE}/query",
    json={"query": "Show all books with rating above 4.5"},
    headers=headers
)

print(response.json())
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Authorization': 'Bearer your-api-key-here',
    'Content-Type': 'application/json'
  }
});

// Connect to database
await api.post('/connect', {
  db_type: 'sqlite',
  db_path: 'sample.db'
});

// Query database
const response = await api.post('/query', {
  query: 'Show all books with rating above 4.5'
});

console.log(response.data);
```

### cURL Examples

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Connect to database
curl -X POST http://localhost:8000/connect \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"db_type":"sqlite","db_path":"sample.db"}'

# Query database
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query":"Show all books","execute":true}'
```

## Deployment

### Docker

```bash
# Build and run
docker build -t nl2sql-api .
docker run -p 8000:8000 -e NL2SQL_API_KEY=your-key nl2sql-api

# Using docker-compose
docker-compose up -d
```

### Environment Variables

- `NL2SQL_API_KEY` - API authentication key
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: INFO)

### Cloud Deployment

The API can be deployed on:
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku**
- **DigitalOcean App Platform**
- **Kubernetes clusters**

### Production Considerations

1. **Security:**
   - Use strong API keys
   - Configure CORS appropriately
   - Use HTTPS in production
   - Implement rate limiting

2. **Performance:**
   - Use GPU-enabled instances for better model performance
   - Implement caching for frequently used queries
   - Monitor memory usage (models can be large)

3. **Monitoring:**
   - Set up health checks
   - Monitor API response times
   - Track error rates
   - Monitor resource usage

## Interactive Documentation

Once the API is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

These provide interactive documentation where you can test endpoints directly.

## Rate Limiting

Default limits (configurable):
- 100 requests per minute per API key
- Batch queries count as multiple requests
- Rate limit headers included in responses

## Support

For issues and support:
1. Check the health endpoint
2. Review API logs
3. Verify database connectivity
4. Check model loading status
5. Consult the troubleshooting section in README.md 