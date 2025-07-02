#!/usr/bin/env python3
"""
FastAPI Endpoint for Schema-Aware NL2SQL Agent
Production-ready API for natural language to SQL conversion
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import logging
import time
import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn

from src.nl2sql_agent import NL2SQLAgent
from src.schema_retriever import create_schema_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
nl2sql_agent: Optional[NL2SQLAgent] = None

# Security
security = HTTPBearer(auto_error=False)
API_KEY = os.getenv("NL2SQL_API_KEY", "your-secure-api-key-here")

# Pydantic Models
class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    db_type: str = Field(..., description="Database type: sqlite, postgresql, mysql")
    db_path: Optional[str] = Field(None, description="Path for SQLite database")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: Optional[str] = Field(None, description="Database name")
    user: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")

class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language question", min_length=1)
    execute: bool = Field(True, description="Whether to execute the generated SQL")
    include_schema: bool = Field(False, description="Include schema information in response")
    use_few_shot: bool = Field(False, description="Use few-shot learning for complex queries")
    use_enhanced_prompts: bool = Field(True, description="Use enhanced prompt engineering")

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[str] = Field(..., description="List of natural language questions")
    execute: bool = Field(True, description="Whether to execute generated SQL queries")

class ModelConfig(BaseModel):
    """Model configuration"""
    model_name: str = Field("mrm8488/t5-base-finetuned-wikiSQL", description="Hugging Face model name")

class SQLExecuteRequest(BaseModel):
    """Direct SQL execution request"""
    sql: str = Field(..., description="SQL query to execute")
    validate_only: bool = Field(False, description="Only validate, don't execute")

# Response Models
class QueryResponse(BaseModel):
    """Query response"""
    success: bool
    generated_sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
    schema_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    database_connected: bool
    model_loaded: bool
    uptime: float

class SchemaResponse(BaseModel):
    """Schema information response"""
    success: bool
    database_type: Optional[str] = None
    tables: Optional[List[str]] = None
    total_tables: Optional[int] = None
    schema_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("🚀 Starting NL2SQL API service...")
    yield
    logger.info("🛑 Shutting down NL2SQL API service...")

app = FastAPI(
    title="Schema-Aware NL2SQL API",
    description="Convert natural language questions to SQL queries with schema awareness",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking
start_time = time.time()
request_count = 0

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return credentials.credentials

# Utility functions
def get_agent() -> NL2SQLAgent:
    """Get the global agent instance"""
    global nl2sql_agent
    if nl2sql_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NL2SQL agent not initialized. Please connect to a database first."
        )
    return nl2sql_agent

async def increment_request_count():
    """Increment request counter"""
    global request_count
    request_count += 1

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Schema-Aware NL2SQL API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global nl2sql_agent, start_time
    
    database_connected = False
    model_loaded = False
    
    if nl2sql_agent is not None:
        database_connected = nl2sql_agent.schema_retriever is not None
        model_loaded = nl2sql_agent.model is not None
    
    return HealthResponse(
        status="healthy" if (database_connected and model_loaded) else "partial",
        timestamp=time.time(),
        database_connected=database_connected,
        model_loaded=model_loaded,
        uptime=time.time() - start_time
    )

@app.post("/connect", dependencies=[Depends(verify_api_key)])
async def connect_database(
    config: DatabaseConfig,
    background_tasks: BackgroundTasks
):
    """Connect to a database"""
    global nl2sql_agent
    
    background_tasks.add_task(increment_request_count)
    
    try:
        # Initialize agent if not exists
        if nl2sql_agent is None:
            nl2sql_agent = NL2SQLAgent()
        
        # Prepare connection parameters
        db_params = {}
        if config.db_path:
            db_params['db_path'] = config.db_path
        if config.host:
            db_params['host'] = config.host
        if config.port:
            db_params['port'] = config.port
        if config.database:
            db_params['database'] = config.database
        if config.user:
            db_params['user'] = config.user
        if config.password:
            db_params['password'] = config.password
        
        # Connect to database
        success = nl2sql_agent.connect_database(config.db_type, **db_params)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to connect to database"
            )
        
        # Load default model if not loaded
        if nl2sql_agent.model is None:
            model_success = nl2sql_agent.load_model()
            if not model_success:
                logger.warning("Failed to load default model")
        
        return {
            "success": True,
            "message": "Successfully connected to database",
            "database_type": config.db_type,
            "tables_count": len(nl2sql_agent.current_schema.get('tables', {}))
        }
        
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {str(e)}"
        )

@app.post("/load-model", dependencies=[Depends(verify_api_key)])
async def load_model(
    config: ModelConfig,
    background_tasks: BackgroundTasks
):
    """Load a specific NL2SQL model"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        success = agent.load_model(config.model_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to load model"
            )
        
        return {
            "success": True,
            "message": f"Successfully loaded model: {config.model_name}",
            "model_name": config.model_name
        }
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Process a natural language query"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        # Use enhanced prompt engineering if requested
        if request.use_enhanced_prompts and agent.prompt_engineer:
            result = agent.process_query(request.query, use_few_shot=request.use_few_shot)
        else:
            # Fallback to basic processing
            result = agent.process_query(request.query)
        
        # Add schema info if requested
        schema_info = None
        if request.include_schema and result.get("success"):
            schema_info = agent.get_schema_info()
        
        return QueryResponse(
            success=result.get("success", False),
            generated_sql=result.get("generated_sql"),
            results=result.get("results") if request.execute else None,
            confidence_score=result.get("confidence_score"),
            processing_time=result.get("processing_time"),
            row_count=result.get("row_count"),
            error=result.get("error"),
            schema_info=schema_info
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/batch-query", dependencies=[Depends(verify_api_key)])
async def process_batch_queries(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks
):
    """Process multiple natural language queries"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        results = []
        for query in request.queries:
            result = agent.process_query(query)
            results.append({
                "query": query,
                "success": result.get("success", False),
                "generated_sql": result.get("generated_sql"),
                "results": result.get("results") if request.execute else None,
                "confidence_score": result.get("confidence_score"),
                "processing_time": result.get("processing_time"),
                "row_count": result.get("row_count"),
                "error": result.get("error")
            })
        
        return {
            "success": True,
            "total_queries": len(request.queries),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch query processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query processing failed: {str(e)}"
        )

@app.post("/execute-sql", dependencies=[Depends(verify_api_key)])
async def execute_sql(
    request: SQLExecuteRequest,
    background_tasks: BackgroundTasks
):
    """Execute SQL directly"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        if request.validate_only:
            # Only validate
            validation = agent.schema_retriever.simple_validate_query(request.sql)
            return {
                "success": validation["valid"],
                "message": validation["message"],
                "sql": request.sql,
                "validated_only": True
            }
        else:
            # Execute SQL
            results_df = agent.schema_retriever.execute_query(request.sql)
            results = results_df.to_dict('records') if not results_df.empty else []
            
            return {
                "success": True,
                "sql": request.sql,
                "results": results,
                "row_count": len(results),
                "columns": list(results_df.columns)
            }
            
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SQL execution failed: {str(e)}"
        )

@app.get("/schema", response_model=SchemaResponse, dependencies=[Depends(verify_api_key)])
async def get_schema(
    table_name: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Get database schema information"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        schema_info = agent.get_schema_info(table_name)
        
        return SchemaResponse(
            success=schema_info.get("success", False),
            database_type=schema_info.get("database_type"),
            tables=schema_info.get("tables"),
            total_tables=schema_info.get("total_tables"),
            schema_details=schema_info,
            error=schema_info.get("error")
        )
        
    except Exception as e:
        logger.error(f"Schema retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema retrieval failed: {str(e)}"
        )

@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_query_history(
    limit: int = 10,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Get query history"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        history = agent.get_query_history(limit)
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History retrieval failed: {str(e)}"
        )

@app.get("/statistics", dependencies=[Depends(verify_api_key)])
async def get_statistics(background_tasks: BackgroundTasks = BackgroundTasks()):
    """Get usage statistics"""
    agent = get_agent()
    background_tasks.add_task(increment_request_count)
    
    try:
        stats = agent.get_statistics()
        stats["api_requests"] = request_count
        stats["uptime"] = time.time() - start_time
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Statistics retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval failed: {str(e)}"
        )

@app.delete("/disconnect", dependencies=[Depends(verify_api_key)])
async def disconnect_database(background_tasks: BackgroundTasks = BackgroundTasks()):
    """Disconnect from database"""
    global nl2sql_agent
    background_tasks.add_task(increment_request_count)
    
    try:
        if nl2sql_agent:
            nl2sql_agent.close_connections()
            nl2sql_agent = None
        
        return {
            "success": True,
            "message": "Disconnected from database"
        }
        
    except Exception as e:
        logger.error(f"Disconnect error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disconnect failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": time.time()
        }
    )

# Main function for running the API
def main():
    """Run the FastAPI server"""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main() 