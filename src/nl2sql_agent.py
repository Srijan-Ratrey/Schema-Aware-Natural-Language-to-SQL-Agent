"""
Main NL2SQL Agent that orchestrates schema-aware SQL generation
Integrates schema retrieval, model inference, and query execution
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .schema_retriever import SchemaRetriever, create_schema_retriever
from .nl2sql_model import NL2SQLModel, load_model
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NL2SQLAgent:
    """
    Schema-Aware Natural Language to SQL Agent
    
    Main orchestrator that:
    1. Connects to databases and extracts schemas
    2. Processes natural language queries
    3. Generates SQL using fine-tuned models
    4. Executes queries and returns results
    """
    
    def __init__(
        self,
        model_name: str = "mrm8488/t5-base-finetuned-wikiSQL",
        db_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NL2SQL Agent
        
        Args:
            model_name: Hugging Face model name or local path
            db_config: Database configuration dictionary
        """
        self.model_name = model_name
        self.db_config = db_config or {}
        
        # Initialize components
        self.model = None
        self.schema_retriever = None
        self.current_schema = None
        
        # Performance tracking
        self.query_history = []
        self.statistics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "start_time": time.time()
        }
        
        logger.info("NL2SQL Agent initialized")
    
    def connect_database(self, db_type: str, **db_params) -> bool:
        """
        Connect to database and extract schema
        
        Args:
            db_type: Database type ('sqlite', 'postgresql', 'mysql')
            **db_params: Database connection parameters
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to {db_type} database...")
            
            # Create schema retriever
            self.schema_retriever = create_schema_retriever(db_type, **db_params)
            
            # Extract and cache schema
            self.current_schema = self.schema_retriever.get_database_schema()
            
            logger.info(f"Successfully connected to database with {len(self.current_schema['tables'])} tables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load NL2SQL model
        
        Args:
            model_name: Model name to load (optional, uses default if not provided)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_to_load = model_name or self.model_name
            logger.info(f"Loading model: {model_to_load}")
            
            self.model = load_model(model_to_load)
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return SQL + results
        
        Args:
            natural_language_query: User's question in natural language
            
        Returns:
            Dictionary containing generated SQL, results, and metadata
        """
        start_time = time.time()
        
        try:
            # Get schema information (use simplified version for WikiSQL models)
            schema_prompt = self.schema_retriever.get_simple_schema_prompt()
            
            # Generate SQL using the model
            generation_result = self.model.generate_sql(
                natural_language_query, 
                schema_prompt
            )
            
            if generation_result.get("error"):
                return {
                    "success": False,
                    "error": generation_result["error"],
                    "query": natural_language_query,
                    "processing_time": time.time() - start_time
                }
            
            generated_sql = generation_result["sql_query"]
            confidence_score = generation_result["confidence_score"]
            
            # Validate SQL
            validation_result = self.schema_retriever.simple_validate_query(generated_sql)
            
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"SQL validation failed: {validation_result['message']}",
                    "generated_sql": generated_sql,
                    "query": natural_language_query,
                    "confidence_score": confidence_score,
                    "processing_time": time.time() - start_time
                }
            
            # Execute query
            try:
                results_df = self.schema_retriever.execute_query(generated_sql)
                results = results_df.to_dict('records') if not results_df.empty else []
                
                # Store in history
                query_result = {
                    "query": natural_language_query,
                    "sql": generated_sql,
                    "results": results,
                    "confidence": confidence_score,
                    "timestamp": time.time(),
                    "success": True,
                    "row_count": len(results)
                }
                
                self.query_history.append(query_result)
                self.statistics["total_queries"] += 1
                self.statistics["successful_queries"] += 1
                
                processing_time = time.time() - start_time
                self.statistics["total_processing_time"] += processing_time
                
                return {
                    "success": True,
                    "generated_sql": generated_sql,
                    "results": results,
                    "results_df": results_df,
                    "confidence_score": confidence_score,
                    "query": natural_language_query,
                    "processing_time": processing_time,
                    "row_count": len(results)
                }
                
            except Exception as e:
                error_msg = f"Query execution failed: {str(e)}"
                logger.error(error_msg)
                
                self.query_history.append({
                    "query": natural_language_query,
                    "sql": generated_sql,
                    "error": error_msg,
                    "confidence": confidence_score,
                    "timestamp": time.time(),
                    "success": False
                })
                
                self.statistics["total_queries"] += 1
                self.statistics["failed_queries"] += 1
                
                return {
                    "success": False,
                    "error": error_msg,
                    "generated_sql": generated_sql,
                    "query": natural_language_query,
                    "confidence_score": confidence_score,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            
            self.statistics["total_queries"] += 1
            self.statistics["failed_queries"] += 1
            
            return {
                "success": False,
                "error": error_msg,
                "query": natural_language_query,
                "processing_time": time.time() - start_time
            }
    
    def explain_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Explain what an SQL query does in natural language
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Dictionary with explanation and query analysis
        """
        try:
            # Basic SQL analysis
            sql_upper = sql_query.upper().strip()
            
            explanation = {
                "query_type": self._identify_query_type(sql_upper),
                "tables_involved": self._extract_table_names(sql_query),
                "estimated_complexity": self._estimate_complexity(sql_query),
                "explanation": self._generate_explanation(sql_query)
            }
            
            return {"success": True, **explanation}
            
        except Exception as e:
            return self._error_response(f"Failed to explain SQL: {str(e)}")
    
    def get_schema_info(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database schema information
        
        Args:
            table_name: Specific table name (optional)
            
        Returns:
            Schema information
        """
        if not self.current_schema:
            return self._error_response("No database connected")
        
        if table_name:
            if table_name in self.current_schema["tables"]:
                return {
                    "success": True,
                    "table": table_name,
                    "schema": self.current_schema["tables"][table_name]
                }
            else:
                return self._error_response(f"Table '{table_name}' not found")
        else:
            return {
                "success": True,
                "database_type": self.current_schema["database_type"],
                "tables": list(self.current_schema["tables"].keys()),
                "total_tables": len(self.current_schema["tables"]),
                "relationships": self.current_schema["relationships"]
            }
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of the table
            limit: Number of sample rows
            
        Returns:
            Sample data
        """
        if not self.schema_retriever:
            return self._error_response("No database connected")
        
        try:
            sample_df = self.schema_retriever.get_sample_data(table_name, limit)
            
            return {
                "success": True,
                "table": table_name,
                "sample_data": sample_df.to_dict('records'),
                "columns": list(sample_df.columns),
                "sample_size": len(sample_df)
            }
            
        except Exception as e:
            return self._error_response(f"Failed to get sample data: {str(e)}")
    
    def batch_query(self, queries: List[str], execute: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of natural language queries
            execute: Whether to execute generated SQL
            
        Returns:
            List of query results
        """
        results = []
        
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        
        return results
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent query history
        
        Args:
            limit: Number of recent queries to return
            
        Returns:
            List of recent queries and results
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent usage statistics
        
        Returns:
            Usage statistics
        """
        if not self.query_history:
            return {"total_queries": 0}
        
        successful_queries = sum(1 for q in self.query_history if q.get("success", False))
        executed_queries = sum(1 for q in self.query_history if q.get("executed", False))
        
        avg_confidence = sum(q.get("confidence_score", 0) for q in self.query_history) / len(self.query_history)
        avg_execution_time = sum(q.get("execution_time", 0) for q in self.query_history) / len(self.query_history)
        
        return {
            "total_queries": len(self.query_history),
            "successful_queries": successful_queries,
            "executed_queries": executed_queries,
            "success_rate": successful_queries / len(self.query_history),
            "execution_rate": executed_queries / len(self.query_history),
            "average_confidence": avg_confidence,
            "average_execution_time": avg_execution_time
        }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": message,
            "timestamp": time.time()
        }
    
    def _log_query(self, nl_query: str, sql_query: str, response: Dict[str, Any]):
        """Log query for history tracking"""
        log_entry = {
            "natural_language_query": nl_query,
            "generated_sql": sql_query,
            "success": response.get("success", False),
            "executed": response.get("executed", False),
            "confidence_score": response.get("confidence_score", 0.0),
            "execution_time": response.get("execution_time", 0.0),
            "timestamp": response.get("timestamp", time.time())
        }
        
        self.query_history.append(log_entry)
        
        # Keep only last 100 queries
        if len(self.query_history) > 100:
            self.query_history.pop(0)
    
    def _identify_query_type(self, sql: str) -> str:
        """Identify the type of SQL query"""
        if sql.startswith("SELECT"):
            return "SELECT"
        elif sql.startswith("INSERT"):
            return "INSERT"
        elif sql.startswith("UPDATE"):
            return "UPDATE"
        elif sql.startswith("DELETE"):
            return "DELETE"
        else:
            return "OTHER"
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query (basic implementation)"""
        import re
        
        # Simple regex to find table names after FROM and JOIN
        pattern = r'\b(?:FROM|JOIN)\s+(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return list(set(matches))
    
    def _estimate_complexity(self, sql: str) -> str:
        """Estimate query complexity"""
        sql_upper = sql.upper()
        complexity_score = 0
        
        # Add points for various SQL features
        if 'JOIN' in sql_upper:
            complexity_score += 2
        if 'SUBQUERY' in sql_upper or '(' in sql:
            complexity_score += 3
        if 'GROUP BY' in sql_upper:
            complexity_score += 1
        if 'HAVING' in sql_upper:
            complexity_score += 2
        if 'ORDER BY' in sql_upper:
            complexity_score += 1
        
        if complexity_score <= 2:
            return "Low"
        elif complexity_score <= 5:
            return "Medium"
        else:
            return "High"
    
    def _generate_explanation(self, sql: str) -> str:
        """Generate basic explanation of SQL query"""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith("SELECT"):
            return "This query retrieves data from the database by selecting specific columns from one or more tables."
        elif sql_upper.startswith("INSERT"):
            return "This query adds new data to a database table."
        elif sql_upper.startswith("UPDATE"):
            return "This query modifies existing data in a database table."
        elif sql_upper.startswith("DELETE"):
            return "This query removes data from a database table."
        else:
            return "This is a database query."
    
    def close_connections(self):
        """Close database connections and cleanup"""
        if self.schema_retriever:
            self.schema_retriever.close_connection()
        
        logger.info("NL2SQL Agent connections closed") 