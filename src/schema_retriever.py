"""
Schema Retriever Module for Dynamic Database Schema Extraction
Supports multiple database dialects through SQLAlchemy
"""

import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaRetriever:
    """Dynamic database schema extractor supporting multiple dialects"""
    
    def __init__(self, connection_string: str):
        """
        Initialize schema retriever with database connection
        
        Args:
            connection_string: SQLAlchemy connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.metadata = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string, echo=False)
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            logger.info(f"Successfully connected to database")
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.error(f"Connection string: {self.connection_string}")
            raise
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        Extract complete database schema
        
        Returns:
            Dictionary containing database schema information
        """
        try:
            inspector = inspect(self.engine)
            schema = {
                "database_type": self.engine.dialect.name,
                "tables": {},
                "relationships": []
            }
            
            # Get all table names
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                schema["tables"][table_name] = self._get_table_schema(table_name, inspector)
            
            # Get foreign key relationships
            schema["relationships"] = self._get_relationships(inspector, table_names)
            
            return schema
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving schema: {e}")
            raise
    
    def _get_table_schema(self, table_name: str, inspector) -> Dict[str, Any]:
        """Extract schema for a specific table"""
        try:
            columns = inspector.get_columns(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            indexes = inspector.get_indexes(table_name)
            
            return {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"],
                        "default": col.get("default"),
                        "primary_key": col["name"] in primary_keys.get("constrained_columns", [])
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    }
                    for fk in foreign_keys
                ],
                "indexes": [
                    {
                        "name": idx["name"],
                        "columns": idx["column_names"],
                        "unique": idx["unique"]
                    }
                    for idx in indexes
                ]
            }
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving table schema for {table_name}: {e}")
            return {}
    
    def _get_relationships(self, inspector, table_names: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships between tables"""
        relationships = []
        
        for table_name in table_names:
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    relationships.append({
                        "from_table": table_name,
                        "from_columns": fk["constrained_columns"],
                        "to_table": fk["referred_table"],
                        "to_columns": fk["referred_columns"],
                        "constraint_name": fk.get("name")
                    })
            except SQLAlchemyError as e:
                logger.warning(f"Error getting relationships for {table_name}: {e}")
                continue
        
        return relationships
    
    def get_schema_prompt(self) -> str:
        """
        Generate a natural language prompt describing the database schema
        This will be used for the T5 model
        """
        schema = self.get_database_schema()
        
        prompt_parts = [
            f"Database Schema ({schema['database_type']}):",
            ""
        ]
        
        # Add explicit table names list at the beginning
        table_names = list(schema["tables"].keys())
        prompt_parts.append(f"IMPORTANT: Use these exact table names: {', '.join(table_names)}")
        prompt_parts.append("")
        
        # Add table information
        for table_name, table_info in schema["tables"].items():
            prompt_parts.append(f"Table: {table_name}")
            prompt_parts.append("Columns:")
            
            for col in table_info["columns"]:
                col_desc = f"  - {col['name']} ({col['type']})"
                if col['primary_key']:
                    col_desc += " [PRIMARY KEY]"
                if not col['nullable']:
                    col_desc += " [NOT NULL]"
                prompt_parts.append(col_desc)
            
            # Add foreign keys
            if table_info["foreign_keys"]:
                prompt_parts.append("Foreign Keys:")
                for fk in table_info["foreign_keys"]:
                    fk_desc = f"  - {', '.join(fk['columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                    prompt_parts.append(fk_desc)
            
            prompt_parts.append("")
        
        # Add relationships summary
        if schema["relationships"]:
            prompt_parts.append("Table Relationships:")
            for rel in schema["relationships"]:
                rel_desc = f"  - {rel['from_table']}.{', '.join(rel['from_columns'])} -> {rel['to_table']}.{', '.join(rel['to_columns'])}"
                prompt_parts.append(rel_desc)
            prompt_parts.append("")
        
        # Add final reminder about table names
        prompt_parts.append(f"REMEMBER: Use exact table names: {', '.join(table_names)}")
        
        return "\n".join(prompt_parts)
    
    def get_simple_schema_prompt(self) -> str:
        """
        Generate a simplified schema prompt optimized for T5-based models
        """
        schema = self.get_database_schema()
        
        # Start with explicit table names
        table_names = list(schema["tables"].keys())
        prompt_parts = [f"Available tables: {', '.join(table_names)}"]
        prompt_parts.append("")
        
        # For single table databases, provide detailed column info
        if len(schema["tables"]) == 1:
            table_name, table_info = next(iter(schema["tables"].items()))
            
            # Provide detailed column information
            columns_info = []
            for col in table_info["columns"]:
                col_info = f"{col['name']} ({col['type']})"
                if col['primary_key']:
                    col_info += " [PK]"
                columns_info.append(col_info)
            
            prompt_parts.append(f"Table '{table_name}' has columns: {', '.join(columns_info)}")
        
        # For multi-table, provide table names and key columns
        else:
            for table_name, table_info in schema["tables"].items():
                # Get primary key columns
                pk_columns = [col['name'] for col in table_info["columns"] if col['primary_key']]
                # Get a few key columns (first 3-4)
                key_columns = [col['name'] for col in table_info["columns"][:4]]
                
                table_info_str = f"Table '{table_name}': {', '.join(key_columns)}"
                if pk_columns:
                    table_info_str += f" (PK: {', '.join(pk_columns)})"
                prompt_parts.append(table_info_str)
        
        # Add important note about table names
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT: Use exact table names as shown above. Do not use singular/plural variations.")
        
        return "\n".join(prompt_parts)
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql_query(sql_query, connection)
                logger.info(f"Query executed successfully, returned {len(result)} rows")
                return result
                
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query without executing it
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Validation result with success status and message
        """
        try:
            with self.engine.connect() as connection:
                # Use EXPLAIN to validate without executing
                if self.engine.dialect.name == 'postgresql':
                    validation_query = f"EXPLAIN {sql_query}"
                elif self.engine.dialect.name == 'sqlite':
                    validation_query = f"EXPLAIN QUERY PLAN {sql_query}"
                else:
                    validation_query = f"EXPLAIN {sql_query}"
                
                # Execute the validation query
                result = connection.execute(text(validation_query))
                result.fetchall()  # Consume the result
                return {"valid": True, "message": "Query is valid"}
                
        except SQLAlchemyError as e:
            return {"valid": False, "message": str(e)}
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {str(e)}"}
        
    def simple_validate_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Simple validation by checking SQL syntax without EXPLAIN
        """
        try:
            # Basic syntax validation
            if not sql_query or not sql_query.strip():
                return {"valid": False, "message": "Empty query"}
            
            # Check for basic SQL keywords
            sql_upper = sql_query.upper().strip()
            if not any(keyword in sql_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                return {"valid": False, "message": "No valid SQL command found"}
            
            # For read-only operations, just check if it parses
            if sql_upper.startswith('SELECT'):
                try:
                    with self.engine.connect() as connection:
                        # Try to prepare the statement
                        stmt = text(sql_query)
                        # Just compile, don't execute
                        stmt.compile(dialect=self.engine.dialect)
                        return {"valid": True, "message": "Query syntax is valid"}
                except Exception as e:
                    return {"valid": False, "message": f"Syntax error: {str(e)}"}
            
            return {"valid": True, "message": "Query appears valid"}
            
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {str(e)}"}
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of the table
            limit: Number of sample rows
            
        Returns:
            Sample data as DataFrame
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Factory function for easy instantiation
def create_schema_retriever(db_type: str, **kwargs) -> SchemaRetriever:
    """
    Factory function to create schema retriever for different database types
    
    Args:
        db_type: Database type ('sqlite', 'postgresql', 'mysql', etc.)
        **kwargs: Database connection parameters
        
    Returns:
        SchemaRetriever instance
    """
    if db_type.lower() == 'sqlite':
        db_path = kwargs.get('database_path', kwargs.get('db_path', 'database.db'))
        connection_string = f"sqlite:///{db_path}"
    
    elif db_type.lower() == 'postgresql':
        user = kwargs.get('user', 'postgres')
        password = kwargs.get('password', '')
        host = kwargs.get('host', 'localhost')
        port = kwargs.get('port', 5432)
        database = kwargs.get('database', 'postgres')
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    elif db_type.lower() == 'mysql':
        user = kwargs.get('user', 'root')
        password = kwargs.get('password', '')
        host = kwargs.get('host', 'localhost')
        port = kwargs.get('port', 3306)
        database = kwargs.get('database', 'mysql')
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    return SchemaRetriever(connection_string) 