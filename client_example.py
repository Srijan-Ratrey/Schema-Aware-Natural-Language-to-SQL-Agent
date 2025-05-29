#!/usr/bin/env python3
"""
NL2SQL API Client Example
Demonstrates how to use the Schema-Aware NL2SQL API
"""

import requests
import json
import time
from typing import Dict, Any, List

class NL2SQLClient:
    """Python client for NL2SQL API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "your-api-key-here"):
        """
        Initialize the client
        
        Args:
            base_url: API base URL
            api_key: API authentication key
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    def connect_database(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to a database
        
        Args:
            db_config: Database configuration
            
        Example:
            db_config = {"db_type": "sqlite", "db_path": "sample.db"}
        """
        try:
            response = requests.post(
                f"{self.base_url}/connect",
                headers=self.headers,
                json=db_config
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def query(self, natural_language_query: str, execute: bool = True, include_schema: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            natural_language_query: User's question
            execute: Whether to execute the generated SQL
            include_schema: Include schema information in response
        """
        try:
            response = requests.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json={
                    "query": natural_language_query,
                    "execute": execute,
                    "include_schema": include_schema
                }
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Query failed: {str(e)}"}
    
    def batch_query(self, queries: List[str], execute: bool = True) -> Dict[str, Any]:
        """
        Process multiple natural language queries
        
        Args:
            queries: List of natural language questions
            execute: Whether to execute generated SQL queries
        """
        try:
            response = requests.post(
                f"{self.base_url}/batch-query",
                headers=self.headers,
                json={
                    "queries": queries,
                    "execute": execute
                }
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Batch query failed: {str(e)}"}
    
    def execute_sql(self, sql: str, validate_only: bool = False) -> Dict[str, Any]:
        """
        Execute SQL directly
        
        Args:
            sql: SQL query to execute
            validate_only: Only validate, don't execute
        """
        try:
            response = requests.post(
                f"{self.base_url}/execute-sql",
                headers=self.headers,
                json={
                    "sql": sql,
                    "validate_only": validate_only
                }
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"SQL execution failed: {str(e)}"}
    
    def get_schema(self, table_name: str = None) -> Dict[str, Any]:
        """
        Get database schema information
        
        Args:
            table_name: Specific table name (optional)
        """
        try:
            url = f"{self.base_url}/schema"
            if table_name:
                url += f"?table_name={table_name}"
            
            response = requests.get(url, headers=self.headers)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Schema retrieval failed: {str(e)}"}
    
    def get_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get query history"""
        try:
            response = requests.get(
                f"{self.base_url}/history?limit={limit}",
                headers=self.headers
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"History retrieval failed: {str(e)}"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/statistics",
                headers=self.headers
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Statistics retrieval failed: {str(e)}"}


def main():
    """Example usage of the NL2SQL client"""
    print("🧠 NL2SQL API Client Example")
    print("=" * 40)
    
    # Initialize client
    # Note: Replace with your actual API key and URL
    client = NL2SQLClient(
        base_url="http://localhost:8000",
        api_key="test-api-key-123"  # Replace with your actual API key
    )
    
    # 1. Health Check
    print("\n1. 🔍 Health Check")
    health = client.health_check()
    print(f"Health Status: {health}")
    
    # For demonstration, we'll simulate the API responses
    # In real usage, you'd start the API server first
    
    # 2. Connect to Database
    print("\n2. 🔌 Connect to Database")
    db_config = {
        "db_type": "sqlite",
        "db_path": "quickstart_sample.db"  # Using our sample database
    }
    
    print(f"Connecting to: {db_config}")
    # connection_result = client.connect_database(db_config)
    # print(f"Connection Result: {connection_result}")
    
    # 3. Sample Queries
    print("\n3. 💬 Sample Natural Language Queries")
    
    sample_queries = [
        "Show all books",
        "What are the top 3 highest rated books?",
        "How many books were published after 1950?",
        "Show the most expensive book",
        "List all fantasy books"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n  Query {i}: '{query}'")
        # result = client.query(query)
        # print(f"  Result: {result}")
        
        # Simulate expected response structure
        print(f"  Expected: Generate SQL and execute against database")
    
    # 4. Batch Processing
    print("\n4. 📦 Batch Query Processing")
    print(f"Processing {len(sample_queries)} queries in batch...")
    # batch_result = client.batch_query(sample_queries[:3])
    # print(f"Batch Result: {batch_result}")
    
    # 5. Direct SQL Execution
    print("\n5. ⚡ Direct SQL Execution")
    sql_query = "SELECT COUNT(*) as total_books FROM books;"
    print(f"Executing: {sql_query}")
    # sql_result = client.execute_sql(sql_query)
    # print(f"SQL Result: {sql_result}")
    
    # 6. Schema Information
    print("\n6. 🗂️  Database Schema")
    # schema = client.get_schema()
    # print(f"Schema: {schema}")
    
    # 7. Statistics
    print("\n7. 📊 Usage Statistics")
    # stats = client.get_statistics()
    # print(f"Statistics: {stats}")
    
    print("\n" + "=" * 40)
    print("✅ Client example completed!")
    print("\nTo use this client:")
    print("1. Start the API server: python api.py")
    print("2. Set your API key in the client")
    print("3. Connect to your database")
    print("4. Start querying with natural language!")


def demo_with_running_api():
    """Demo that works with a running API server"""
    client = NL2SQLClient(
        base_url="http://localhost:8000",
        api_key="test-api-key-123"
    )
    
    print("🚀 Testing with live API...")
    
    # Test health endpoint
    health = client.health_check()
    if "error" in health:
        print(f"❌ API not available: {health['error']}")
        print("💡 Start the API first: python api.py")
        return
    
    print(f"✅ API is healthy: {health.get('status', 'unknown')}")
    
    # Test other endpoints...
    # (Add more tests here once the API is running)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        demo_with_running_api()
    else:
        main() 