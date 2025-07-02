#!/usr/bin/env python3
"""
Unit tests for NL2SQL API endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from api import app
from src.nl2sql_agent import NL2SQLAgent

class TestNL2SQLAPI:
    """Test class for NL2SQL API"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
        self.api_key = "test-api-key-123"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def teardown_method(self):
        """Reset global agent state after each test"""
        import api
        api.nl2sql_agent = None
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Schema-Aware NL2SQL API"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "database_connected" in data
        assert "model_loaded" in data
    
    def test_unauthorized_access(self):
        """Test endpoints without API key"""
        response = self.client.post("/connect", json={"db_type": "sqlite"})
        assert response.status_code == 401
    
    def test_invalid_api_key(self):
        """Test endpoints with invalid API key"""
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.post("/connect", 
                                  json={"db_type": "sqlite"}, 
                                  headers=headers)
        assert response.status_code == 403
    
    @patch('api.NL2SQLAgent')
    def test_connect_database(self, mock_agent_class):
        """Test database connection endpoint"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.connect_database.return_value = True
        mock_agent.load_model.return_value = True
        mock_agent.current_schema = {"tables": {"books": {}}}
        mock_agent_class.return_value = mock_agent
        
        db_config = {
            "db_type": "sqlite",
            "db_path": "test.db"
        }
        
        response = self.client.post("/connect", 
                                  json=db_config, 
                                  headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "database_type" in data
    
    @patch('api.get_agent')
    def test_query_endpoint(self, mock_get_agent):
        """Test natural language query endpoint"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.process_query.return_value = {
            "success": True,
            "generated_sql": "SELECT * FROM books;",
            "results": [{"id": 1, "title": "Test Book"}],
            "confidence_score": 0.85,
            "processing_time": 1.2,
            "row_count": 1
        }
        mock_get_agent.return_value = mock_agent
        
        query_request = {
            "query": "Show all books",
            "execute": True,
            "include_schema": False
        }
        
        response = self.client.post("/query", 
                                  json=query_request, 
                                  headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "generated_sql" in data
        assert "results" in data
    
    @patch('api.get_agent')
    def test_batch_query(self, mock_get_agent):
        """Test batch query endpoint"""
        mock_agent = Mock()
        mock_agent.process_query.return_value = {
            "success": True,
            "generated_sql": "SELECT * FROM books;",
            "results": [],
            "confidence_score": 0.85
        }
        mock_get_agent.return_value = mock_agent
        
        batch_request = {
            "queries": ["Show all books", "Count books"],
            "execute": False
        }
        
        response = self.client.post("/batch-query", 
                                  json=batch_request, 
                                  headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_queries"] == 2
        assert "results" in data
    
    @patch('api.get_agent')
    def test_execute_sql(self, mock_get_agent):
        """Test direct SQL execution endpoint"""
        mock_agent = Mock()
        mock_schema_retriever = Mock()
        mock_df = Mock()
        mock_df.to_dict.return_value = [{"count": 5}]
        mock_df.empty = False
        mock_df.columns = ["count"]
        mock_schema_retriever.execute_query.return_value = mock_df
        mock_agent.schema_retriever = mock_schema_retriever
        mock_get_agent.return_value = mock_agent
        
        sql_request = {
            "sql": "SELECT COUNT(*) as count FROM books;",
            "validate_only": False
        }
        
        response = self.client.post("/execute-sql", 
                                  json=sql_request, 
                                  headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
    
    @patch('api.get_agent')
    def test_get_schema(self, mock_get_agent):
        """Test schema information endpoint"""
        mock_agent = Mock()
        mock_agent.get_schema_info.return_value = {
            "success": True,
            "database_type": "sqlite",
            "tables": ["books", "authors"],
            "total_tables": 2
        }
        mock_get_agent.return_value = mock_agent
        
        response = self.client.get("/schema", headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "database_type" in data
        assert "tables" in data
    
    @patch('api.get_agent')
    def test_get_statistics(self, mock_get_agent):
        """Test statistics endpoint"""
        mock_agent = Mock()
        mock_agent.get_statistics.return_value = {
            "total_queries": 100,
            "successful_queries": 95,
            "success_rate": 0.95
        }
        mock_get_agent.return_value = mock_agent
        
        response = self.client.get("/statistics", headers=self.headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "statistics" in data
    
    @patch('api.get_agent')
    def test_service_unavailable(self, mock_get_agent):
        """Test service unavailable when agent not initialized"""
        # Mock get_agent to raise HTTPException with 503
        from fastapi import HTTPException
        from fastapi import status
        mock_get_agent.side_effect = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NL2SQL agent not initialized. Please connect to a database first."
        )
        
        # This should fail because no agent is initialized
        query_request = {
            "query": "Show all books",
            "execute": True
        }
        
        response = self.client.post("/query", 
                                  json=query_request, 
                                  headers=self.headers)
        
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 