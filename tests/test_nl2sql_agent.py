#!/usr/bin/env python3
"""
Unit tests for NL2SQL Agent
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sqlite3

from src.nl2sql_agent import NL2SQLAgent
from src.nl2sql_model import NL2SQLModel
from src.schema_retriever import SchemaRetriever


class TestNL2SQLAgent:
    """Test class for NL2SQL Agent"""
    
    def setup_method(self):
        """Setup test agent"""
        self.agent = NL2SQLAgent()
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create test database schema
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                price REAL
            )
        ''')
        cursor.execute('''
            INSERT INTO books (title, author, price) VALUES
            ('Test Book 1', 'Author 1', 10.99),
            ('Test Book 2', 'Author 2', 15.99)
        ''')
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Cleanup test resources"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.model is None
        assert self.agent.schema_retriever is None
        assert self.agent.current_schema == {}
        assert self.agent.query_history == []
        assert isinstance(self.agent.statistics, dict)
    
    def test_connect_database_sqlite(self):
        """Test SQLite database connection"""
        success = self.agent.connect_database("sqlite", db_path=self.temp_db_path)
        
        assert success is True
        assert self.agent.schema_retriever is not None
        assert len(self.agent.current_schema["tables"]) > 0
        assert "books" in self.agent.current_schema["tables"]
    
    def test_connect_database_invalid_type(self):
        """Test connection with invalid database type"""
        success = self.agent.connect_database("invalid_db_type")
        assert success is False
    
    @patch('src.nl2sql_agent.NL2SQLModel')
    def test_load_model(self, mock_model_class):
        """Test model loading"""
        # Mock model
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        success = self.agent.load_model("test-model")
        
        assert success is True
        assert self.agent.model is not None
        mock_model_class.assert_called_once_with("test-model")
    
    @patch('src.nl2sql_agent.NL2SQLModel')
    def test_load_model_failure(self, mock_model_class):
        """Test model loading failure"""
        mock_model_class.side_effect = Exception("Model load failed")
        
        success = self.agent.load_model("invalid-model")
        assert success is False
    
    def test_process_query_without_setup(self):
        """Test query processing without proper setup"""
        result = self.agent.process_query("Show all books")
        
        assert result["success"] is False
        assert "error" in result
    
    @patch('src.nl2sql_agent.NL2SQLModel')
    def test_process_query_with_setup(self, mock_model_class):
        """Test query processing with proper setup"""
        # Setup database
        self.agent.connect_database("sqlite", db_path=self.temp_db_path)
        
        # Mock model
        mock_model = Mock()
        mock_model.generate_sql.return_value = {
            "sql": "SELECT * FROM books;",
            "confidence": 0.85
        }
        mock_model_class.return_value = mock_model
        self.agent.load_model("test-model")
        
        # Process query
        result = self.agent.process_query("Show all books")
        
        assert result["success"] is True
        assert "generated_sql" in result
        assert "results" in result
        assert result["row_count"] == 2  # Based on test data
    
    def test_get_schema_info_without_connection(self):
        """Test schema info without database connection"""
        schema_info = self.agent.get_schema_info()
        
        assert schema_info["success"] is False
        assert "error" in schema_info
    
    def test_get_schema_info_with_connection(self):
        """Test schema info with database connection"""
        self.agent.connect_database("sqlite", db_path=self.temp_db_path)
        
        schema_info = self.agent.get_schema_info()
        
        assert schema_info["success"] is True
        assert schema_info["database_type"] == "sqlite"
        assert "books" in schema_info["tables"]
        assert schema_info["total_tables"] == 1
    
    def test_get_schema_info_specific_table(self):
        """Test schema info for specific table"""
        self.agent.connect_database("sqlite", db_path=self.temp_db_path)
        
        schema_info = self.agent.get_schema_info("books")
        
        assert schema_info["success"] is True
        assert "table_details" in schema_info
    
    def test_query_history(self):
        """Test query history tracking"""
        # Initially empty
        history = self.agent.get_query_history()
        assert len(history) == 0
        
        # Add query to history
        self.agent._add_to_history("Show all books", "SELECT * FROM books;", True, 2)
        
        history = self.agent.get_query_history()
        assert len(history) == 1
        assert history[0]["query"] == "Show all books"
        assert history[0]["success"] is True
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        # Initial statistics
        stats = self.agent.get_statistics()
        assert stats["total_queries"] == 0
        assert stats["successful_queries"] == 0
        
        # Add some queries to history
        self.agent._add_to_history("Query 1", "SQL 1", True, 1)
        self.agent._add_to_history("Query 2", "SQL 2", False, 0)
        self.agent._add_to_history("Query 3", "SQL 3", True, 5)
        
        stats = self.agent.get_statistics()
        assert stats["total_queries"] == 3
        assert stats["successful_queries"] == 2
        assert stats["failed_queries"] == 1
        assert stats["success_rate"] == 2/3
    
    def test_close_connections(self):
        """Test closing database connections"""
        self.agent.connect_database("sqlite", db_path=self.temp_db_path)
        assert self.agent.schema_retriever is not None
        
        self.agent.close_connections()
        # After closing, the retriever should still exist but connection should be handled properly
        # (SQLAlchemy manages connection pooling)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 