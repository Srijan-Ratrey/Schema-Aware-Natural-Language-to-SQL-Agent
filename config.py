"""
Configuration settings for Schema-Aware NL2SQL Agent
"""

import os
from typing import Dict, Any
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Model Configuration
DEFAULT_MODEL = "mrm8488/t5-base-finetuned-wikiSQL"
ALTERNATIVE_MODELS = [
    "mrm8488/t5-base-finetuned-wikiSQL",  # WikiSQL trained T5
    "tscholak/cxmefzzi",                  # Spider T5-3B (large)
    "tscholak/1zha5ono",                  # Spider T5-base
    "t5-small",                           # Standard T5 small
    "t5-base"                             # Standard T5 base
]

# Database Configuration
DATABASE_CONFIG = {
    'sqlite': {
        'database_path': PROJECT_ROOT / 'data' / 'sample_database.db',
        'connection_string': f"sqlite:///{PROJECT_ROOT / 'data' / 'sample_database.db'}"
    },
    "postgresql": {
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",
        "password": ""
    },
    "mysql": {
        "host": "localhost", 
        "port": 3306,
        "database": "mysql",
        "user": "root",
        "password": ""
    }
}

# Default database type
DEFAULT_DATABASE_TYPE = 'sqlite'

# API configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_DEBUG = os.getenv('API_DEBUG', 'True').lower() == 'true'

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = PROJECT_ROOT / 'logs' / 'nl2sql.log'

# Model Generation Parameters
GENERATION_CONFIG = {
    "max_length": 512,
    "num_beams": 4,
    "temperature": 0.1,
    "do_sample": False,
    "early_stopping": True
}

# Training Configuration (for fine-tuning)
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100
}

# Web Application Configuration
STREAMLIT_CONFIG = {
    "page_title": "Schema-Aware NL2SQL Agent",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "nl2sql_agent.log"
}

# Cache Configuration
CACHE_CONFIG = {
    "model_cache_dir": "./models",
    "schema_cache_ttl": 300,  # 5 minutes
    "query_history_limit": 100
}

# Security Configuration
SECURITY_CONFIG = {
    "max_query_length": 1000,
    "allowed_sql_keywords": [
        "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", 
        "HAVING", "LIMIT", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER"
    ],
    "forbidden_sql_keywords": [
        "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"
    ]
}

# Database schema information for NL2SQL
DATABASE_SCHEMA = {
    'tables': {
        'Customer': {
            'description': 'Customer information including personal details and contact information',
            'columns': {
                'Id': 'Primary key, unique customer identifier',
                'FirstName': 'Customer first name',
                'LastName': 'Customer last name', 
                'City': 'City where customer is located',
                'Country': 'Country where customer is located',
                'Phone': 'Customer phone number'
            }
        },
        'Supplier': {
            'description': 'Supplier information for products',
            'columns': {
                'Id': 'Primary key, unique supplier identifier',
                'CompanyName': 'Name of the supplier company',
                'ContactName': 'Contact person name',
                'City': 'City where supplier is located',
                'Country': 'Country where supplier is located',
                'Phone': 'Supplier phone number',
                'Fax': 'Supplier fax number'
            }
        },
        'Product': {
            'description': 'Product catalog with pricing and supplier information',
            'columns': {
                'Id': 'Primary key, unique product identifier',
                'ProductName': 'Name of the product',
                'SupplierId': 'Foreign key to Supplier table',
                'UnitPrice': 'Price per unit of the product',
                'Package': 'Product packaging description',
                'IsDiscontinued': 'Boolean indicating if product is discontinued'
            }
        },
        'Order': {
            'description': 'Customer orders with order details',
            'columns': {
                'Id': 'Primary key, unique order identifier',
                'OrderDate': 'Date when order was placed',
                'OrderNumber': 'Order reference number',
                'CustomerId': 'Foreign key to Customer table',
                'TotalAmount': 'Total amount for the order'
            }
        },
        'OrderItem': {
            'description': 'Individual items within orders linking products to orders',
            'columns': {
                'Id': 'Primary key, unique order item identifier',
                'OrderId': 'Foreign key to Order table',
                'ProductId': 'Foreign key to Product table',
                'UnitPrice': 'Price per unit at time of order',
                'Quantity': 'Quantity of product ordered'
            }
        }
    },
    'relationships': [
        'Product.SupplierId -> Supplier.Id',
        'Order.CustomerId -> Customer.Id', 
        'OrderItem.OrderId -> Order.Id',
        'OrderItem.ProductId -> Product.Id'
    ]
}

def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    return {
        "huggingface_token": os.getenv("HUGGINGFACE_API_TOKEN"),
        "default_model": os.getenv("DEFAULT_MODEL", DEFAULT_MODEL),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "sqlite_db_path": os.getenv("SQLITE_DB_PATH", "database.db"),
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", 5432)),
        "postgres_db": os.getenv("POSTGRES_DB", "postgres"),
        "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
        "mysql_host": os.getenv("MYSQL_HOST", "localhost"),
        "mysql_port": int(os.getenv("MYSQL_PORT", 3306)),
        "mysql_db": os.getenv("MYSQL_DB", "mysql"),
        "mysql_user": os.getenv("MYSQL_USER", "root"),
        "mysql_password": os.getenv("MYSQL_PASSWORD", ""),
        "api_host": API_HOST,
        "api_port": API_PORT,
        "api_debug": API_DEBUG,
        "log_file": LOG_FILE
    } 