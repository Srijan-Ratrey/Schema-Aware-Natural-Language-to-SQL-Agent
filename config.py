"""
Configuration settings for Schema-Aware NL2SQL Agent
"""

import os
from typing import Dict, Any

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
DEFAULT_DB_CONFIG = {
    "sqlite": {
        "db_path": "database.db"
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
        "mysql_password": os.getenv("MYSQL_PASSWORD", "")
    } 