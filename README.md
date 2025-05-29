# 🧠 Schema-Aware NL2SQL: Production-Ready Natural Language to SQL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![API Docs](https://img.shields.io/badge/API-Documentation-green.svg)](docs/API_DOCUMENTATION.md)

## 🚀 Overview

A comprehensive **Schema-Aware Natural Language to SQL (NL2SQL) system** that converts natural language questions into accurate SQL queries across dynamic database schemas. Features both a web interface and production-ready REST API with deployment support for any cloud platform.

## ✨ Key Features

- 🧠 **Schema-Aware Intelligence**: Dynamic schema extraction and understanding
- 🔄 **Multi-Database Support**: SQLite, PostgreSQL, MySQL with dialect transpilation
- 🌐 **Production API**: Complete REST API with authentication and monitoring
- 🖥️ **Web Interface**: Intuitive Streamlit UI for interactive querying
- 🚀 **Cloud Ready**: Docker, Kubernetes, and multi-cloud deployment support
- 🔒 **Security First**: API authentication, SQL injection prevention, query validation
- 📊 **Analytics**: Query history, confidence scoring, and usage statistics
- 🧪 **Fully Tested**: Comprehensive test suite with CI/CD ready structure

## 📁 Complete Project Structure

```
Schema-Aware-NL2SQL/
├── 📄 README.md                       # Main documentation
├── 📄 requirements.txt                # Python dependencies
├── 📄 setup.py                        # Package setup
├── 📄 config.py                       # Configuration management
├── 📄 .env.example                    # Environment template
├── 📄 .gitignore                      # Git ignore rules
│
├── 🔧 api.py                          # FastAPI REST API server
├── 🖥️ app.py                          # Streamlit web interface
├── 🎯 demo.py                         # Comprehensive demo script
│
├── 📂 src/                            # Core source code
│   ├── __init__.py
│   ├── nl2sql_agent.py                # Main orchestrator
│   ├── nl2sql_model.py                # T5 model wrapper
│   └── schema_retriever.py            # Database schema extraction
│
├── 📂 docs/                           # Documentation
│   ├── README.md                      # Detailed documentation
│   ├── API_DOCUMENTATION.md           # API reference
│   ├── SETUP_COMPLETE.md              # Setup guide
│   ├── ENVIRONMENT_SETUP.md           # Environment guide
│   └── GITHUB_SETUP.md                # GitHub integration
│
├── 📂 examples/                       # Example scripts
│   ├── quickstart.py                  # Quick start demo
│   └── client_example.py              # API client example
│
├── 📂 tests/                          # Test suite
│   ├── __init__.py
│   ├── test_api.py                    # API endpoint tests
│   └── test_nl2sql_agent.py           # Core functionality tests
│
├── 📂 scripts/                        # Utility scripts
│   ├── deploy.sh                      # Deployment automation
│   ├── run_tests.sh                   # Test runner
│   └── setup_new_environment.py       # Environment setup
│
├── 📂 deployment/                     # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile                 # Container definition
│   │   └── docker-compose.yml         # Multi-service orchestration
│   ├── kubernetes/
│   │   └── deployment.yaml            # K8s deployment config
│   └── cloud/
│       └── aws-ecs-task.json          # AWS ECS task definition
│
├── 📂 data/                           # Database files
│   └── quickstart_sample.db           # Sample SQLite database
│
├── 📂 models/                         # Model cache (auto-created)
├── 📂 logs/                           # Application logs (auto-created)
└── 📂 nl2sql_env/                     # Virtual environment
```

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/Srijan-Ratrey/Schema-Aware-Natural-Language-to-SQL-Agent.git
cd Schema-Aware-Natural-Language-to-SQL-Agent

# Quick setup with deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh dev
```

### 2. Start the Web Interface

```bash
streamlit run app.py
```

### 3. Start the API Server

```bash
python api.py
```
Access API documentation at: `http://localhost:8000/docs`

## 🌐 API Quick Start

```python
import requests

# API configuration
API_BASE = "http://localhost:8000"
API_KEY = "your-api-key-here"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Connect to database
requests.post(f"{API_BASE}/connect", 
    json={"db_type": "sqlite", "db_path": "data/quickstart_sample.db"},
    headers=headers
)

# Query database
response = requests.post(f"{API_BASE}/query",
    json={"query": "Show all books with rating above 4.5"},
    headers=headers
)

print(response.json())
```

## 🚀 Deployment Options

### Development
```bash
./scripts/deploy.sh dev
```

### Docker
```bash
./scripts/deploy.sh docker
```

### Docker Compose (Full Stack)
```bash
./scripts/deploy.sh compose
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### Cloud Platforms
- **AWS ECS**: Use `deployment/cloud/aws-ecs-task.json`
- **Google Cloud Run**: Build with Docker and deploy
- **Azure Container Instances**: Deploy with Docker image
- **Heroku**: Deploy with git push

## 🧪 Testing

```bash
# Run comprehensive test suite
./scripts/run_tests.sh

# Run specific tests
python -m pytest tests/test_api.py -v
python -m pytest tests/test_nl2sql_agent.py -v
```

## 📊 Features Overview

### Core Capabilities
- ✅ Dynamic schema extraction and understanding
- ✅ Fine-tuned T5 models (Spider dataset trained)
- ✅ Multi-database support (SQLite, PostgreSQL, MySQL)
- ✅ Real-time SQL generation and execution
- ✅ Confidence scoring and query validation
- ✅ Query history and analytics

### Web Interface Features
- ✅ Interactive Streamlit UI
- ✅ Schema visualization
- ✅ Query result visualization
- ✅ Batch query processing
- ✅ Export capabilities

### API Features
- ✅ RESTful API with OpenAPI documentation
- ✅ Bearer token authentication
- ✅ Rate limiting and security
- ✅ Batch query processing
- ✅ Health monitoring
- ✅ Comprehensive error handling

### Production Features
- ✅ Docker containerization
- ✅ Kubernetes deployment
- ✅ Multi-cloud support
- ✅ Logging and monitoring
- ✅ Auto-scaling ready
- ✅ Security best practices

## 🛡️ Security Features

- 🔐 API key authentication
- 🛡️ SQL injection prevention
- ✅ Query validation and sanitization
- 🔒 Read-only query enforcement
- 📊 Rate limiting and monitoring
- 🔍 Comprehensive logging

## 📈 Performance

- ⚡ Optimized T5 model inference
- 🚀 Async API endpoints
- 💾 Schema caching
- 📊 Query result caching
- 🔄 Connection pooling
- 📈 Horizontal scaling support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `./scripts/run_tests.sh`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open Pull Request

## 📚 Documentation

- 📖 [Complete Setup Guide](docs/SETUP_COMPLETE.md)
- 🔗 [API Documentation](docs/API_DOCUMENTATION.md)
- 🐳 [Deployment Guide](scripts/deploy.sh)
- 🧪 [Testing Guide](scripts/run_tests.sh)
- 🔧 [Environment Setup](docs/ENVIRONMENT_SETUP.md)

## 🔗 Related Resources

- 📘 [Spider Dataset](https://yale-lily.github.io/spider) - Training data
- 🤖 [Hugging Face Models](https://huggingface.co/models?search=text2sql) - Pre-trained models
- ⚙️ [SQLGlot](https://github.com/tobymao/sqlglot) - SQL transpilation
- 📚 [NL2SQL Papers](https://github.com/HKUSTDial/NL2SQL_Handbook) - Research

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spider Dataset Team** for high-quality NL2SQL benchmarks
- **Hugging Face** for transformer models and infrastructure
- **FastAPI & Streamlit** teams for excellent frameworks
- **SQLAlchemy & SQLGlot** for robust SQL handling

---

**🌟 Star this repo if you find it useful!**

> "Making databases conversational, one query at a time." 🚀 