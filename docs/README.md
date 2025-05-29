# 🧠 Schema-Aware NL2SQL: Adaptive Query Generation Across Databases

## 🚀 Overview

This project implements a comprehensive **Schema-Aware Natural Language to SQL (NL2SQL) Agent** that can understand natural language questions and generate accurate SQL queries across dynamic database schemas. The system enables users to query any relational database without knowing SQL, using state-of-the-art transformer models fine-tuned on the Spider dataset.

## ✨ Features

### 🔧 Core Capabilities
- **Dynamic Schema Extraction**: Automatically extracts database schemas from SQLite, PostgreSQL, and MySQL
- **Fine-tuned T5 Models**: Uses Spider-trained models for high-quality SQL generation
- **Multi-Database Support**: Works with different SQL dialects using SQLGlot transpilation
- **Real-time Query Execution**: Validates and executes generated SQL queries safely
- **Confidence Scoring**: Provides confidence metrics for generated queries
- **Query History & Analytics**: Tracks performance and provides usage statistics

### 🎯 Web Interface Features
- **Intuitive Streamlit UI**: User-friendly interface for natural language queries
- **Schema Explorer**: Interactive database schema visualization and exploration
- **Query Validation**: Real-time SQL validation before execution
- **Data Visualization**: Automatic chart generation for query results
- **Batch Processing**: Support for multiple queries in one session
- **Example Queries**: Pre-built examples for quick testing

### 🛡️ Security & Safety
- **SQL Injection Prevention**: Validates queries before execution
- **Read-only Operations**: Focuses on SELECT queries for data safety
- **Query Complexity Analysis**: Estimates and displays query complexity
- **Error Handling**: Comprehensive error reporting and graceful failure handling

## 🧱 Architecture

```
[User Input: Natural Language]
         ↓
[Schema Retriever (SQLAlchemy)]
         ↓
[T5 Model (Fine-tuned on Spider)]
         ↓
[SQL Generation & Validation]
         ↓
[Query Execution Engine]
         ↓
[Results & Visualization]
         ↓
[Streamlit Web Interface]
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, FastAPI |
| **Models** | Transformers, T5 (Spider fine-tuned) |
| **Database** | SQLAlchemy (SQLite, PostgreSQL, MySQL) |
| **SQL Processing** | SQLGlot for dialect transpilation |
| **Frontend** | Streamlit with custom UI components |
| **Visualization** | Plotly, Pandas |
| **ML Framework** | PyTorch, Hugging Face Transformers |

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd NL_2_SQL
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the demo**
```bash
python demo.py
```

4. **Launch the web interface**
```bash
streamlit run app.py
```

### Alternative Installation with Virtual Environment

```bash
# Create virtual environment
python -m venv nl2sql_env
source nl2sql_env/bin/activate  # On Windows: nl2sql_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🚀 Usage

### 1. Web Interface (Recommended)

Launch the Streamlit application:
```bash
streamlit run app.py
```

1. **Connect to Database**: Use the sidebar to connect to your database
2. **Load Model**: Select and load a pre-trained NL2SQL model
3. **Ask Questions**: Enter natural language queries in the main interface
4. **View Results**: See generated SQL, execution results, and visualizations

### 2. Python API

```python
from src.nl2sql_agent import NL2SQLAgent

# Initialize agent
agent = NL2SQLAgent()

# Connect to database
agent.connect_database("sqlite", db_path="sample_database.db")

# Load model
agent.load_model("tscholak/t5-base-spider")

# Query the database
result = agent.query("Show all customers from New York")

print(f"Generated SQL: {result['generated_sql']}")
print(f"Results: {result['results']}")
```

### 3. Command Line Demo

```bash
python demo.py
```

This runs a comprehensive demonstration including:
- Schema extraction
- Natural language query processing
- SQL generation and execution
- Advanced features showcase

## 📊 Example Queries

The system can handle various types of natural language queries:

### Basic Queries
- "Show all employees in the Engineering department"
- "What are the top 5 most expensive products?"
- "List all customers from California"

### Aggregation Queries
- "What is the average salary by department?"
- "How many orders were placed last month?"
- "What is the total revenue for 2023?"

### Complex Queries
- "Which customer has spent the most money?"
- "Show employees hired after 2020 with salary > 50000"
- "List projects with budget over 100k that are in progress"

## 🎯 Model Options

The system supports multiple pre-trained models:

| Model | Description | Size | Performance |
|-------|-------------|------|-------------|
| `tscholak/t5-base-spider` | Original Spider fine-tuned T5 | 220M | High accuracy |
| `gaussalgo/T5-LM-Large-text2sql-spider` | Large Spider model | 770M | Highest accuracy |
| `t5-small` | Smaller T5 for fast inference | 60M | Good for testing |
| `t5-base` | Base T5 for custom fine-tuning | 220M | Customizable |

## 🗄️ Database Support

### SQLite (Default)
```python
agent.connect_database("sqlite", db_path="database.db")
```

### PostgreSQL
```python
agent.connect_database(
    "postgresql",
    host="localhost",
    port=5432,
    database="mydb",
    user="username",
    password="password"
)
```

### MySQL
```python
agent.connect_database(
    "mysql",
    host="localhost",
    port=3306,
    database="mydb",
    user="username",
    password="password"
)
```

## 🔧 Configuration

Create a `.env` file for custom configuration:

```env
# Model settings
DEFAULT_MODEL=tscholak/t5-base-spider
HUGGINGFACE_API_TOKEN=your_token_here

# Database settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Logging
LOG_LEVEL=INFO
```

## 📈 Performance & Evaluation

### Metrics Tracked
- **Exact Match (EM)**: Percentage of exactly matching SQL queries
- **Execution Accuracy**: Percentage of queries that execute successfully
- **Confidence Scores**: Model confidence in generated queries
- **Response Time**: Average query processing time

### Evaluation on Spider Dataset
The models achieve competitive performance on the Spider benchmark:
- **Execution Accuracy**: ~80% on Spider dev set
- **Exact Match**: ~65% on Spider dev set

## 🔄 Fine-tuning

To fine-tune the model on your own data:

```python
from src.nl2sql_model import NL2SQLModel, SpiderDataProcessor

# Load base model
model = NL2SQLModel("t5-base")

# Prepare your dataset
dataset = SpiderDataProcessor.prepare_spider_dataset(
    "your_data.json",
    "your_tables.json", 
    model.tokenizer
)

# Fine-tune
model.fine_tune(
    train_dataset=dataset,
    output_dir="./custom_model",
    num_epochs=5
)
```

## 🧪 Testing

Run the test suite:
```bash
# Run demo with comprehensive testing
python demo.py

# Test individual components
python -m pytest tests/
```

## 📁 Project Structure

```
NL_2_SQL/
├── src/
│   ├── __init__.py
│   ├── nl2sql_agent.py          # Main orchestrator
│   ├── nl2sql_model.py          # T5 model wrapper
│   └── schema_retriever.py      # Database schema extraction
├── app.py                       # Streamlit web interface
├── demo.py                      # Comprehensive demo script
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Key Resources

- 📘 [Spider Dataset](https://yale-lily.github.io/spider) - Training data source
- 🤖 [T5 Spider Model](https://huggingface.co/tscholak/t5-base-spider) - Pre-trained model
- ⚙️ [SQLGlot](https://github.com/tobymao/sqlglot) - SQL transpilation
- 📚 [NL2SQL Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook) - Research reference


## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure you have sufficient memory (4GB+ recommended)
   - Try using smaller models like `t5-small` for testing

2. **Database Connection Issues**
   - Verify database credentials and network connectivity
   - Check firewall settings for remote databases

3. **Query Generation Problems**
   - Ensure database schema is properly extracted
   - Try rephrasing questions more clearly
   - Check if tables/columns exist in the database

### Getting Help
- Check the demo script output for detailed error messages
- Review logs in `nl2sql_agent.log`
- Open an issue on GitHub with error details


## 🙏 Acknowledgments

- **Spider Dataset Team** at Yale University for the high-quality NL2SQL dataset
- **Hugging Face** for providing pre-trained models and infrastructure
- **SQLAlchemy** and **SQLGlot** teams for excellent SQL handling libraries
- **Streamlit** for enabling rapid web application development

---

> **"Talk to your data as you talk to a human."**  
> Making databases accessible, one query at a time. 🚀
