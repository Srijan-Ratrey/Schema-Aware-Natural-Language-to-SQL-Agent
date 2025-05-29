# 🚀 Setting Up Schema-Aware NL2SQL in a New Environment

This guide will help you set up the Schema-Aware NL2SQL project in a fresh Python environment.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 4GB of available RAM (for model loading)
- Internet connection (for downloading models and packages)

## 🎯 Quick Setup (Automated)

### Option 1: Run the Setup Script
```bash
python setup_new_environment.py
```

This automated script will:
- ✅ Check Python version compatibility
- ✅ Create a new virtual environment
- ✅ Install all required dependencies
- ✅ Verify the installation
- ✅ Provide usage instructions

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment
```bash
# Create a new virtual environment
python -m venv nl2sql_env

# Activate the environment
# On Windows:
nl2sql_env\Scripts\activate
# On macOS/Linux:
source nl2sql_env/bin/activate
```

#### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Install PyTorch first (large dependency)
pip install torch==2.1.1

# Install all other dependencies
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
python -c "import torch, transformers, streamlit; print('✅ Installation successful!')"
```

## 🧪 Testing the Installation

### Quick Test
```bash
# Activate your environment first!
python quickstart.py
```

### Web Interface Test
```bash
streamlit run app.py
```

### Comprehensive Demo
```bash
python demo.py
```

## 📦 Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.1.1 | Deep learning framework |
| `transformers` | 4.35.2 | Hugging Face models (T5) |
| `streamlit` | 1.28.1 | Web interface |
| `sqlalchemy` | 2.0.23 | Database connections |
| `pandas` | 2.1.3 | Data manipulation |
| `sqlglot` | 18.17.0 | SQL dialect transpilation |
| `plotly` | 5.17.0 | Data visualization |
| `datasets` | 2.14.6 | Dataset processing |
| `accelerate` | 0.24.1 | Model acceleration |

## 🔧 Environment Management

### Activate Environment
```bash
# Windows
nl2sql_env\Scripts\activate

# macOS/Linux  
source nl2sql_env/bin/activate
```

### Deactivate Environment
```bash
deactivate
```

### Delete Environment
```bash
# Deactivate first, then remove folder
deactivate
rm -rf nl2sql_env  # Linux/macOS
# or
rmdir /s nl2sql_env  # Windows
```

## 🐛 Troubleshooting

### Common Issues

#### 1. PyTorch Installation Fails
```bash
# Try CPU-only version
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Memory Issues During Model Loading
- Use a smaller model: `t5-small` instead of `t5-base-spider`
- Close other applications to free up RAM
- Consider using a machine with more memory

#### 3. Database Connection Issues
- For SQLite: Ensure the database file exists and has proper permissions
- For PostgreSQL/MySQL: Check connection parameters and firewall settings

#### 4. Import Errors
```bash
# Verify your environment is activated
which python  # Should point to your virtual environment

# Reinstall problematic packages
pip uninstall package_name
pip install package_name
```

### Performance Optimization

#### For CPU-only Systems
```bash
# Install CPU-optimized PyTorch
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

#### For GPU Systems
```bash
# Install CUDA-enabled PyTorch (if you have NVIDIA GPU)
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

## 📂 Project Structure

```
NL_2_SQL/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── nl2sql_agent.py          # Main orchestrator
│   ├── nl2sql_model.py          # T5 model wrapper
│   └── schema_retriever.py      # Database schema extraction
├── app.py                       # Streamlit web interface
├── demo.py                      # Comprehensive demo
├── quickstart.py               # Quick start script
├── setup_new_environment.py    # Automated setup
├── requirements.txt            # Dependencies
├── config.py                   # Configuration
└── README.md                   # Documentation
```

## 🎉 Next Steps

Once your environment is set up:

1. **Start with Quick Test**: `python quickstart.py`
2. **Try the Web Interface**: `streamlit run app.py`
3. **Connect Your Database**: Use the sidebar in the web app
4. **Ask Questions**: Start with simple queries to test functionality

## 🤝 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify all dependencies are installed correctly
3. Ensure you have sufficient system resources
4. Check the logs for detailed error messages

## 🔄 Updating the Environment

To update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

To add new dependencies:
```bash
pip install new_package
pip freeze > requirements.txt  # Update requirements file
```

---

**Happy querying! 🚀** 