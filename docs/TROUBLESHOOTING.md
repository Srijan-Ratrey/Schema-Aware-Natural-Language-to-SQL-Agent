# Troubleshooting Guide

## Common Installation Issues

### 1. "Installer returned a non-zero exit code"

This error typically occurs when there are dependency conflicts or missing system requirements.

#### Solutions:

**Option A: Use the Installation Script**
```bash
python scripts/install_dependencies.py
```

**Option B: Manual Installation**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies in groups
pip install numpy pandas requests python-dotenv
pip install fastapi uvicorn streamlit pydantic
pip install sqlalchemy sqlglot psycopg2-binary
pip install torch transformers datasets accelerate
pip install plotly matplotlib seaborn openpyxl
```

**Option C: Use Virtual Environment**
```bash
# Create new virtual environment
python -m venv nl2sql_env_new
source nl2sql_env_new/bin/activate  # On Windows: nl2sql_env_new\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. PyTorch Installation Issues

PyTorch can be problematic on some systems.

#### Solutions:

**For CPU-only installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA (if you have NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. System-Specific Issues

#### macOS:
```bash
# Install system dependencies
brew install postgresql
brew install libpq

# Then install Python dependencies
pip install psycopg2-binary
```

#### Windows:
```bash
# Install Visual C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install dependencies
pip install -r requirements.txt
```

#### Linux (Ubuntu/Debian):
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev libpq-dev postgresql postgresql-contrib

# Then install Python dependencies
pip install -r requirements.txt
```

### 4. Memory Issues

If you encounter memory issues during installation:

```bash
# Install with reduced memory usage
pip install --no-cache-dir -r requirements.txt

# Or install packages one by one
pip install fastapi
pip install streamlit
pip install torch --no-cache-dir
pip install transformers --no-cache-dir
```

### 5. Version Conflicts

If you have version conflicts:

```bash
# Create a clean environment
python -m venv clean_env
source clean_env/bin/activate

# Install with specific versions
pip install fastapi==0.104.0
pip install streamlit==1.28.0
pip install torch==2.1.1
pip install transformers==4.35.2
```

### 6. Network Issues

If you're behind a proxy or have network issues:

```bash
# Use alternative package index
pip install -r requirements.txt -i https://pypi.org/simple/

# Or use conda
conda install -c conda-forge fastapi streamlit torch transformers
```

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "import fastapi, streamlit, torch, transformers, sqlalchemy; print('All imports successful!')"

# Test the application
python app.py
```

## Getting Help

If you're still having issues:

1. Check the error logs in the terminal
2. Try the installation script: `python scripts/install_dependencies.py`
3. Create a new virtual environment
4. Check your Python version (3.8+ required)
5. Ensure you have sufficient disk space (at least 5GB free)

## Common Error Messages

- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools
- **"Permission denied"**: Use `pip install --user` or activate virtual environment
- **"No module named 'torch'"**: Install PyTorch separately first
- **"SSL certificate verify failed"**: Use `pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org` 