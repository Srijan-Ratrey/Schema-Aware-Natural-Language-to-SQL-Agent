#!/usr/bin/env python3
"""
Setup Script for Schema-Aware NL2SQL Project in New Environment
This script helps you set up the project in a fresh Python environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def create_virtual_environment(env_name="nl2sql_env"):
    """Create a new virtual environment"""
    if os.path.exists(env_name):
        print(f"⚠️  Environment '{env_name}' already exists")
        response = input("Do you want to remove it and create a new one? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(env_name)
        else:
            return env_name
    
    success = run_command(f"python -m venv {env_name}", f"Creating virtual environment '{env_name}'")
    return env_name if success else None


def get_activation_command(env_name, os_type):
    """Get the activation command for the virtual environment"""
    if os_type == "windows":
        return f"{env_name}\\Scripts\\activate"
    else:  # Unix-like (Linux, macOS)
        return f"source {env_name}/bin/activate"


def install_dependencies(env_name):
    """Install project dependencies"""
    # Determine OS type
    import platform
    os_type = "windows" if platform.system() == "Windows" else "unix"
    
    # Get Python executable path in virtual environment
    if os_type == "windows":
        python_exe = f"{env_name}\\Scripts\\python"
        pip_exe = f"{env_name}\\Scripts\\pip"
    else:
        python_exe = f"{env_name}/bin/python"
        pip_exe = f"{env_name}/bin/pip"
    
    # Upgrade pip first
    success = run_command(f"{python_exe} -m pip install --upgrade pip", "Upgrading pip")
    if not success:
        return False
    
    # Install PyTorch first (it's a large dependency)
    print("📦 Installing PyTorch (this may take a while)...")
    success = run_command(f"{pip_exe} install torch==2.1.1", "Installing PyTorch")
    if not success:
        print("⚠️  PyTorch installation failed. Trying CPU-only version...")
        success = run_command(f"{pip_exe} install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU-only)")
        if not success:
            return False
    
    # Install other dependencies
    deps_to_install = [
        "transformers==4.35.2",
        "datasets==2.14.6", 
        "streamlit==1.28.1",
        "sqlalchemy==2.0.23",
        "pandas==2.1.3",
        "sqlglot==18.17.0",
        "psycopg2-binary==2.9.9",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "huggingface-hub==0.19.4",
        "accelerate==0.24.1",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "gradio==4.7.1",
        "plotly==5.17.0",  # Added for visualization
        "openpyxl==3.1.2"  # Added for Excel support
    ]
    
    for dep in deps_to_install:
        success = run_command(f"{pip_exe} install {dep}", f"Installing {dep.split('==')[0]}")
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing with others...")
    
    return True


def verify_installation(env_name):
    """Verify that the installation was successful"""
    import platform
    os_type = "windows" if platform.system() == "Windows" else "unix"
    
    if os_type == "windows":
        python_exe = f"{env_name}\\Scripts\\python"
    else:
        python_exe = f"{env_name}/bin/python"
    
    # Test imports
    test_script = '''
import sys
try:
    import torch
    import transformers
    import streamlit
    import sqlalchemy
    import pandas
    print("✅ All major dependencies imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Streamlit version: {streamlit.__version__}")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    success = run_command(f"{python_exe} test_imports.py", "Testing imports")
    os.remove("test_imports.py")
    
    return success


def main():
    """Main setup function"""
    print("🧠 Schema-Aware NL2SQL Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher and try again.")
        return 1
    
    # Get environment name
    env_name = input("Enter virtual environment name (default: nl2sql_env): ").strip()
    if not env_name:
        env_name = "nl2sql_env"
    
    # Create virtual environment
    env_name = create_virtual_environment(env_name)
    if not env_name:
        print("Failed to create virtual environment")
        return 1
    
    # Install dependencies
    if not install_dependencies(env_name):
        print("Failed to install some dependencies")
        return 1
    
    # Verify installation
    if not verify_installation(env_name):
        print("Installation verification failed")
        return 1
    
    # Print success message and instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    import platform
    os_type = "windows" if platform.system() == "Windows" else "unix"
    activation_cmd = get_activation_command(env_name, os_type)
    
    print(f"\nTo activate your new environment:")
    if os_type == "windows":
        print(f"   {env_name}\\Scripts\\activate")
    else:
        print(f"   source {env_name}/bin/activate")
    
    print(f"\nTo test the installation:")
    print(f"   python quickstart.py")
    print(f"   # or")
    print(f"   streamlit run app.py")
    
    print(f"\nTo deactivate the environment:")
    print(f"   deactivate")
    
    print("\n📁 Project Structure:")
    print("   src/               - Core NL2SQL modules")
    print("   app.py            - Streamlit web interface") 
    print("   demo.py           - Comprehensive demo")
    print("   quickstart.py     - Quick start script")
    print("   requirements.txt  - Dependencies list")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 