"""
Setup script for Schema-Aware NL2SQL Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="schema-aware-nl2sql",
    version="1.0.0",
    author="NL2SQL Team",
    author_email="contact@nl2sql.com",
    description="Schema-Aware Natural Language to SQL Agent with Fine-tuned T5 Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/schema-aware-nl2sql",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "streamlit>=1.28.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nl2sql-demo=demo:main",
            "nl2sql-app=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    keywords=[
        "natural language processing",
        "sql generation", 
        "database querying",
        "machine learning",
        "transformers",
        "t5",
        "spider dataset",
        "text-to-sql",
        "schema-aware"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/schema-aware-nl2sql/issues",
        "Source": "https://github.com/your-username/schema-aware-nl2sql",
        "Documentation": "https://github.com/your-username/schema-aware-nl2sql/blob/main/README.md",
    },
) 