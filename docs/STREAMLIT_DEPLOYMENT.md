# Streamlit Cloud Deployment Guide

## Quick Setup

1. **Fork or upload your repository to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select your repository**
5. **Set the main file path to: `app.py`**
6. **Set the Python version to: 3.10**
7. **Deploy!**

## Configuration Files

### requirements.txt
The main requirements file includes all dependencies. If you encounter issues, try using `requirements-streamlit.txt` instead.

### .streamlit/config.toml
This file configures Streamlit for production deployment.

## Common Issues and Solutions

### 1. "Installer returned a non-zero exit code"

**Solution:** Use the simplified requirements file:
- Rename `requirements-streamlit.txt` to `requirements.txt`
- Remove problematic dependencies like `psycopg2-binary`, `gradio`, `openpyxl`, `sentencepiece`

### 2. Memory Issues

**Solution:** Streamlit Cloud has memory limits. The app will work with:
- Smaller models
- CPU-only PyTorch
- Reduced batch sizes

### 3. Database Connection Issues

**Solution:** For Streamlit Cloud:
- Use SQLite databases (included in the repo)
- Don't use external database connections
- Store data files in the `data/` directory

## Optimized for Streamlit Cloud

### File Structure
```
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .streamlit/config.toml    # Streamlit config
├── data/                     # Database files
│   └── sample_database.db
├── src/                      # Source code
└── docs/                     # Documentation
```

### Key Changes for Deployment

1. **Database**: Use SQLite only (no PostgreSQL/MySQL)
2. **Models**: Use smaller, CPU-optimized models
3. **Dependencies**: Remove heavy packages
4. **Configuration**: Use production settings

## Testing Locally

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment Checklist

- [ ] All dependencies in requirements.txt
- [ ] Database files in data/ directory
- [ ] No external API keys in code
- [ ] Streamlit config file present
- [ ] Main file path set to app.py
- [ ] Python version set to 3.10

## Troubleshooting

### If deployment fails:

1. **Check the logs** in Streamlit Cloud dashboard
2. **Try the simplified requirements** file
3. **Remove heavy dependencies** temporarily
4. **Test with a minimal app** first

### Common error messages:

- **"No module named X"**: Add missing dependency to requirements.txt
- **"Permission denied"**: Check file permissions
- **"Memory limit exceeded"**: Reduce model size or dependencies
- **"Database connection failed"**: Use SQLite only

## Performance Tips

1. **Use smaller models** for faster loading
2. **Cache expensive operations** with `@st.cache_data`
3. **Load models once** and reuse
4. **Use pagination** for large datasets
5. **Optimize database queries**

## Support

If you continue to have issues:
1. Check the Streamlit Cloud documentation
2. Review the error logs carefully
3. Try deploying a minimal version first
4. Contact Streamlit support if needed 