# Custom Database Connection Guide

This guide explains how to connect your own database to the NL2SQL system.

## Supported Database Types

The NL2SQL system supports the following database types:

1. **SQLite** (.db files)
2. **PostgreSQL** (server connection)
3. **MySQL** (server connection)

## Method 1: SQLite Database

### Step 1: Prepare Your SQLite Database
1. Create or obtain your SQLite database file (`.db` extension)
2. Place the file in the project directory or provide the full path

### Step 2: Connect via Streamlit App
1. Open the Streamlit app: `streamlit run app.py`
2. In the sidebar, select "sqlite" as database type
3. Uncheck "Use Sample Database"
4. Enter the path to your database file (e.g., `my_database.db`)
5. Click "Connect to Database"

### Step 3: Verify Connection
- Check the status in the sidebar - it should show "Database: Connected"
- The app will automatically detect your tables and schema

## Method 2: PostgreSQL Database

### Prerequisites
- PostgreSQL server must be running
- Database must exist
- User must have appropriate permissions

### Step 1: Prepare Connection Details
Gather the following information:
- **Host**: Server address (usually `localhost`)
- **Port**: Server port (default: `5432`)
- **Database**: Database name
- **User**: Username
- **Password**: User password

### Step 2: Connect via Streamlit App
1. Open the Streamlit app: `streamlit run app.py`
2. In the sidebar, select "postgresql" as database type
3. Uncheck "Use Sample Database"
4. Enter your connection details:
   - Host: `localhost` (or your server address)
   - Port: `5432` (or your custom port)
   - Database: Your database name
   - User: Your username
   - Password: Your password
5. Click "Connect to Database"

## Method 3: MySQL Database

### Prerequisites
- MySQL server must be running
- Database must exist
- User must have appropriate permissions

### Step 1: Prepare Connection Details
Gather the following information:
- **Host**: Server address (usually `localhost`)
- **Port**: Server port (default: `3306`)
- **Database**: Database name
- **User**: Username
- **Password**: User password

### Step 2: Connect via Streamlit App
1. Open the Streamlit app: `streamlit run app.py`
2. In the sidebar, select "mysql" as database type
3. Uncheck "Use Sample Database"
4. Enter your connection details:
   - Host: `localhost` (or your server address)
   - Port: `3306` (or your custom port)
   - Database: Your database name
   - User: Your username
   - Password: Your password
5. Click "Connect to Database"

## Method 4: Programmatic Connection

You can also connect programmatically using the Python API:

```python
from src.nl2sql_agent import NL2SQLAgent

# Initialize agent
agent = NL2SQLAgent()

# SQLite connection
success = agent.connect_database("sqlite", db_path="path/to/your/database.db")

# PostgreSQL connection
success = agent.connect_database(
    "postgresql",
    host="localhost",
    port=5432,
    database="your_database",
    user="your_user",
    password="your_password"
)

# MySQL connection
success = agent.connect_database(
    "mysql",
    host="localhost",
    port=3306,
    database="your_database",
    user="your_user",
    password="your_password"
)

if success:
    print("✅ Connected successfully!")
else:
    print("❌ Connection failed!")
```

## Troubleshooting

### Common Issues

1. **"Database file not found" (SQLite)**
   - Ensure the file path is correct
   - Use absolute path if needed
   - Check file permissions

2. **"Connection refused" (PostgreSQL/MySQL)**
   - Verify server is running
   - Check host and port settings
   - Ensure firewall allows connections

3. **"Authentication failed"**
   - Verify username and password
   - Check user permissions
   - Ensure user can access the database

4. **"Database does not exist"**
   - Create the database first
   - Check database name spelling
   - Verify user has access to the database

### Testing Your Connection

After connecting, you can test with simple queries:

```python
# Test query
result = agent.process_query("Show me all tables")
print(result)
```

## Database Schema Requirements

For best results, ensure your database has:

1. **Clear table names** (avoid special characters)
2. **Descriptive column names** (avoid abbreviations)
3. **Proper data types** (text, numeric, date, etc.)
4. **Foreign key relationships** (for complex queries)

## Example Database Structures

### E-commerce Database
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL,
    category TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount REAL
);
```

### Employee Database
```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    salary REAL,
    hire_date DATE
);
```

## Security Considerations

1. **Never commit database credentials** to version control
2. **Use environment variables** for sensitive information
3. **Limit database user permissions** to read-only if possible
4. **Use connection pooling** for production applications
5. **Regularly backup** your databases

## Performance Tips

1. **Index important columns** for faster queries
2. **Use appropriate data types** to save space
3. **Avoid overly complex schemas** for better NL2SQL performance
4. **Consider database size** - very large databases may be slower

## Getting Help

If you encounter issues:

1. Check the error messages in the Streamlit app
2. Verify your database connection details
3. Test the connection outside the NL2SQL system
4. Check the logs for detailed error information
5. Ensure your database schema is compatible

## Next Steps

Once connected:

1. **Load the model** using the sidebar
2. **Test with simple queries** to verify everything works
3. **Explore your data** with natural language queries
4. **Fine-tune queries** based on your specific needs 