"""
Streamlit Web Application for Schema-Aware NL2SQL Agent
Provides an intuitive interface for natural language to SQL conversion
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import json
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our custom modules
from src.nl2sql_agent import NL2SQLAgent

# Page configuration
st.set_page_config(
    page_title="🧠 Schema-Aware NL2SQL Agent",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def create_sample_database():
    """Create a sample SQLite database for demonstration"""
    db_path = "sample_ecommerce.db"
    
    if os.path.exists(db_path):
        return db_path
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            city TEXT,
            country TEXT,
            registration_date DATE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            price DECIMAL(10, 2),
            stock_quantity INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            total_amount DECIMAL(10, 2),
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price DECIMAL(10, 2),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)
    
    # Insert sample data
    customers_data = [
        (1, 'John', 'Doe', 'john.doe@email.com', 'New York', 'USA', '2023-01-15'),
        (2, 'Jane', 'Smith', 'jane.smith@email.com', 'London', 'UK', '2023-02-20'),
        (3, 'Bob', 'Johnson', 'bob.johnson@email.com', 'Toronto', 'Canada', '2023-03-10'),
        (4, 'Alice', 'Brown', 'alice.brown@email.com', 'Sydney', 'Australia', '2023-04-05'),
        (5, 'Charlie', 'Wilson', 'charlie.wilson@email.com', 'Berlin', 'Germany', '2023-05-12')
    ]
    
    products_data = [
        (1, 'Laptop', 'Electronics', 999.99, 50),
        (2, 'Smartphone', 'Electronics', 699.99, 100),
        (3, 'Headphones', 'Electronics', 199.99, 75),
        (4, 'Book', 'Education', 29.99, 200),
        (5, 'Coffee Mug', 'Home', 15.99, 150)
    ]
    
    orders_data = [
        (1, 1, '2023-06-01', 1199.98, 'Completed'),
        (2, 2, '2023-06-02', 699.99, 'Completed'),
        (3, 3, '2023-06-03', 245.97, 'Processing'),
        (4, 1, '2023-06-04', 29.99, 'Shipped'),
        (5, 4, '2023-06-05', 715.98, 'Completed')
    ]
    
    order_items_data = [
        (1, 1, 1, 1, 999.99),
        (2, 1, 3, 1, 199.99),
        (3, 2, 2, 1, 699.99),
        (4, 3, 3, 1, 199.99),
        (5, 3, 4, 1, 29.99),
        (6, 3, 5, 1, 15.99),
        (7, 4, 4, 1, 29.99),
        (8, 5, 2, 1, 699.99),
        (9, 5, 5, 1, 15.99)
    ]
    
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers_data)
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders_data)
    cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", order_items_data)
    
    conn.commit()
    conn.close()
    
    return db_path


def sidebar_database_connection():
    """Database connection sidebar"""
    st.sidebar.header("🗄️ Database Connection")
    
    db_type = st.sidebar.selectbox(
        "Database Type:",
        ["sqlite", "postgresql", "mysql"],
        help="Choose your database type"
    )
    
    # Sample database option
    use_sample = st.sidebar.checkbox("Use Sample Database", value=True, help="Use built-in sample database for testing")
    
    if use_sample:
        if db_type == "sqlite":
            # Sample database selection
            sample_db = st.sidebar.selectbox(
                "Select Sample Database:",
                ["sample_database.db", "sample_ecommerce.db", "demo_company.db"],
                help="Choose between different sample databases"
            )
            
            # Show database description
            if sample_db == "sample_database.db":
                st.sidebar.info("📊 Working Database: Customer, Product, Supplier, Order, OrderItem")
            elif sample_db == "sample_ecommerce.db":
                st.sidebar.info("📊 E-commerce Database: customers, products, orders, order_items")
            else:
                st.sidebar.info("🏢 Company Database: employees, departments, projects")
            
            # Check if sample database exists
            if sample_db == "sample_database.db":
                # Try multiple possible paths
                possible_paths = [
                    "data/sample_database.db",
                    "./data/sample_database.db",
                    os.path.join(os.getcwd(), "data", "sample_database.db"),
                    os.path.join(os.path.dirname(__file__), "data", "sample_database.db")
                ]
                
                db_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        db_path = path
                        break
                
                if db_path:
                    st.sidebar.success(f"✅ {sample_db} found at {db_path}!")
                else:
                    st.sidebar.error(f"Sample database {sample_db} not found. Tried paths: {possible_paths}")
                    return
            elif sample_db == "sample_ecommerce.db":
                if not os.path.exists(sample_db):
                    create_sample_database()
                st.sidebar.success(f"✅ {sample_db} found!")
            else:
                if not os.path.exists(sample_db):
                    st.sidebar.error(f"Sample database {sample_db} not found")
                    return
                else:
                    st.sidebar.success(f"✅ {sample_db} found!")
            
            # Connect to the sample database
            if st.sidebar.button("Connect to Sample Database"):
                try:
                    if not st.session_state.agent:
                        st.session_state.agent = NL2SQLAgent()
                    
                    # Use the correct database path
                    if sample_db == "sample_database.db":
                        # Use the path we found earlier
                        possible_paths = [
                            "data/sample_database.db",
                            "./data/sample_database.db",
                            os.path.join(os.getcwd(), "data", "sample_database.db"),
                            os.path.join(os.path.dirname(__file__), "data", "sample_database.db")
                        ]
                        
                        db_path = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                db_path = path
                                break
                        
                        if not db_path:
                            st.sidebar.error("Database path not found")
                            return
                    elif sample_db == "sample_ecommerce.db":
                        db_path = sample_db
                    else:
                        db_path = sample_db
                    
                    success = st.session_state.agent.connect_database("sqlite", database_path=db_path)
                    if success:
                        st.session_state.connected = True
                        st.sidebar.success("✅ Connected to sample database!")
                    else:
                        st.sidebar.error("❌ Failed to connect to sample database")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection error: {str(e)}")
        else:
            st.sidebar.warning("Sample databases only available for SQLite")
            return
    else:
        # Custom database connection
        st.sidebar.subheader("📁 Custom Database")
        
        if db_type == "sqlite":
            db_path = st.sidebar.text_input(
                "Database File Path:", 
                value="your_database.db",
                help="Path to your SQLite database file (e.g., /path/to/your/database.db)"
            )
            
            if st.sidebar.button("Connect to Database"):
                if os.path.exists(db_path):
                    if not st.session_state.agent:
                        st.session_state.agent = NL2SQLAgent()
                    
                    success = st.session_state.agent.connect_database("sqlite", database_path=db_path)
                    if success:
                        st.session_state.connected = True
                        st.sidebar.success("✅ Connected to database!")
                    else:
                        st.sidebar.error("❌ Failed to connect to database")
                else:
                    st.sidebar.error(f"❌ Database file not found: {db_path}")
        
        elif db_type == "postgresql":
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value="localhost")
                port = st.number_input("Port", value=5432)
                database = st.text_input("Database", value="your_database")
            with col2:
                user = st.text_input("User", value="postgres")
                password = st.text_input("Password", type="password")
            
            if st.button("Connect to Database"):
                try:
                    success = st.session_state.agent.connect_database(
                        "postgresql",
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password
                    )
                    if success:
                        st.session_state.connected = True
                        st.sidebar.success("✅ Connected to database!")
                    else:
                        st.sidebar.error("❌ Failed to connect to database")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection error: {str(e)}")
        
        elif db_type == "mysql":
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value="localhost")
                port = st.number_input("Port", value=3306)
                database = st.text_input("Database", value="your_database")
            with col2:
                user = st.text_input("User", value="root")
                password = st.text_input("Password", type="password")
            
            if st.button("Connect to Database"):
                try:
                    success = st.session_state.agent.connect_database(
                        "mysql",
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password
                    )
                    if success:
                        st.session_state.connected = True
                        st.sidebar.success("✅ Connected to database!")
                    else:
                        st.sidebar.error("❌ Failed to connect to database")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection error: {str(e)}")
    
    # Database connection instructions
    with st.sidebar.expander("📖 How to Connect Your Database"):
        st.markdown("""
        **SQLite Database:**
        1. Place your `.db` file in the project directory
        2. Enter the filename (e.g., `my_database.db`)
        3. Click "Connect to Database"
        
        **PostgreSQL Database:**
        1. Ensure PostgreSQL server is running
        2. Enter your connection details
        3. Make sure the database exists
        4. Click "Connect to Database"
        
        **MySQL Database:**
        1. Ensure MySQL server is running
        2. Enter your connection details
        3. Make sure the database exists
        4. Click "Connect to Database"
        
        **Supported Database Formats:**
        - SQLite (.db files)
        - PostgreSQL (server connection)
        - MySQL (server connection)
        """)
    
    # Model loading section
    st.sidebar.header("🤖 Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Select Model:",
        [
            "gaussalgo/T5-LM-Large-text2sql-spider"
        ],
        index=0
    )
    
    # Enhanced prompt engineering options
    st.sidebar.subheader("🧠 Prompt Engineering")
    
    use_enhanced_prompts = st.sidebar.checkbox(
        "Use Enhanced Prompts", 
        value=st.session_state.get('use_enhanced_prompts', True),
        help="Use advanced prompt engineering for better SQL generation"
    )
    
    use_few_shot = st.sidebar.checkbox(
        "Use Few-Shot Learning", 
        value=st.session_state.get('use_few_shot', False),
        help="Use few-shot learning for complex queries (slower but more accurate)"
    )
    
    auto_error_correction = st.sidebar.checkbox(
        "Auto Error Correction", 
        value=st.session_state.get('auto_error_correction', True),
        help="Automatically attempt to correct failed SQL queries"
    )
    
    # Store settings in session state
    st.session_state.use_enhanced_prompts = use_enhanced_prompts
    st.session_state.use_few_shot = use_few_shot
    st.session_state.auto_error_correction = auto_error_correction
    
    if st.sidebar.button("Load Model"):
        try:
            if not st.session_state.agent:
                st.session_state.agent = NL2SQLAgent()
            
            with st.spinner("Loading model..."):
                success = st.session_state.agent.load_model(model_name)
                if success:
                    st.session_state.model_loaded = True
                    st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load model")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading error: {str(e)}")
            st.sidebar.error("Make sure you have activated the virtual environment: source nl2sql_env/bin/activate")
    
    # Connection status
    st.sidebar.header("📊 Status")
    if st.session_state.connected:
        st.sidebar.success("Database: Connected")
    else:
        st.sidebar.warning("Database: Not Connected")
    
    if st.session_state.model_loaded:
        st.sidebar.success("Model: Loaded")
    else:
        st.sidebar.warning("Model: Not Loaded")


def main_interface():
    """Main application interface"""
    st.markdown('<h1 class="main-header">🧠 Schema-Aware NL2SQL Agent</h1>', unsafe_allow_html=True)
    
    # Check if system is ready
    if not st.session_state.connected or not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Welcome to Schema-Aware NL2SQL Agent!</h3>
            <p>To get started:</p>
            <ol>
                <li>Connect to your database using the sidebar</li>
                <li>Load a pre-trained NL2SQL model</li>
                <li>Start asking questions in natural language!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    # Predefined example queries
    example_queries = [
        "Show all records from the first table",
        "What is the total count of records?",
        "Show the top 5 records",
        "What are the unique values in the first column?",
        "Show records ordered by the first column",
        "What is the average value of numeric columns?",
        "Show records where the first column is not null",
        "What is the maximum value in numeric columns?"
    ]
    
    selected_example = st.selectbox(
        "Choose an example query or write your own:",
        [""] + example_queries
    )
    
    # Query input
    query_input = st.text_area(
        "Natural Language Query:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="e.g., 'Show me all customers who placed orders in the last month'"
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    with col1:
        execute_query = st.checkbox("Auto-Execute Generated SQL", value=True, help="Automatically execute the SQL generated by the model")
    with col2:
        show_sql = st.checkbox("Show Generated SQL", value=True, help="Display the SQL generated by the model")
    with col3:
        validate_only = st.checkbox("Validate Only (No Execution)", value=False, help="Only validate SQL without executing it")
    
    # Query button
    if st.button("🔍 Process Query", type="primary"):
        if query_input.strip():
            with st.spinner("Processing your question..."):
                try:
                    # Get prompt engineering settings from session state
                    use_enhanced_prompts = st.session_state.get('use_enhanced_prompts', True)
                    use_few_shot = st.session_state.get('use_few_shot', False)
                    
                    # Process the query with enhanced prompts if available
                    if use_enhanced_prompts and hasattr(st.session_state.agent, 'prompt_engineer'):
                        result = st.session_state.agent.process_query(query_input, use_few_shot=use_few_shot)
                    else:
                        result = st.session_state.agent.process_query(query_input)
                    
                    # If validate_only is checked, modify the result to not show execution
                    if validate_only and result.get("success"):
                        result["executed"] = False
                        result["execution_message"] = "SQL validation only - not executed"
                    
                    # Store in history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now(),
                        "query": query_input,
                        "result": result
                    })
                    
                    display_query_result(result)
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question!")

    # --- Raw SQL Execution Section ---
    st.header("📝 Run Raw SQL Query ")
    with st.expander("Run SQL directly on your connected database"):
        sql_query = st.text_area("Enter your SQL query here", height=100, key="raw_sql_input")
        if st.button("Execute SQL", key="execute_raw_sql"):
            if not st.session_state.connected:
                st.error("Please connect to a database first!")
            elif not sql_query.strip():
                st.warning("Please enter a SQL query.")
            else:
                try:
                    result = st.session_state.agent.execute_sql(sql_query)
                    if result.get("success"):
                        st.success("✅ SQL executed successfully!")
                        if result.get("results"):
                            st.dataframe(pd.DataFrame(result["results"]))
                        else:
                            st.info("Query executed but returned no results.")
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error executing SQL: {str(e)}")


def display_query_result(result: Dict[str, Any]):
    """Display the result of a query"""
    if result.get("success"):
        st.markdown('<div class="success-box">✅ Query processed successfully!</div>', unsafe_allow_html=True)
        
        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{result.get('confidence_score', 0):.2f}")
        with col2:
            st.metric("Processing Time", f"{result.get('processing_time', 0):.3f}s")
        with col3:
            # Check if results were returned (indicating execution)
            has_results = bool(result.get("results") or result.get("results_df") is not None)
            execution_status = "Executed" if has_results else "Not Executed"
            st.metric("Execution Status", execution_status)
        
        # Show generated SQL
        if "generated_sql" in result:
            st.subheader("🔧 Generated SQL")
            st.code(result["generated_sql"], language="sql")
            
            # Add execute button if SQL was generated but no results returned
            if not result.get("results") and not result.get("results_df") is not None:
                if st.button("🚀 Execute This SQL", key=f"execute_generated_{hash(result['generated_sql'])}"):
                    try:
                        sql_result = st.session_state.agent.execute_sql(result["generated_sql"])
                        if sql_result.get("success"):
                            st.success("✅ SQL executed successfully!")
                            if sql_result.get("results"):
                                st.subheader("📊 Execution Results")
                                st.dataframe(pd.DataFrame(sql_result["results"]), use_container_width=True)
                                st.info(f"📈 Returned {sql_result.get('row_count', 0)} rows")
                            else:
                                st.info("Query executed but returned no results.")
                        else:
                            st.error(f"❌ Execution failed: {sql_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error executing SQL: {str(e)}")
        
        # Show results
        if result.get("results") or result.get("results_df") is not None:
            st.subheader("📊 Query Results")
            
            # Use results_df if available, otherwise convert results to DataFrame
            if result.get("results_df") is not None:
                results_df = result["results_df"]
            else:
                results_df = pd.DataFrame(result["results"])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Show result summary
            row_count = result.get("row_count", len(results_df))
            st.info(f"📈 Returned {row_count} rows")
            
            # Offer data visualization for numeric data
            if len(results_df) > 0 and len(results_df.select_dtypes(include=['number']).columns) > 0:
                if st.button("📊 Create Visualization", key=f"viz_{hash(str(results_df))}"):
                    create_visualization(results_df)
        
        elif not result.get("results") and not result.get("results_df") is not None:
            st.info("SQL generated but not executed. Use the 'Execute This SQL' button above to run it.")
    
    else:
        st.markdown(f'<div class="error-box">❌ {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)


def create_visualization(df: pd.DataFrame):
    """Create simple visualizations for query results"""
    st.subheader("📈 Data Visualization")
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_columns) > 0:
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
        )
        
        if viz_type == "Bar Chart" and categorical_columns:
            x_col = st.selectbox("X-axis (categorical):", categorical_columns)
            y_col = st.selectbox("Y-axis (numeric):", numeric_columns)
            
            if x_col and y_col:
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            if len(numeric_columns) >= 2:
                x_col = st.selectbox("X-axis:", numeric_columns)
                y_col = st.selectbox("Y-axis:", [col for col in numeric_columns if col != x_col])
                
                if x_col and y_col:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            col = st.selectbox("Column for histogram:", numeric_columns)
            if col:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)


def schema_explorer():
    """Database schema exploration interface"""
    if not st.session_state.connected:
        st.warning("Please connect to a database first!")
        return
    
    st.header("🗄️ Database Schema Explorer")
    
    try:
        schema_info = st.session_state.agent.get_schema_info()
        
        if schema_info.get("success"):
            st.subheader(f"Database Type: {schema_info.get('database_type', 'Unknown')}")
            st.info(f"Total Tables: {schema_info.get('total_tables', 0)}")
            
            # Table selector
            tables = schema_info.get("tables", [])
            selected_table = st.selectbox("Select a table to explore:", [""] + tables)
            
            if selected_table:
                # Get table details
                table_info = st.session_state.agent.get_schema_info(selected_table)
                
                if table_info.get("success"):
                    table_schema = table_info["schema"]
                    
                    # Display table schema
                    st.subheader(f"Table: {selected_table}")
                    
                    # Columns information
                    st.write("**Columns:**")
                    columns_data = []
                    for col in table_schema.get("columns", []):
                        columns_data.append({
                            "Column": col["name"],
                            "Type": col["type"],
                            "Nullable": "Yes" if col["nullable"] else "No",
                            "Primary Key": "Yes" if col["primary_key"] else "No"
                        })
                    
                    if columns_data:
                        st.dataframe(pd.DataFrame(columns_data), hide_index=True)
                    
                    # Foreign keys
                    if table_schema.get("foreign_keys"):
                        st.write("**Foreign Keys:**")
                        for fk in table_schema["foreign_keys"]:
                            st.write(f"- {', '.join(fk['columns'])} → {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
                    
                    # Sample data
                    if st.button(f"Show Sample Data from {selected_table}"):
                        sample_result = st.session_state.agent.get_sample_data(selected_table)
                        if sample_result.get("success"):
                            st.subheader("Sample Data")
                            sample_df = pd.DataFrame(sample_result["sample_data"])
                            st.dataframe(sample_df, use_container_width=True)
            
            # Relationships
            relationships = schema_info.get("relationships", [])
            if relationships:
                st.subheader("Table Relationships")
                for rel in relationships:
                    st.write(f"- {rel['from_table']}.{', '.join(rel['from_columns'])} → {rel['to_table']}.{', '.join(rel['to_columns'])}")
        
    except Exception as e:
        st.error(f"Error exploring schema: {str(e)}")


def query_history():
    """Display query history and statistics"""
    st.header("📜 Query History")
    
    if not st.session_state.query_history:
        st.info("No queries executed yet. Start by asking a question!")
        return
    
    # Display recent queries
    st.subheader("Recent Queries")
    for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
        with st.expander(f"Query {len(st.session_state.query_history) - i}: {entry['query'][:50]}..."):
            st.write(f"**Time:** {entry['timestamp']}")
            st.write(f"**Query:** {entry['query']}")
            
            result = entry['result']
            if result.get('success'):
                st.success("✅ Successful")
                if 'generated_sql' in result:
                    st.code(result['generated_sql'], language='sql')
                if result.get('results') or result.get('results_df') is not None:
                    row_count = result.get('row_count', 0)
                    st.write(f"**Results:** {row_count} rows")
            else:
                st.error(f"❌ Failed: {result.get('error')}")
    
    # Statistics from agent
    if st.session_state.agent:
        try:
            stats = st.session_state.agent.get_statistics()
            
            if stats.get("total_queries", 0) > 0:
                st.subheader("📊 Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries", stats.get("total_queries", 0))
                with col2:
                    st.metric("Success Rate", f"{stats.get('success_rate', 0)*100:.1f}%")
                with col3:
                    st.metric("Execution Rate", f"{stats.get('execution_rate', 0)*100:.1f}%")
                with col4:
                    st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.2f}")
        
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")


def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar
    sidebar_database_connection()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query Interface", "🗄️ Schema Explorer", "📜 History"])
    
    with tab1:
        main_interface()
    
    with tab2:
        schema_explorer()
    
    with tab3:
        query_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🧠 Schema-Aware NL2SQL Agent | Built with Streamlit<br>
        Talk to your data as you talk to a human
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 