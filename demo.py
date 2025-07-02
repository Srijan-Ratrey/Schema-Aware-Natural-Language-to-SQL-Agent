"""
Demo Script for Schema-Aware NL2SQL Agent
Demonstrates the capabilities of the system with a sample database
"""

import os
from src.nl2sql_agent import NL2SQLAgent
from src.schema_retriever import create_schema_retriever
import sqlite3
import pandas as pd


def create_demo_database():
    """Create a sample database for demonstration"""
    db_path = "demo_company.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT,
            position TEXT,
            salary DECIMAL(10, 2),
            hire_date DATE,
            manager_id INTEGER,
            FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
        )
    """)
    
    # Create departments table
    cursor.execute("""
        CREATE TABLE departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL,
            location TEXT,
            budget DECIMAL(12, 2)
        )
    """)
    
    # Create projects table
    cursor.execute("""
        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            department_id INTEGER,
            start_date DATE,
            end_date DATE,
            budget DECIMAL(12, 2),
            status TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(department_id)
        )
    """)
    
    # Insert sample data
    employees_data = [
        (1, 'John', 'Smith', 'john.smith@company.com', 'Engineering', 'Software Engineer', 75000, '2022-01-15', 5),
        (2, 'Sarah', 'Johnson', 'sarah.johnson@company.com', 'Marketing', 'Marketing Specialist', 55000, '2022-02-20', 6),
        (3, 'Mike', 'Brown', 'mike.brown@company.com', 'Engineering', 'DevOps Engineer', 80000, '2021-11-10', 5),
        (4, 'Lisa', 'Davis', 'lisa.davis@company.com', 'HR', 'HR Specialist', 60000, '2022-03-05', 7),
        (5, 'David', 'Wilson', 'david.wilson@company.com', 'Engineering', 'Engineering Manager', 95000, '2020-05-15', None),
        (6, 'Emma', 'Miller', 'emma.miller@company.com', 'Marketing', 'Marketing Manager', 85000, '2021-08-20', None),
        (7, 'Robert', 'Taylor', 'robert.taylor@company.com', 'HR', 'HR Manager', 90000, '2020-03-10', None),
        (8, 'Jennifer', 'Anderson', 'jennifer.anderson@company.com', 'Engineering', 'Senior Software Engineer', 85000, '2021-06-01', 5),
        (9, 'James', 'Thomas', 'james.thomas@company.com', 'Marketing', 'Content Creator', 50000, '2022-04-12', 6),
        (10, 'Michelle', 'Garcia', 'michelle.garcia@company.com', 'HR', 'Recruiter', 55000, '2022-01-30', 7)
    ]
    
    departments_data = [
        (1, 'Engineering', 'Building A', 500000),
        (2, 'Marketing', 'Building B', 200000),
        (3, 'HR', 'Building C', 150000),
        (4, 'Finance', 'Building A', 300000)
    ]
    
    projects_data = [
        (1, 'Website Redesign', 1, '2023-01-01', '2023-06-30', 150000, 'In Progress'),
        (2, 'Mobile App Development', 1, '2023-03-01', '2023-12-31', 300000, 'In Progress'),
        (3, 'Brand Campaign', 2, '2023-02-01', '2023-08-31', 75000, 'Completed'),
        (4, 'HR System Upgrade', 3, '2023-01-15', '2023-05-15', 50000, 'Completed'),
        (5, 'Data Analytics Platform', 1, '2023-04-01', '2024-03-31', 400000, 'Planning')
    ]
    
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", employees_data)
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments_data)
    cursor.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)", projects_data)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Demo database created: {db_path}")
    return db_path


def demo_schema_retrieval(db_path):
    """Demonstrate schema retrieval capabilities"""
    print("\n" + "="*50)
    print("🗄️ SCHEMA RETRIEVAL DEMO")
    print("="*50)
    
    # Create schema retriever
    retriever = create_schema_retriever("sqlite", db_path=db_path)
    
    # Get database schema
    schema = retriever.get_database_schema()
    
    print(f"Database Type: {schema['database_type']}")
    print(f"Number of Tables: {len(schema['tables'])}")
    print(f"Tables: {list(schema['tables'].keys())}")
    
    # Show schema prompt
    print("\nSchema Prompt for Model:")
    print("-" * 30)
    print(retriever.get_schema_prompt())
    
    # Show sample data
    print("\nSample Data from employees table:")
    sample_df = retriever.get_sample_data("employees", 3)
    print(sample_df.to_string(index=False))
    
    retriever.close_connection()


def demo_nl2sql_queries(db_path):
    """Demonstrate NL2SQL query generation and execution"""
    print("\n" + "="*50)
    print("🧠 NL2SQL GENERATION DEMO")
    print("="*50)
    
    # Initialize agent
    agent = NL2SQLAgent()
    
    # Connect to database
    print("Connecting to database...")
    success = agent.connect_database("sqlite", db_path=db_path)
    if not success:
        print("❌ Failed to connect to database")
        return
    
    # Load model (this will take some time for first run)
    print("Loading NL2SQL model...")
    success = agent.load_model()
    if not success:
        print("❌ Failed to load model")
        return
    
    # Demo queries
    demo_queries = [
        "Show all employees in the Engineering department",
        "What is the average salary by department?",
        "List all projects that are currently in progress",
        "Who are the managers in the company?",
        "What is the total budget for all departments?",
        "Show employees hired in 2022",
        "Which department has the highest budget?"
    ]
    
    print(f"\n🔍 Testing {len(demo_queries)} natural language queries:")
    print("-" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            result = agent.process_query(query)
            
            if result.get("success"):
                print(f"✅ Success (Confidence: {result.get('confidence_score', 0):.2f})")
                print(f"Generated SQL: {result.get('generated_sql')}")
                
                if result.get('results'):
                    df = pd.DataFrame(result['results'])
                    print(f"Results ({len(df)} rows):")
                    print(df.to_string(index=False))
                else:
                    print("No results returned")
            else:
                print(f"❌ Failed: {result.get('error')}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
    
    # Show statistics
    stats = agent.get_statistics()
    print("\n📊 Session Statistics:")
    print(f"Total Queries: {stats.get('total_queries', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"Execution Rate: {stats.get('execution_rate', 0)*100:.1f}%")
    print(f"Average Confidence: {stats.get('average_confidence', 0):.2f}")
    
    agent.close_connections()


def demo_advanced_features(db_path):
    """Demonstrate advanced features"""
    print("\n" + "="*50)
    print("🚀 ADVANCED FEATURES DEMO")
    print("="*50)
    
    agent = NL2SQLAgent()
    agent.connect_database("sqlite", db_path=db_path)
    agent.load_model()
    
    # Schema exploration
    print("1. Schema Information:")
    schema_info = agent.get_schema_info()
    if schema_info.get("success"):
        print(f"   Database Type: {schema_info['database_type']}")
        print(f"   Tables: {schema_info['tables']}")
        print(f"   Relationships: {len(schema_info['relationships'])} found")
    
    # Sample data
    print("\n2. Sample Data Retrieval:")
    sample_result = agent.get_sample_data("employees", 2)
    if sample_result.get("success"):
        print(f"   Retrieved {sample_result['sample_size']} sample rows")
        print(f"   Columns: {sample_result['columns']}")
    
    # Batch processing
    print("\n3. Batch Query Processing:")
    batch_queries = [
        "How many employees are there?",
        "What is the total budget across all departments?"
    ]
    
    batch_results = agent.batch_query(batch_queries)
    print(f"   Processed {len(batch_results)} queries in batch")
    for i, result in enumerate(batch_results):
        status = "✅ Success" if result.get("success") else "❌ Failed"
        print(f"   Query {i+1}: {status}")
    
    # SQL explanation
    print("\n4. SQL Explanation:")
    sql_query = "SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department"
    explanation = agent.explain_sql(sql_query)
    if explanation.get("success"):
        print(f"   Query Type: {explanation['query_type']}")
        print(f"   Complexity: {explanation['estimated_complexity']}")
        print(f"   Tables: {explanation['tables_involved']}")
    
    agent.close_connections()


def main():
    """Main demo function"""
    print("🧠 Schema-Aware NL2SQL Agent Demo")
    print("=" * 50)
    
    try:
        # Create demo database
        db_path = create_demo_database()
        
        # Run demo components
        demo_schema_retrieval(db_path)
        demo_nl2sql_queries(db_path)
        demo_advanced_features(db_path)
        
        print("\n" + "="*50)
        print("✅ Demo completed successfully!")
        print(f"Demo database saved as: {db_path}")
        print("\nTo run the web interface: streamlit run app.py")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 