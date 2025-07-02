#!/usr/bin/env python3
"""
Database Setup Script for NL2SQL Project
This script creates a SQLite database with sample data for testing the NL2SQL functionality.
"""

import sqlite3
import os
import sys
from pathlib import Path

def setup_database():
    """Set up the SQLite database with schema and sample data."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    db_path = data_dir / "sample_database.db"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Remove existing database if it exists
    if db_path.exists():
        print(f"Removing existing database: {db_path}")
        db_path.unlink()
    
    # Connect to SQLite database (this will create it)
    print(f"Creating new database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Read and execute schema
        schema_file = data_dir / "schema.sql"
        if schema_file.exists():
            print("Creating database schema...")
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
                cursor.executescript(schema_sql)
            print("Schema created successfully!")
        else:
            print("Warning: schema.sql not found in data directory")
            return False
        
        # Read and execute sample data (SQLite version)
        data_file = data_dir / "sample-data-sqlite.sql"
        if data_file.exists():
            print("Loading sample data...")
            with open(data_file, 'r') as f:
                data_sql = f.read()
                cursor.executescript(data_sql)
            print("Sample data loaded successfully!")
        else:
            print("Warning: sample-data-sqlite.sql not found in data directory")
            return False
        
        # Commit changes
        conn.commit()
        
        # Verify the setup
        print("\nVerifying database setup...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Created tables: {[table[0] for table in tables]}")
        
        # Show record counts
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count} records")
        
        print(f"\nDatabase setup completed successfully!")
        print(f"Database location: {db_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()

def main():
    """Main function to run the database setup."""
    print("NL2SQL Database Setup")
    print("=" * 50)
    
    success = setup_database()
    
    if success:
        print("\n✅ Database setup completed successfully!")
        print("You can now use this database with your NL2SQL project.")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 