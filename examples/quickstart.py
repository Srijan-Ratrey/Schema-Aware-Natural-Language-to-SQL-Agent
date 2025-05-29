#!/usr/bin/env python3
"""
Quick Start Script for Schema-Aware NL2SQL Agent
Demonstrates basic usage with minimal setup
"""

import os
import sys
from src.nl2sql_agent import NL2SQLAgent
import sqlite3


def create_sample_database():
    """Create a simple sample database"""
    db_path = "quickstart_sample.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a simple books table
    cursor.execute("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            genre TEXT,
            publication_year INTEGER,
            price DECIMAL(10, 2),
            rating DECIMAL(3, 1)
        )
    """)
    
    # Insert sample data
    books_data = [
        (1, "The Great Gatsby", "F. Scott Fitzgerald", "Fiction", 1925, 12.99, 4.2),
        (2, "To Kill a Mockingbird", "Harper Lee", "Fiction", 1960, 14.99, 4.5),
        (3, "1984", "George Orwell", "Dystopian", 1949, 13.99, 4.4),
        (4, "Pride and Prejudice", "Jane Austen", "Romance", 1813, 11.99, 4.3),
        (5, "The Catcher in the Rye", "J.D. Salinger", "Fiction", 1951, 13.50, 3.8),
        (6, "Lord of the Flies", "William Golding", "Fiction", 1954, 12.50, 4.0),
        (7, "Harry Potter and the Sorcerer's Stone", "J.K. Rowling", "Fantasy", 1997, 15.99, 4.7),
        (8, "The Hobbit", "J.R.R. Tolkien", "Fantasy", 1937, 14.50, 4.6),
        (9, "Fahrenheit 451", "Ray Bradbury", "Science Fiction", 1953, 13.99, 4.1),
        (10, "Jane Eyre", "Charlotte Brontë", "Romance", 1847, 12.99, 4.2)
    ]
    
    cursor.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?)", books_data)
    conn.commit()
    conn.close()
    
    return db_path


def main():
    """Main quickstart function"""
    print("🧠 Schema-Aware NL2SQL Agent - Quick Start")
    print("=" * 50)
    
    try:
        # Step 1: Create sample database
        print("1. Creating sample database...")
        db_path = create_sample_database()
        print(f"   ✅ Sample database created: {db_path}")
        
        # Step 2: Initialize agent
        print("\n2. Initializing NL2SQL Agent...")
        agent = NL2SQLAgent()
        print("   ✅ Agent initialized")
        
        # Step 3: Connect to database
        print("\n3. Connecting to database...")
        success = agent.connect_database("sqlite", db_path=db_path)
        if not success:
            print("   ❌ Failed to connect to database")
            return
        print("   ✅ Connected to database")
        
        # Step 4: Load model
        print("\n4. Loading NL2SQL model (this may take a moment)...")
        success = agent.load_model()  # Uses default Spider-trained model
        if not success:
            print("   ❌ Failed to load model")
            return
        print("   ✅ Model loaded successfully")
        
        # Step 5: Interactive query loop
        print("\n5. Ready for queries! Try asking questions about the books database.")
        print("   Sample questions:")
        print("   - 'Show all books by J.K. Rowling'")
        print("   - 'What are the top 3 highest rated books?'")
        print("   - 'How many books were published after 1950?'")
        print("   - 'List all fantasy books'")
        print("\n   Type 'quit' to exit, 'help' for more examples")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_question = input("\n🤔 Ask a question about the books: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_question.lower() in ['help', 'h']:
                    print("\n📚 Example questions you can ask:")
                    print("   • Show all books")
                    print("   • What is the average rating of books?")
                    print("   • List books published before 1950")
                    print("   • Show the most expensive book")
                    print("   • How many different genres are there?")
                    print("   • Find books with rating above 4.5")
                    continue
                
                if not user_question:
                    print("   Please enter a question!")
                    continue
                
                # Process the question
                print(f"\n🔍 Processing: {user_question}")
                
                result = agent.process_query(user_question)
                
                if result.get("success"):
                    print(f"✅ Generated SQL: {result.get('generated_sql')}")
                    print(f"   Confidence: {result.get('confidence_score', 0):.2f}")
                    
                    if result.get('results'):
                        print(f"\n📊 Results ({result.get('row_count', 0)} rows):")
                        
                        # Display results in a simple table format
                        results = result['results']
                        if results:
                            # Print column headers
                            columns = list(results[0].keys()) if results else []
                            if columns:
                                header = " | ".join(f"{col:15}" for col in columns)
                                print(f"   {header}")
                                print(f"   {'-' * len(header)}")
                                
                                # Print rows
                                for row in results[:10]:  # Limit to 10 rows
                                    row_str = " | ".join(f"{str(row.get(col, ''))[:15]:15}" for col in columns)
                                    print(f"   {row_str}")
                                
                                if len(results) > 10:
                                    print(f"   ... and {len(results) - 10} more rows")
                    else:
                        print("ℹ️  Query executed but returned no results")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
        
        # Cleanup
        print("\n6. Cleaning up...")
        agent.close_connections()
        print("   ✅ Connections closed")
        
        print("\n" + "=" * 50)
        print("🎉 Quickstart completed successfully!")
        print("\nNext steps:")
        print("• Try the full web interface: `streamlit run app.py`")
        print("• Run the comprehensive demo: `python demo.py`")
        print("• Connect to your own database using the Python API")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Quickstart failed: {str(e)}")
        print("\nTroubleshooting:")
        print("• Make sure all dependencies are installed: `pip install -r requirements.txt`")
        print("• Check that you have sufficient memory (4GB+ recommended)")
        print("• Try using a smaller model if you encounter memory issues")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 