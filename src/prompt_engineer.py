"""
Enhanced Prompt Engineering for NL2SQL
Provides better prompts with examples and clear instructions
"""

import logging
from typing import Dict, List, Any, Optional
from .schema_retriever import SchemaRetriever

logger = logging.getLogger(__name__)


class PromptEngineer:
    """Enhanced prompt engineering for better SQL generation"""
    
    def __init__(self, schema_retriever: SchemaRetriever):
        self.schema_retriever = schema_retriever
        self.schema = schema_retriever.get_database_schema()
    
    def get_enhanced_schema_prompt(self) -> str:
        """
        Generate an enhanced schema prompt with examples and clear instructions
        """
        schema = self.schema
        table_names = list(schema["tables"].keys())
        
        prompt_parts = [
            "=== DATABASE SCHEMA AND SQL GENERATION INSTRUCTIONS ===",
            "",
            "TASK: Convert natural language questions to SQL queries.",
            "CRITICAL RULES:",
            "1. ALWAYS use EXACT table names as shown below",
            "2. ALWAYS use EXACT column names as shown below", 
            "3. ALWAYS include a table name after FROM",
            "4. ALWAYS include column names after SELECT (use * for all columns)",
            "5. For aggregations, use proper SQL functions (COUNT, SUM, AVG, etc.)",
            "6. For joins, use proper JOIN syntax",
            "7. For filtering, use WHERE clauses",
            "8. For grouping, use GROUP BY",
            "9. For ordering, use ORDER BY",
            "10. Always end queries with semicolon (;)",
            "11. NEVER use 'SELECT' as a table name",
            "12. NEVER leave FROM clause empty",
            "",
            f"AVAILABLE TABLES: {', '.join(table_names)}",
            ""
        ]
        
        # Add detailed table information
        for table_name, table_info in schema["tables"].items():
            prompt_parts.append(f"TABLE: {table_name}")
            
            # Add column information
            columns = []
            for col in table_info["columns"]:
                col_info = f"{col['name']} ({col['type']})"
                if col['primary_key']:
                    col_info += " [PRIMARY KEY]"
                if not col['nullable']:
                    col_info += " [NOT NULL]"
                columns.append(col_info)
            
            prompt_parts.append("COLUMNS:")
            for col in columns:
                prompt_parts.append(f"  {col}")
            
            # Add foreign keys
            if table_info["foreign_keys"]:
                prompt_parts.append("FOREIGN KEYS:")
                for fk in table_info["foreign_keys"]:
                    fk_desc = f"  {', '.join(fk['columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                    prompt_parts.append(fk_desc)
            
            prompt_parts.append("")
        
        # Add relationships
        if schema["relationships"]:
            prompt_parts.append("TABLE RELATIONSHIPS:")
            for rel in schema["relationships"]:
                rel_desc = f"  {rel['from_table']}.{', '.join(rel['from_columns'])} -> {rel['to_table']}.{', '.join(rel['to_columns'])}"
                prompt_parts.append(rel_desc)
            prompt_parts.append("")
        
        # Add examples
        prompt_parts.extend(self._get_example_queries())
        
        # Add final instructions
        prompt_parts.extend([
            "=== GENERATION INSTRUCTIONS ===",
            "1. Analyze the question carefully",
            "2. Identify which tables and columns are needed",
            "3. Determine if joins are required",
            "4. Apply appropriate filters and aggregations",
            "5. Generate clean, valid SQL",
            "",
            "IMPORTANT: Your response must be a valid SQL query only.",
            "Do not include any explanations or additional text.",
            "Start with SELECT and end with semicolon.",
            "",
            "Generate SQL for the following question:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_example_queries(self) -> List[str]:
        """Get example queries based on the actual schema"""
        examples = [
            "=== EXAMPLE QUERIES ===",
            ""
        ]
        
        # Get table names for examples
        table_names = list(self.schema["tables"].keys())
        
        if "Customer" in table_names:
            examples.extend([
                "Q: Show all customers from Germany",
                "SQL: SELECT * FROM Customer WHERE Country = 'Germany';",
                "",
                "Q: Count the number of customers",
                "SQL: SELECT COUNT(*) FROM Customer;",
                "",
                "Q: List customers with their phone numbers",
                "SQL: SELECT FirstName, LastName, Phone FROM Customer;",
                "",
                "Q: Show all customers",
                "SQL: SELECT * FROM Customer;",
                ""
            ])
        
        if "Product" in table_names:
            examples.extend([
                "Q: List all products with their prices",
                "SQL: SELECT ProductName, UnitPrice FROM Product;",
                "",
                "Q: Show products that are not discontinued",
                "SQL: SELECT ProductName, UnitPrice FROM Product WHERE IsDiscontinued = 0;",
                "",
                "Q: Count the number of products",
                "SQL: SELECT COUNT(*) FROM Product;",
                ""
            ])
        
        if "Order" in table_names and "Customer" in table_names:
            examples.extend([
                "Q: Show orders with customer names",
                "SQL: SELECT o.Id, o.OrderDate, c.FirstName, c.LastName FROM `Order` o JOIN Customer c ON o.CustomerId = c.Id;",
                "",
                "Q: Count orders per customer",
                "SQL: SELECT c.FirstName, c.LastName, COUNT(o.Id) as OrderCount FROM Customer c LEFT JOIN `Order` o ON c.Id = o.CustomerId GROUP BY c.Id, c.FirstName, c.LastName;",
                ""
            ])
        
        if "OrderItem" in table_names and "Product" in table_names:
            examples.extend([
                "Q: Show order items with product names",
                "SQL: SELECT oi.Id, p.ProductName, oi.Quantity, oi.UnitPrice FROM OrderItem oi JOIN Product p ON oi.ProductId = p.Id;",
                "",
                "Q: Calculate total sales per product",
                "SQL: SELECT p.ProductName, SUM(oi.Quantity * oi.UnitPrice) as TotalSales FROM OrderItem oi JOIN Product p ON oi.ProductId = p.Id GROUP BY p.Id, p.ProductName;",
                ""
            ])
        
        if "Supplier" in table_names:
            examples.extend([
                "Q: List suppliers from the UK",
                "SQL: SELECT CompanyName, ContactName, Phone FROM Supplier WHERE Country = 'UK';",
                "",
                "Q: Count suppliers by country",
                "SQL: SELECT Country, COUNT(*) as SupplierCount FROM Supplier GROUP BY Country;",
                "",
                "Q: List suppliers from Germany",
                "SQL: SELECT * FROM Supplier WHERE Country = 'Germany';",
                ""
            ])
        
        examples.append("")
        return examples
    
    def get_contextual_prompt(self, question: str) -> str:
        """
        Generate a contextual prompt based on the specific question
        """
        # Analyze question to determine what tables/operations are needed
        question_lower = question.lower()
        
        # Determine relevant tables
        relevant_tables = []
        if any(word in question_lower for word in ['customer', 'customers']):
            relevant_tables.append('Customer')
        if any(word in question_lower for word in ['product', 'products']):
            relevant_tables.append('Product')
        if any(word in question_lower for word in ['order', 'orders']):
            relevant_tables.append('Order')
        if any(word in question_lower for word in ['supplier', 'suppliers']):
            relevant_tables.append('Supplier')
        if any(word in question_lower for word in ['item', 'items']):
            relevant_tables.append('OrderItem')
        
        # Determine operation type
        operation_hints = []
        if any(word in question_lower for word in ['count', 'how many', 'number of']):
            operation_hints.append("Use COUNT() for counting")
        if any(word in question_lower for word in ['total', 'sum', 'amount']):
            operation_hints.append("Use SUM() for totals")
        if any(word in question_lower for word in ['average', 'avg']):
            operation_hints.append("Use AVG() for averages")
        if any(word in question_lower for word in ['group', 'per', 'by']):
            operation_hints.append("Use GROUP BY for grouping")
        if len(relevant_tables) > 1:
            operation_hints.append("Use JOIN to combine multiple tables")
        
        # Build contextual prompt
        prompt_parts = [
            "=== CONTEXTUAL SQL GENERATION ===",
            "",
            f"QUESTION: {question}",
            "",
            "ANALYSIS:",
            f"Relevant tables: {', '.join(relevant_tables) if relevant_tables else 'All tables'}"
        ]
        
        if operation_hints:
            prompt_parts.append("Required operations:")
            for hint in operation_hints:
                prompt_parts.append(f"  - {hint}")
        
        prompt_parts.extend([
            "",
            "SCHEMA:",
            self._get_relevant_schema_info(relevant_tables),
            "",
            "IMPORTANT: Generate ONLY a valid SQL query.",
            "Start with SELECT and end with semicolon.",
            "Use exact table and column names from the schema above.",
            "",
            "Generate SQL for this specific question:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_relevant_schema_info(self, relevant_tables: List[str]) -> str:
        """Get schema information for relevant tables only"""
        if not relevant_tables:
            return self.get_enhanced_schema_prompt()
        
        schema_parts = []
        for table_name in relevant_tables:
            if table_name in self.schema["tables"]:
                table_info = self.schema["tables"][table_name]
                schema_parts.append(f"TABLE: {table_name}")
                
                columns = []
                for col in table_info["columns"]:
                    col_info = f"{col['name']} ({col['type']})"
                    if col['primary_key']:
                        col_info += " [PK]"
                    columns.append(col_info)
                
                schema_parts.append("COLUMNS:")
                for col in columns:
                    schema_parts.append(f"  {col}")
                
                # Add foreign keys
                if table_info["foreign_keys"]:
                    schema_parts.append("FOREIGN KEYS:")
                    for fk in table_info["foreign_keys"]:
                        fk_desc = f"  {', '.join(fk['columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                        schema_parts.append(fk_desc)
                
                schema_parts.append("")
        
        return "\n".join(schema_parts)
    
    def get_few_shot_prompt(self, question: str, examples: List[Dict[str, str]] = None) -> str:
        """
        Generate a few-shot learning prompt with examples
        """
        if examples is None:
            examples = [
                {
                    "question": "Show all customers from Germany",
                    "sql": "SELECT * FROM Customer WHERE Country = 'Germany';"
                },
                {
                    "question": "Count the number of products",
                    "sql": "SELECT COUNT(*) FROM Product;"
                },
                {
                    "question": "List products with their prices",
                    "sql": "SELECT ProductName, UnitPrice FROM Product;"
                }
            ]
        
        prompt_parts = [
            "Convert natural language to SQL. Use exact table and column names.",
            ""
        ]
        
        # Add examples
        for example in examples:
            prompt_parts.extend([
                f"Q: {example['question']}",
                f"SQL: {example['sql']}",
                ""
            ])
        
        # Add schema information
        prompt_parts.extend([
            "SCHEMA:",
            self._get_minimal_schema_info(),
            "",
            f"Q: {question}",
            "SQL:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_minimal_schema_info(self) -> str:
        """Get minimal schema information for few-shot prompts"""
        schema_parts = []
        for table_name, table_info in self.schema["tables"].items():
            columns = [col['name'] for col in table_info["columns"]]
            schema_parts.append(f"{table_name}({', '.join(columns)})")
        
        return " | ".join(schema_parts)
    
    def get_error_correction_prompt(self, question: str, failed_sql: str, error_message: str) -> str:
        """
        Generate a prompt to correct a failed SQL query
        """
        return f"""
=== SQL ERROR CORRECTION ===

Original Question: {question}
Failed SQL: {failed_sql}
Error: {error_message}

Please correct the SQL query. Common issues to check:
1. Use exact table names: {', '.join(self.schema['tables'].keys())}
2. Use exact column names from the schema
3. Check SQL syntax (proper JOIN, WHERE, GROUP BY syntax)
4. Ensure all referenced tables and columns exist

Corrected SQL:
"""
    
    def get_structured_prompt(self, question: str) -> str:
        """
        Generate a highly structured prompt for better SQL generation
        """
        schema = self.schema
        table_names = list(schema["tables"].keys())
        
        # Analyze question to determine relevant tables
        question_lower = question.lower()
        relevant_tables = []
        
        if any(word in question_lower for word in ['customer', 'customers']):
            relevant_tables.append('Customer')
        if any(word in question_lower for word in ['product', 'products']):
            relevant_tables.append('Product')
        if any(word in question_lower for word in ['order', 'orders']):
            relevant_tables.append('Order')
        if any(word in question_lower for word in ['supplier', 'suppliers']):
            relevant_tables.append('Supplier')
        if any(word in question_lower for word in ['item', 'items']):
            relevant_tables.append('OrderItem')
        
        # If no specific tables identified, use all tables
        if not relevant_tables:
            relevant_tables = table_names
        
        prompt = f"""SQL Generation Task:
Question: {question}

Available Tables: {', '.join(table_names)}
Relevant Tables: {', '.join(relevant_tables)}

Schema Information:
"""
        
        # Add schema for relevant tables
        for table_name in relevant_tables:
            if table_name in schema["tables"]:
                table_info = schema["tables"][table_name]
                columns = [col['name'] for col in table_info["columns"]]
                prompt += f"{table_name}: {', '.join(columns)}\n"
        
        prompt += f"""
Instructions:
1. Generate ONLY a valid SQL query
2. Use exact table names: {', '.join(table_names)}
3. Use exact column names from the schema above
4. Start with SELECT and end with semicolon
5. Include table name after FROM
6. Use * for all columns, or specify column names

SQL Query:"""
        
        return prompt 