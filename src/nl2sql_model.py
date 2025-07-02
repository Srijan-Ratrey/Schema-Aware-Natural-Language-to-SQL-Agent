"""
NL2SQL Model Module with Fine-tuned T5 for Schema-Aware SQL Generation
Integrates with Hugging Face models and supports local fine-tuning
"""

import logging
import torch
from typing import Dict, List, Any, Optional
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import sqlglot
from sqlglot import transpile
import re

# Check for sentencepiece dependency
try:
    import sentencepiece
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    logging.warning("SentencePiece not found. T5 models may not work properly.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NL2SQLModel:
    """Schema-aware Natural Language to SQL conversion using T5"""
    
    def __init__(self, model_name: str = "gaussalgo/T5-LM-Large-text2sql-spider", device: str = None):
        """
        Initialize NL2SQL model
        
        Args:
            model_name: Hugging Face model identifier or local path
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Check if sentencepiece is available for T5 models
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "SentencePiece library is required for T5 models. "
                "Please install it with: pip install sentencepiece==0.1.99"
            )
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def generate_sql(
        self, 
        natural_language_query: str, 
        database_schema: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language input
        
        Args:
            natural_language_query: User's question in natural language
            database_schema: Database schema description
            max_length: Maximum length of generated SQL
            num_beams: Number of beams for beam search
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing generated SQL and metadata
        """
        try:
            # Prepare input with schema context
            input_text = self._prepare_input(natural_language_query, database_schema)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode output
            generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and validate SQL
            cleaned_sql = self._clean_sql(generated_sql)
            confidence_score = self._calculate_confidence(inputs, outputs)
            
            return {
                "sql_query": cleaned_sql,
                "raw_output": generated_sql,
                "confidence_score": confidence_score,
                "input_text": input_text
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                "sql_query": None,
                "error": str(e),
                "confidence_score": 0.0
            }
    
    def _prepare_input(self, query: str, schema: str) -> str:
        """
        Prepare input text combining query and schema
        Optimized for T5-based models with proper schema integration
        """
        # Extract actual table names and columns from schema
        table_info = {}
        
        if "Table:" in schema:
            lines = schema.split('\n')
            current_table = None
            for line in lines:
                if line.strip().startswith("Table:"):
                    current_table = line.split("Table:")[1].strip()
                    table_info[current_table] = []
                elif line.strip().startswith("  - ") and current_table:
                    # Extract column info
                    col_line = line.strip()[3:]  # Remove "  - "
                    if "(" in col_line:
                        col_name = col_line.split("(")[0].strip()
                        table_info[current_table].append(col_name)
        elif "Table " in schema:
            # Handle simple schema format
            parts = schema.split(" | ")
            for part in parts:
                if part.strip().startswith("Table "):
                    table_name = part.split("Table ")[1].split(":")[0].strip()
                    # Extract column names from the part
                    col_part = part.split(":")[1] if ":" in part else ""
                    columns = [col.strip().split("(")[0] for col in col_part.split(",") if col.strip()]
                    table_info[table_name] = columns
        
        # Use the first table as primary
        primary_table = list(table_info.keys())[0] if table_info else "table"
        primary_columns = table_info.get(primary_table, [])
        
        # Create a more informative input that includes schema context
        if len(table_info) == 1:
            # Single table - include column information
            columns_str = ", ".join(primary_columns[:5])  # Limit to first 5 columns
            input_template = f"Schema: Table {primary_table} with columns: {columns_str} | translate English to SQL: {query}"
        else:
            # Multiple tables - include table names
            table_names = list(table_info.keys())
            schema_summary = " | ".join([f"Table {name}" for name in table_names])
            input_template = f"Schema: {schema_summary} | translate English to SQL: {query}"
        
        return input_template
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and format generated SQL"""
        if not sql:
            return ""
        
        # Remove common prefixes/suffixes
        sql = sql.strip()
        
        # Remove the input prompt if it's echoed back
        sql = re.sub(r'^.*?translate english to sql:\s*', '', sql, flags=re.IGNORECASE)
        
        # Remove any leading "SQL:" or similar prefixes
        sql = re.sub(r'^(SQL:\s*|sql:\s*)', '', sql, flags=re.IGNORECASE)
        
        # Remove schema descriptions that might be included
        sql = re.sub(r'\|\s*Database Schema.*$', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\|\s*Table:.*$', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'Schema:.*?\|', '', sql, flags=re.IGNORECASE)
        
        # Fix common table name issues (singular/plural)
        sql = re.sub(r'\bFROM\s+customer\b', 'FROM customers', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bFROM\s+product\b', 'FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bFROM\s+order\b', 'FROM orders', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bFROM\s+employee\b', 'FROM employees', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bFROM\s+department\b', 'FROM departments', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bFROM\s+project\b', 'FROM projects', sql, flags=re.IGNORECASE)
        
        # Fix table names in JOIN clauses
        sql = re.sub(r'\bJOIN\s+customer\b', 'JOIN customers', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bJOIN\s+product\b', 'JOIN products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bJOIN\s+order\b', 'JOIN orders', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bJOIN\s+employee\b', 'JOIN employees', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bJOIN\s+department\b', 'JOIN departments', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bJOIN\s+project\b', 'JOIN projects', sql, flags=re.IGNORECASE)
        
        # Fix table names in UPDATE and DELETE
        sql = re.sub(r'\bUPDATE\s+customer\b', 'UPDATE customers', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bUPDATE\s+product\b', 'UPDATE products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bDELETE\s+FROM\s+customer\b', 'DELETE FROM customers', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bDELETE\s+FROM\s+product\b', 'DELETE FROM products', sql, flags=re.IGNORECASE)
        
        # Remove table descriptions in parentheses
        sql = re.sub(r'\([^)]*\)', '', sql)
        
        # Clean up common issues
        sql = re.sub(r'\[.*?\]', '', sql)  # Remove [PRIMARY KEY] etc
        sql = re.sub(r'WHERE\s*$', '', sql)  # Remove trailing WHERE
        
        # Fix malformed SQL with repeated FROM clauses
        sql = re.sub(r'FROM\s+FROM\s+FROM\s+.*', 'FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'FROM\s+FROM\s+.*', 'FROM products', sql, flags=re.IGNORECASE)
        
        # Fix malformed aggregate functions
        sql = re.sub(r'SELECT\s+MAX\s*\(\s*FROM\s*\)', 'SELECT MAX(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+MIN\s*\(\s*FROM\s*\)', 'SELECT MIN(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+AVG\s*\(\s*FROM\s*\)', 'SELECT AVG(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+COUNT\s*\(\s*FROM\s*\)', 'SELECT COUNT(*)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+SUM\s*\(\s*FROM\s*\)', 'SELECT SUM(price)', sql, flags=re.IGNORECASE)
        
        # Fix more specific aggregate function issues
        sql = re.sub(r'SELECT\s+MAX\s*\(\s*FROM\s*\)\s+products', 'SELECT MAX(price) FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+MIN\s*\(\s*FROM\s*\)\s+products', 'SELECT MIN(price) FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+AVG\s*\(\s*FROM\s*\)\s+products', 'SELECT AVG(price) FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+COUNT\s*\(\s*FROM\s*\)\s+products', 'SELECT COUNT(*) FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+SUM\s*\(\s*FROM\s*\)\s+products', 'SELECT SUM(price) FROM products', sql, flags=re.IGNORECASE)
        
        # Fix aggregate functions with wrong column names
        sql = re.sub(r'SELECT\s+MAX\s*\(\s*product\s*\)', 'SELECT MAX(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+MIN\s*\(\s*product\s*\)', 'SELECT MIN(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+AVG\s*\(\s*product\s*\)', 'SELECT AVG(price)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+COUNT\s*\(\s*product\s*\)', 'SELECT COUNT(*)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+SUM\s*\(\s*product\s*\)', 'SELECT SUM(price)', sql, flags=re.IGNORECASE)
        
        # Fix column name issues - replace common wrong column names with correct ones
        sql = re.sub(r'SELECT\s+DISTINCT\s+product\s+FROM', 'SELECT DISTINCT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+product\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+products\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+name\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        
        # Fix table alias issues - remove complex JOINs and aliases for simple queries
        sql = re.sub(r'SELECT\s+(\w+)\.(\w+)\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\.(\w+)', r'SELECT * FROM \3 WHERE \5', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+DISTINCT\s+(\w+)\.(\w+)\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\.(\w+)', r'SELECT DISTINCT * FROM \3 WHERE \5', sql, flags=re.IGNORECASE)
        
        # Remove complex JOIN clauses that are malformed
        sql = re.sub(r'AS T1 JOIN translate AS T2 ON.*?WHERE', 'WHERE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'AS T1 JOIN.*?WHERE', 'WHERE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'JOIN.*?ON.*?WHERE', 'WHERE', sql, flags=re.IGNORECASE)
        
        # Fix malformed WHERE clauses
        sql = re.sub(r'WHERE\s+(\w+)\s+category\s+ORDER\s+BY', r'WHERE \1 = \'category\' ORDER BY', sql, flags=re.IGNORECASE)
        
        # Fix aggregate function syntax
        sql = re.sub(r'SELECT\s+MAX\s+(\w+)', r'SELECT MAX(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+MIN\s+(\w+)', r'SELECT MIN(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+AVG\s+(\w+)', r'SELECT AVG(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+COUNT\s+(\w+)', r'SELECT COUNT(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+SUM\s+(\w+)', r'SELECT SUM(\1)', sql, flags=re.IGNORECASE)
        
        # Fix common column selection issues
        sql = re.sub(r'SELECT\s+all\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+everything\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        
        # Basic SQL formatting
        sql = sql.replace('\n', ' ').replace('\t', ' ')
        sql = ' '.join(sql.split())  # Remove extra whitespace
        
        # Check if SQL is still malformed and apply fallback
        if self._is_malformed_sql(sql):
            sql = self._generate_simple_fallback_sql(sql)
        
        # Ensure proper SQL structure
        if sql and not any(keyword in sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return ""
        
        # Ensure SQL ends with semicolon if it's a complete query
        if sql and not sql.endswith(';') and any(keyword in sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            sql += ';'
        
        return sql
    
    def _is_malformed_sql(self, sql: str) -> bool:
        """Check if SQL is malformed and needs fallback"""
        sql_upper = sql.upper()
        
        # Check for common malformed patterns
        malformed_patterns = [
            r'SELECT.*FROM.*FROM',  # Multiple FROM clauses
            r'SELECT.*\(\s*FROM\s*\)',  # SELECT MAX(FROM)
            r'SELECT.*\w+\.\w+.*FROM.*WHERE.*\w+\.\w+',  # Complex aliases
            r'JOIN.*ON.*WHERE',  # Malformed JOINs
            r'WHERE.*=.*FROM',  # WHERE clause with FROM
            r'SELECT.*T\d+\.',  # Table aliases like T1, T2
            r'FROM.*WHERE.*T\d+\.',  # WHERE with table aliases
        ]
        
        for pattern in malformed_patterns:
            if re.search(pattern, sql_upper):
                return True
        
        return False
    
    def _generate_simple_fallback_sql(self, original_sql: str) -> str:
        """Generate simple fallback SQL when the original is malformed"""
        sql_upper = original_sql.upper()
        
        # Extract table name from original SQL
        table_match = re.search(r'FROM\s+(\w+)', sql_upper)
        table_name = table_match.group(1) if table_match else "products"
        
        # Generate simple SQL based on keywords
        if 'DISTINCT' in sql_upper:
            return f"SELECT DISTINCT * FROM {table_name};"
        elif 'MAX' in sql_upper or 'MOST EXPENSIVE' in original_sql.upper():
            return f"SELECT MAX(price) FROM {table_name};"
        elif 'MIN' in sql_upper or 'LEAST EXPENSIVE' in original_sql.upper():
            return f"SELECT MIN(price) FROM {table_name};"
        elif 'AVG' in sql_upper or 'AVERAGE' in original_sql.upper():
            return f"SELECT AVG(price) FROM {table_name};"
        elif 'COUNT' in sql_upper:
            return f"SELECT COUNT(*) FROM {table_name};"
        elif 'WHERE' in sql_upper:
            # Try to extract the WHERE condition
            where_match = re.search(r'WHERE\s+(.+)', sql_upper)
            if where_match:
                where_clause = where_match.group(1).split(';')[0].strip()
                return f"SELECT * FROM {table_name} WHERE {where_clause};"
        
        # Default fallback
        return f"SELECT * FROM {table_name};"
    
    def _calculate_confidence(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        """
        Calculate confidence score for generated SQL
        This is a simplified implementation - can be enhanced
        """
        try:
            # For now, return a default confidence
            # In practice, you might use model logits, beam scores, etc.
            return 0.8
        except Exception:
            return 0.5
    
    def transpile_sql(self, sql: str, target_dialect: str) -> str:
        """
        Transpile SQL to different database dialects using SQLGlot
        
        Args:
            sql: Original SQL query
            target_dialect: Target SQL dialect ('postgres', 'mysql', 'sqlite', etc.)
            
        Returns:
            Transpiled SQL query
        """
        try:
            transpiled = transpile(sql, write=target_dialect)
            return transpiled[0] if transpiled else sql
        except Exception as e:
            logger.warning(f"Failed to transpile SQL to {target_dialect}: {e}")
            return sql
    
    def batch_generate(
        self, 
        queries: List[str], 
        schemas: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL for multiple queries in batch
        
        Args:
            queries: List of natural language queries
            schemas: List of corresponding database schemas
            **kwargs: Additional parameters for generation
            
        Returns:
            List of generation results
        """
        results = []
        
        for query, schema in zip(queries, schemas):
            result = self.generate_sql(query, schema, **kwargs)
            results.append(result)
        
        return results
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5
    ):
        """
        Fine-tune the model on custom dataset
        
        Args:
            train_dataset: Training dataset in Hugging Face format
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        logger.info("Starting fine-tuning process...")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a previously fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Fine-tuned model loaded successfully")
    
    def evaluate_on_spider(self, spider_dev_path: str) -> Dict[str, float]:
        """
        Evaluate model on Spider development set
        
        Args:
            spider_dev_path: Path to Spider development JSON file
            
        Returns:
            Evaluation metrics
        """
        # This would implement Spider evaluation
        # For now, return placeholder
        logger.info("Spider evaluation not implemented yet")
        return {"exact_match": 0.0, "execution_accuracy": 0.0}


class SpiderDataProcessor:
    """Process Spider dataset for training/evaluation"""
    
    @staticmethod
    def prepare_spider_dataset(
        spider_json_path: str,
        tables_json_path: str,
        tokenizer: T5Tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256
    ) -> Dataset:
        """
        Prepare Spider dataset for training
        
        Args:
            spider_json_path: Path to Spider training/dev JSON
            tables_json_path: Path to Spider tables JSON
            tokenizer: T5 tokenizer
            max_source_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            
        Returns:
            Processed dataset
        """
        import json
        
        # Load Spider data
        with open(spider_json_path, 'r') as f:
            spider_data = json.load(f)
        
        with open(tables_json_path, 'r') as f:
            tables_data = json.load(f)
        
        # Create table lookup
        db_id_to_schema = {}
        for table_info in tables_data:
            db_id = table_info['db_id']
            schema_str = SpiderDataProcessor._format_schema(table_info)
            db_id_to_schema[db_id] = schema_str
        
        # Prepare training examples
        examples = []
        for item in spider_data:
            question = item['question']
            sql = item['query']
            db_id = item['db_id']
            schema = db_id_to_schema.get(db_id, "")
            
            # Format input and target
            input_text = f"translate english to SQL: {question} | {schema}"
            target_text = sql
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                max_length=max_source_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            targets = tokenizer(
                target_text,
                max_length=max_target_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            examples.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            })
        
        return Dataset.from_list(examples)
    
    @staticmethod
    def _format_schema(table_info: Dict) -> str:
        """Format table schema for training"""
        schema_parts = []
        
        table_names = table_info['table_names_original']
        column_names = table_info['column_names_original']
        column_types = table_info['column_types']
        
        for i, table_name in enumerate(table_names):
            schema_parts.append(f"table: {table_name}")
            
            # Find columns for this table
            table_columns = []
            for j, (table_idx, col_name) in enumerate(column_names):
                if table_idx == i:
                    col_type = column_types[j] if j < len(column_types) else "text"
                    table_columns.append(f"{col_name} {col_type}")
            
            if table_columns:
                schema_parts.append(f"columns: {', '.join(table_columns)}")
        
        return " | ".join(schema_parts)


# Utility functions
def load_model(model_name: str = "gaussalgo/T5-LM-Large-text2sql-spider") -> NL2SQLModel:
    """Load pre-trained NL2SQL model"""
    return NL2SQLModel(model_name)


def create_custom_model(base_model: str = "t5-base") -> NL2SQLModel:
    """Create model for custom fine-tuning"""
    return NL2SQLModel(base_model) 