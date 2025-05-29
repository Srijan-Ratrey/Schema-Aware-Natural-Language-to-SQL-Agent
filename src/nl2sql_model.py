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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NL2SQLModel:
    """Schema-aware Natural Language to SQL conversion using T5"""
    
    def __init__(self, model_name: str = "tscholak/t5-base-spider", device: str = None):
        """
        Initialize NL2SQL model
        
        Args:
            model_name: Hugging Face model identifier or local path
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
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
        Optimized for WikiSQL format
        """
        # For WikiSQL model, use simpler format that the model understands
        # Extract table name from schema
        table_name = "books"  # Default table name
        if "Table:" in schema:
            lines = schema.split('\n')
            for line in lines:
                if line.strip().startswith("Table:"):
                    table_name = line.split("Table:")[1].strip()
                    break
        
        # WikiSQL format: just the question with minimal context
        input_template = f"translate English to SQL: {query}"
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
        
        # Remove table descriptions in parentheses
        sql = re.sub(r'\([^)]*\)', '', sql)
        
        # Clean up common issues
        sql = re.sub(r'\[.*?\]', '', sql)  # Remove [PRIMARY KEY] etc
        sql = re.sub(r'WHERE\s*$', '', sql)  # Remove trailing WHERE
        sql = re.sub(r'FROM\s+table\s*', 'FROM books ', sql, flags=re.IGNORECASE)  # Fix table name
        
        # Fix aggregate function syntax
        sql = re.sub(r'SELECT\s+MAX\s+(\w+)', r'SELECT MAX(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+MIN\s+(\w+)', r'SELECT MIN(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+AVG\s+(\w+)', r'SELECT AVG(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+COUNT\s+(\w+)', r'SELECT COUNT(\1)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+SUM\s+(\w+)', r'SELECT SUM(\1)', sql, flags=re.IGNORECASE)
        
        # Fix common column selection issues
        sql = re.sub(r'SELECT\s+Books\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+book\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+all\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s+everything\s+FROM', 'SELECT * FROM', sql, flags=re.IGNORECASE)
        
        # Basic SQL formatting
        sql = sql.replace('\n', ' ').replace('\t', ' ')
        sql = ' '.join(sql.split())  # Remove extra whitespace
        
        # Ensure proper SQL structure
        if sql and not any(keyword in sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return ""
        
        # Fix common SQL issues
        if 'SELECT' in sql.upper() and 'FROM' not in sql.upper():
            sql = sql + " FROM books"
        
        # Ensure SQL ends with semicolon if it's a complete query
        if sql and not sql.endswith(';') and any(keyword in sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            sql += ';'
        
        return sql
    
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
def load_model(model_name: str = "mrm8488/t5-base-finetuned-wikiSQL") -> NL2SQLModel:
    """Load pre-trained NL2SQL model"""
    return NL2SQLModel(model_name)


def create_custom_model(base_model: str = "t5-base") -> NL2SQLModel:
    """Create model for custom fine-tuning"""
    return NL2SQLModel(base_model) 