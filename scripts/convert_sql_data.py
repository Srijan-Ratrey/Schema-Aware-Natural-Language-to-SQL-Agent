#!/usr/bin/env python3
"""
Convert SQL Server sample data to SQLite compatible format
"""

import re
from pathlib import Path
from datetime import datetime

def convert_sql_data():
    """Convert SQL Server data to SQLite compatible format."""
    
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "sample-data.sql"
    output_file = project_root / "data" / "sample-data-sqlite.sql"
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return False
    
    print(f"Converting {input_file} to SQLite format...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Remove SET IDENTITY_INSERT lines (with or without brackets/quotes, any whitespace)
        if re.match(r"SET IDENTITY_INSERT.*", line_stripped):
            continue
        # Remove square brackets and double quotes from table/column names
        line = re.sub(r'\[(\w+)\]', r'\1', line)
        line = re.sub(r'"(\w+)"', r'\1', line)
        # Fix booleans: last value in Product/OrderItem is 0/1, keep as is for SQLite
        # Fix Order table: ensure column order matches schema (Id, OrderDate, OrderNumber, CustomerId, TotalAmount)
        if line.startswith('INSERT INTO Order '):
            # Extract values
            m = re.match(r"INSERT INTO Order \(([^)]+)\)VALUES\((.+)\)", line)
            if m:
                columns = [c.strip() for c in m.group(1).split(',')]
                values = [v.strip() for v in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", m.group(2))]
                # Map columns to values
                col_val = dict(zip(columns, values))
                # Convert date to YYYY-MM-DD
                date_val = col_val.get('OrderDate', '').strip("'")
                try:
                    # Try parsing SQL Server date string
                    dt = datetime.strptime(date_val, "%b  %d %Y %I:%M:%S:%f%p")
                    date_val = dt.strftime("%Y-%m-%d")
                except Exception:
                    # If parsing fails, keep as is
                    pass
                # Reorder values
                new_values = [
                    col_val.get('Id', ''),
                    f"'{date_val}'" if date_val else "NULL",
                    col_val.get('OrderNumber', 'NULL'),
                    col_val.get('CustomerId', 'NULL'),
                    col_val.get('TotalAmount', 'NULL')
                ]
                line = f"INSERT INTO Order (Id,OrderDate,OrderNumber,CustomerId,TotalAmount) VALUES({','.join(new_values)})\n"
        # Convert NULLs (keep as is)
        # Remove trailing whitespace
        line = line.rstrip()
        if line:
            converted_lines.append(line + '\n')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(converted_lines)
    
    print(f"Converted data saved to: {output_file}")
    return True

if __name__ == "__main__":
    convert_sql_data() 