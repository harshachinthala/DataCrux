"""
Generate sample Excel file for testing
"""

import pandas as pd
from pathlib import Path

# Create sample sales data
data = {
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West',
               'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
    'Product': ['Widget A', 'Widget A', 'Widget A', 'Widget A',
                'Widget B', 'Widget B', 'Widget B', 'Widget B',
                'Widget C', 'Widget C', 'Widget C', 'Widget C',
                'Widget D', 'Widget D', 'Widget D', 'Widget D'],
    'Sales': [120000, 98000, 75000, 65000,
              85000, 72000, 68000, 55000,
              95000, 88000, 78000, 70000,
              110000, 92000, 81000, 74000],
    'Units': [1200, 980, 750, 650,
              850, 720, 680, 550,
              950, 880, 780, 700,
              1100, 920, 810, 740],
    'Profit_Margin': [0.25, 0.22, 0.20, 0.18,
                      0.30, 0.28, 0.26, 0.24,
                      0.27, 0.25, 0.23, 0.21,
                      0.28, 0.26, 0.24, 0.22]
}

df = pd.DataFrame(data)

# Save to Excel
output_path = Path('data/samples/sample_sales_data.xlsx')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(output_path, index=False, sheet_name='Sales')

print(f"Sample file created: {output_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
