# Permanent Visualization Fix Plan

## Root Causes Identified:
1. LLM requests unsupported chart types (e.g., "heatmap")
2. LLM creates non-existent column names ("Age Group", "Average Income")
3. Preprocessing logic exists but validation still fails
4. No fallback when chart generation fails

## Permanent Solution:

### 1. Simplify the entire flow - Use ONLY existing columns
- Remove all preprocessing logic
- Force LLM to use exact column names from schema
- Let Plotly handle aggregations natively

### 2. Add supported chart types
- Add heatmap support using plotly.express.imshow()
- Update prompt to list ONLY supported types

### 3. Robust fallback
- If chart fails, return a simple bar chart of first 2 numeric columns
- Always show SOMETHING to the user

## Implementation:
1. Update prompt to be VERY strict about column names
2. Simplify chart_generator to use columns as-is
3. Add heatmap support
4. Add fallback chart generation
