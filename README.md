
# DataCrux

An advanced Retrieval-Augmented Generation (RAG) system for analyzing Excel data through natural language queries with AI-powered insights and visualizations.

Demo Link : https://huggingface.co/spaces/harshachinthala/DataCrux

### 🧠 Tech Stack & Skills
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI-success)
![RAG](https://img.shields.io/badge/Architecture-RAG-important)
![LangChain](https://img.shields.io/badge/LangChain-Framework-black)
![VectorDB](https://img.shields.io/badge/Vector%20DB-FAISS-orange)
![Embeddings](https://img.shields.io/badge/Embeddings-HuggingFace%20%7C%20OpenAI-purple)
![Gradio](https://img.shields.io/badge/UI-Gradio-lightgrey)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-blueviolet)
![Excel](https://img.shields.io/badge/Data-Excel%20%7C%20CSV-green)
![Semantic Search](https://img.shields.io/badge/Search-Semantic-yellowgreen)
![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)
![Deployment](https://img.shields.io/badge/Deployment-HuggingFace%20%7C%20Render-lightgrey)


## Features

- 📊 **Excel File Support**: Upload `.xlsx`, `.xls`, and `.csv` files
- 🤖 **Natural Language Queries**: Ask questions about your data in plain English
- 🔍 **Semantic Search**: Intelligent retrieval using vector embeddings
- 📈 **Auto-Visualizations**: AI-generated charts (bar, line, scatter, pie, histogram)
- 💡 **Smart Insights**: Analytical commentary and trend detection
- 🎯 **Multiple Chunking Strategies**: Row-based, group-based, sheet-based, or sliding window
- 🔧 **Flexible Embeddings**: Support for HuggingFace and OpenAI models

## Architecture

```
                          User Query 
                              ↓
                          Gradio UI 
                              ↓
                          Retriever 
                              ↓
                       Vector DB (FAISS)
                              ↓
                         Top-K Chunks
                              ↓
                    LLM (OpenAI + LangChain)
                              ↓
              Structured Response + Chart Spec
                              ↓
                    Plotly Chart Renderer
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd ML_class_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Run the Application

```bash
python main.py
```

The Gradio UI will launch at `http://localhost:7860`

## Deployment

### Deploy to Render (Free)

1. **Create a new Web Service** on [Render](https://render.com)
2. **Connect your GitHub repository**
3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. **Deploy!**

### Deploy to Hugging Face Spaces

1. **Create a new Space** on [Hugging Face](https://huggingface.co/spaces)
2. **Choose "Gradio" SDK** (or convert to Gradio)
3. **Upload files** and deploy


## Usage

1. **Upload Excel File**: Click "Upload Excel File" and select your `.xlsx`, `.xls`, or `.csv` file
2. **Preview Data**: View the first 10 rows and file statistics
3. **Ask Questions**: Type natural language questions like:
   - "What is the total sales by region?"
   - "Show me the trend of monthly revenue"
   - "Which product has the highest profit margin?"
   - "Compare Q1 vs Q2 performance"
4. **View Results**: Get structured answers with:
   - High-level summary
   - Detailed breakdowns
   - Analytical insights
   - Interactive visualizations

## Configuration Options

Edit `.env` to customize:

### Embedding Models
- **HuggingFace** (default): `EMBEDDING_MODEL=huggingface`
  - Free, runs locally
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
- **OpenAI**: `EMBEDDING_MODEL=openai`
  - Requires API credits
  - Model: `text-embedding-3-small`

### LLM Models
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o` (more capable)
- `gpt-3.5-turbo` (fastest)

### Chunking Strategies
- `row`: One chunk per row (best for detailed queries)
- `group`: Multiple rows per chunk (reduces total chunks)
- `sheet`: Summary per sheet (high-level overview)
- `sliding`: Overlapping windows (preserves context)

### Retrieval Settings
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum relevance score (default: 0.5)

## Project Structure

```
ML_class_project/
├── src/
│   ├── config.py              # Configuration management
│   ├── ingestion/             # Excel parsing & chunking
│   ├── embeddings/            # Embedding generation & vector store
│   ├── retrieval/             # Query processing & retrieval
│   ├── llm/                   # RAG chain & prompts
│   ├── visualization/         # Chart generation
│   └── ui/                    # Gradio interface
├── data/
│   ├── uploads/               # Uploaded files
│   ├── vector_db/             # FAISS index
│   └── samples/               # Sample data
├── logs/                      # Application logs
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── .env                       # Configuration (create from .env.example)
```

## Advanced Features

### Custom Chunking Strategy

```python
# In .env
CHUNKING_STRATEGY=group  # Change to 'row', 'sheet', or 'sliding'
CHUNK_SIZE=500           # Maximum characters per chunk
CHUNK_OVERLAP=50         # Overlap for sliding window
```

### Metadata Filtering

The system automatically tracks:
- Source filename
- Sheet name
- Row indices
- Column names
- Chunk type

### Response Format

Every response follows a structured format:

1. **HIGH_LEVEL_ANSWER**: 2-6 sentence summary
2. **DETAILS**: Bullet points with figures and references
3. **INSIGHTS**: 2-5 analytical observations
4. **CHART_SUGGESTION**: JSON specification for visualization

## Troubleshooting

### "No relevant context found"
- Ensure your Excel file is uploaded and processed
- Try rephrasing your question
- Check if the data contains the information you're asking about

### Slow performance
- Reduce `TOP_K_RETRIEVAL` in `.env`
- Use `chunking_strategy=group` to reduce total chunks
- Switch to `gpt-4o-mini` for faster responses

### Chart not rendering
- Ensure your query asks for data that can be visualized
- Check that column names in the chart spec match your data
- Review logs in `logs/app.log` for errors

## Example Queries

**Sales Analysis:**
- "What are the top 5 products by revenue?"
- "Show me sales trends over the last 12 months"
- "Compare performance across regions"

**Financial Analysis:**
- "What is the average profit margin by category?"
- "Which month had the highest expenses?"
- "Calculate year-over-year growth rate"

**Customer Analytics:**
- "How many unique customers do we have?"
- "What is the customer retention rate?"
- "Show customer distribution by segment"

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Adding New Chart Types

Edit `src/visualization/chart_generator.py` and add your chart method to `chart_type_map`.

### Custom System Prompts

Modify `src/llm/prompts.py` to customize the RAG assistant behavior.

## License

MIT License

## Support

For issues or questions, check the logs in `logs/app.log` or review the implementation plan in the artifacts directory.
 
 
