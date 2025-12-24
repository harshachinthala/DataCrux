
"""
Gradio UI for Excel RAG Pipeline
"""

import gradio as gr
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import shutil
from langchain_community.document_loaders import CSVLoader
import pandas as pd
import plotly.graph_objects as go

from ..config import settings
from ..ingestion.excel_parser import ExcelParser
from ..ingestion.chunker import ExcelChunker
from ..embeddings.embedding_generator import create_embedding_generator
from ..embeddings.vector_store import VectorStore
from ..retrieval.retriever import Retriever
from ..llm.rag_chain import RAGChain
from ..visualization.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)


class ExcelRAGApp:
    """Gradio application for Excel RAG pipeline"""
    
    def __init__(self):
        """Initialize the application"""
        self.parser = ExcelParser()
        self.chunker = ExcelChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            strategy=settings.chunking_strategy
        )
        
        # Initialize embedding generator
        self.embedding_generator = create_embedding_generator(
            provider=settings.embedding_model,
            model_name=(
                settings.huggingface_model
                if settings.embedding_model == "huggingface"
                else settings.openai_embedding_model
            ),
            api_key=settings.openai_api_key
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,
            index_path=settings.vector_db_dir / "index"
        )
        
        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=settings.top_k_retrieval,
            score_threshold=settings.similarity_threshold
        )
        
        # Defer RAG chain initialization until first query (requires API key)
        self.rag_chain = None
        
        # Initialize chart generator
        self.chart_generator = ChartGenerator()
        
        # State
        self.current_file_data = None
        self.uploaded_files = []
        
        logger.info("Excel RAG App initialized")
    
    def _clean_and_normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize DataFrame by converting string numbers to actual numbers
        and handling common data quality issues.
        """
        df = df.copy()
        import numpy as np
        
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Try to convert object/string columns to numeric
            try:
                if df[col].dtype == 'object':
                    # Create a temporary series for cleanup
                    # Remove currency symbols ($, €), commas, and extra spaces
                    clean_series = df[col].astype(str).str.replace(r'[$,€\s]', '', regex=True)
                    
                    # Check if the column actually looks numeric (allow some NaNs)
                    # We try to convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(clean_series, errors='coerce')
                    
                    # If we recovered a significant amount of numbers (e.g. > 50% non-nulls are numbers)
                    # then we accept this conversion
                    non_null_count = clean_series.replace(['nan', 'None', ''], np.nan).dropna().count()
                    numeric_count = numeric_series.dropna().count()
                    
                    if non_null_count > 0 and numeric_count / non_null_count > 0.5:
                        df[col] = numeric_series
                        logger.info(f"Converted column '{col}' to numeric")
            except Exception as e:
                logger.debug(f"Could not convert column {col} to numeric: {e}")
                pass
                
        return df

    def upload_file(self, file) -> Tuple[str, str]:
        """Handle file upload using LangChain loaders, chunk, embed and store."""
        if file is None:
            return "No file uploaded", ""

        try:
            # Save uploaded file
            file_path = Path(file.name)
            dest_path = settings.upload_dir / file_path.name
            shutil.copy(file.name, dest_path)

            logger.info(f"Processing uploaded file: {dest_path.name}")

            # ---------- Load data ----------
            documents = []
            data_path = dest_path.parent

            # CSV files
            csv_files = list(data_path.glob("**/*.csv"))
            logger.debug(f"Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
            
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for csv_file in csv_files:
                for encoding in encodings:
                    try:
                        loader = CSVLoader(str(csv_file), encoding=encoding)
                        loaded = loader.load()
                        logger.debug(f"Loaded {len(loaded)} CSV docs from {csv_file} using {encoding}")
                        documents.extend(loaded)
                        break  # Success
                    except Exception as e:
                        logger.warning(f"Failed to load CSV {csv_file} with {encoding}: {e}")
                        if encoding == encodings[-1]:
                            logger.error(f"All encodings failed for {csv_file}")

            # Excel files
            xlsx_files = list(data_path.glob("**/*.xlsx"))
            logger.debug(f"Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
            for xlsx_file in xlsx_files:
                try:
                    # Use pandas to load Excel, avoiding unstructured dependency
                    df_excel = pd.read_excel(str(xlsx_file))
                    # Convert rows to documents
                    from langchain_core.documents import Document
                    loaded = []
                    for i, row in df_excel.iterrows():
                        content = "\n".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
                        meta = {"source": str(xlsx_file), "row": i}
                        loaded.append(Document(page_content=content, metadata=meta))
                        
                    logger.debug(f"Loaded {len(loaded)} Excel docs from {xlsx_file}")
                    documents.extend(loaded)
                except Exception as e:
                    logger.error(f"Failed to load Excel {xlsx_file}: {e}")

            # ---------- Chunk & store ----------
            # Convert LangChain Documents to our internal Chunk objects
            chunks = self.chunker.chunk_documents(documents)
            self.vector_store.add_chunks(chunks)
            
            # Track uploaded file
            self.uploaded_files.append(file_path.name)

            # ---------- UI feedback ----------
            # Generate preview using pandas
            try:
                if dest_path.suffix.lower() == '.csv':
                    # Try multiple encodings for preview
                    df = None
                    last_error = None
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(dest_path, encoding=encoding)
                            break
                        except Exception as e:
                            last_error = e
                    
                    if df is None:
                        raise last_error or Exception("Failed to read CSV with supported encodings")
                else:
                    df = pd.read_excel(dest_path)
                
                
                # Clean and normalize data immediately
                df = self._clean_and_normalize_df(df)
                
                # Store for potential direct access
                self.current_file_data = df
                
                # Create preview HTML
                preview_html = f"""
                <div style="max-height: 400px; overflow-y: auto; overflow-x: auto;">
                    {df.head(10).to_html(index=False, classes="preview-table")}
                </div>
                """
            except Exception as e:
                logger.error(f"Failed to generate preview: {e}")
                preview_html = f"<p>⚠️ Could not generate preview: {e}</p>"

            status = (
                "✅ **File uploaded and indexed!**\n\n"
                f"- Total documents: {len(documents)}\n"
                f"- Created {len(chunks)} chunks\n"
                f"- Vector store now holds {self.vector_store.get_stats()['total_vectors']} vectors\n"
                "You can now ask questions about the data."
            )
            return status, preview_html

        except Exception as e:
            logger.exception("Error uploading file")
            return f"❌ Upload failed: {e}", ""
        

    def _create_simple_chart_from_data(self, df: pd.DataFrame, query: str) -> Optional[go.Figure]:
        """
        Create a simple chart directly from computed DataFrame
        This is the ultimate fallback when both LLM and chart_generator fail
        
        Args:
            df: Computed data DataFrame
            query: User's query for context
            
        Returns:
            Plotly Figure or None
        """
        import plotly.express as px
        import numpy as np
        
        try:
            # Clean the dataframe first
            df = df.copy()
            
            # Convert any string numbers to actual numbers
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Remove currency symbols and commas
                        cleaned = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned, errors='ignore')
                    except:
                        pass
            
            # Get numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.info(f"Creating chart from data with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
            # Strategy 1: If we have 1 categorical + 1 numeric -> Bar chart
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                # Limit to reasonable number of categories
                if len(df[x_col].unique()) > 50:
                    df = df.head(50)
                
                logger.info(f"Creating bar chart: {x_col} vs {y_col}")
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}"
                )
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    showlegend=False
                )
                return fig
            
            # Strategy 2: If we have 2+ numeric columns -> Scatter plot
            elif len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                # Limit data points
                plot_df = df.head(1000)
                
                logger.info(f"Creating scatter plot: {x_col} vs {y_col}")
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                return fig
            
            # Strategy 3: Single numeric column -> Histogram
            elif len(numeric_cols) == 1:
                col = numeric_cols[0]
                
                logger.info(f"Creating histogram for {col}")
                fig = px.histogram(
                    df.head(1000),
                    x=col,
                    title=f"Distribution of {col}"
                )
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=col,
                    yaxis_title="Count"
                )
                return fig
            
            else:
                logger.warning("No suitable columns found for visualization")
                return None
                
        except Exception as e:
            logger.error(f"Error in _create_simple_chart_from_data: {e}", exc_info=True)
            return None
    
    def process_query(self, query: str, chat_history: List) -> Tuple[List, Optional[gr.Plot]]:
        """
        Process user query
        
        Args:
            query: User question
            chat_history: Chat history in Gradio messages format
            
        Returns:
            Tuple of (updated chat history, chart figure)
        """
        if not query.strip():
            return chat_history, None
        
        # Check if API key is configured
        if not settings.openai_api_key:
            error_msg = (
                "⚠️ **OpenAI API key not configured!**\n\n"
                "Please add your API key to the `.env` file:\n"
                "1. Copy `.env.example` to `.env`\n"
                "2. Add your key: `OPENAI_API_KEY=sk-your-key-here`\n"
                "3. Restart the application"
            )
            # Tuples format
            chat_history.append([query, error_msg])
            return chat_history, None
        
        if self.vector_store.get_stats().get("status") == "empty":
            chat_history.append([query, "⚠️ Please upload an Excel file first!"])
            return chat_history, None
        
        # Initialize variables to prevent UnboundLocalError
        chart_fig = None
        execution_result = None
        computed_data = None
        analysis = ""
        
        try:
            # Lazy initialize RAG chain if not already created
            if self.rag_chain is None:
                logger.info("Initializing RAG chain...")
                self.rag_chain = RAGChain(
                    retriever=self.retriever,
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    api_key=settings.openai_api_key
                )
            
            # Generate schema description from current data
            schema_desc = "Schema not available"
            sample_data_desc = ""
            
            if self.current_file_data is not None:
                # Get schema
                buffer = []
                buffer.append("Available Columns:")
                for col in self.current_file_data.columns:
                    dtype = self.current_file_data[col].dtype
                    buffer.append(f"- {col} ({dtype})")
                schema_desc = "\n".join(buffer)
                
                # Get sample data (first 5 rows)
                sample_data_desc = "\nSample Data (first 5 rows):\n"
                sample_data_desc += self.current_file_data.head(5).to_string(index=False)
                
                logger.info(f"Generated schema with {len(self.current_file_data.columns)} columns")

            # Get LLM response (code generation)
            result = self.rag_chain.query(
                user_query=query,
                schema=schema_desc + sample_data_desc
            )
            
            if not result.get("success"):
                error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                chat_history.append([query, error_msg])
                return chat_history, None
            
            # Extract analysis and code
            analysis = result["response"].get("analysis", "")
            code = result["response"].get("code")
            
            # Execute code if we have it and data
            if code and self.current_file_data is not None:
                logger.info("Executing generated code...")
                
                # Import code executor
                from ..execution.code_executor import CodeExecutor
                executor = CodeExecutor(timeout_seconds=10)
                
                # Execute code
                execution_result = executor.execute(code, self.current_file_data)
                
                if execution_result["success"]:
                    chart_fig = execution_result.get("fig")
                    computed_data = execution_result.get("result_data")
                    
                    # Validate figure has data
                    if chart_fig:
                        try:
                            # Check if figure has traces/data
                            if not chart_fig.data:
                                logger.warning("Generated figure has no data traces. Discarding.")
                                chart_fig = None
                            # Check if traces have actual data points
                            elif all(len(trace.x or []) == 0 and len(trace.y or []) == 0 for trace in chart_fig.data):
                                logger.warning("Generated figure traces are empty. Discarding.")
                                chart_fig = None
                        except Exception as e:
                            logger.warning(f"Error validating figure data: {e}")
                            # Keep fig if validation fails but object exists, or set to None?
                            # Safer to trust it if we can't inspect it, unless it's clearly broken.
                            pass

                    # Generate insights based on computed results
                    insights_text = self._generate_insights(
                        query=query,
                        analysis=analysis,
                        result_data=computed_data,
                        context=result.get("raw_response", "")
                    )
                    
                    # Build response with analysis and results
                    response_parts = ["# 📊 Analysis Results\n"]
                    
                    if insights_text:
                        response_parts.append(f"{insights_text}\n")
                    elif analysis:
                        response_parts.append(f"**Analysis**: {analysis}\n")
                    
                    # Show computed results if available
                    if computed_data is not None:
                        response_parts.append("\n## Computed Results:")
                        if isinstance(computed_data, pd.DataFrame):
                            # Limit display to first 10 rows
                            display_df = computed_data.head(10)
                            response_parts.append(f"\n```\n{display_df.to_string(index=False)}\n```")
                            if len(computed_data) > 10:
                                response_parts.append(f"\n*Showing first 10 of {len(computed_data)} rows*")
                        elif isinstance(computed_data, pd.Series):
                            response_parts.append(f"\n```\n{computed_data.to_string()}\n```")
                        else:
                            response_parts.append(f"\n```\n{computed_data}\n```")
                    
                    if chart_fig:
                        response_parts.append("\n## 📈 Visualization\nSee the chart below.")
                    
                    response_text = "\n".join(response_parts)
                else:
                # Code execution failed
                    error = execution_result.get("error", "Unknown error")
                    response_text = f"# ⚠️ Execution Error\n\n**Analysis**: {analysis}\n\n**Error**: {error}\n\nThe generated code could not be executed. Please try rephrasing your question."
                    logger.error(f"Code execution failed: {error}")
            else:
                # No code generated or no data
                response_text = f"# 📊 Analysis\n\n{analysis if analysis else 'No analysis available.'}"


            # ULTIMATE FALLBACK: If we have result data but no chart, create one directly
            if chart_fig is None and computed_data is not None and isinstance(computed_data, pd.DataFrame) and len(computed_data) > 0:
                logger.info("No chart generated. Creating simple chart directly from computed data...")
                try:
                    chart_fig = self._create_simple_chart_from_data(computed_data, query)
                    if chart_fig:
                        logger.info("Direct chart created successfully")
                        response_text += "\n\n## 📈 Visualization\nSee the chart below."
                except Exception as e:
                    logger.error(f"Direct chart creation failed: {e}", exc_info=True)
            
            
            # Update chat history
            chat_history.append([query, response_text])
            
            # CRITICAL FIX for HF Spaces: Ensure figure is properly formatted
            if chart_fig is not None:
                try:
                    # Force update layout to ensure all data is serialized
                    if chart_fig.layout is None:
                        chart_fig.update_layout(template="plotly_white")
                        
                    chart_fig.update_layout(
                        autosize=True,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    # Ensure figure has proper data
                    if not chart_fig.data or len(chart_fig.data) == 0:
                        logger.warning("Figure has no data, setting to None")
                        chart_fig = None
                except Exception as e:
                    logger.error(f"Error updating figure layout: {e}")
                    chart_fig = None
            
            return chat_history, chart_fig
        
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            error_msg = f"❌ Error processing query: {str(e)}"
            chat_history.append([query, error_msg])
            return chat_history, None
    
    def _generate_insights(
        self,
        query: str,
        analysis: str,
        result_data: Any,
        context: str
    ) -> str:
        """
        Generate narrative insights based on computed results
        
        Args:
            query: User's original query
            analysis: Initial analysis from code generation
            result_data: Computed results from code execution
            context: Retrieved context from vector DB
            
        Returns:
            Narrative insights text
        """
        try:
            # Convert result_data to string for LLM
            if result_data is not None:
                if isinstance(result_data, pd.DataFrame):
                    results_str = result_data.to_string(index=False)
                elif isinstance(result_data, pd.Series):
                    results_str = result_data.to_string()
                else:
                    results_str = str(result_data)
            else:
                results_str = "No computed results available"
            
            # Create insights prompt
            insights_prompt = f"""Based on the following computed results, provide detailed insights and interpretation.

USER QUESTION: {query}

COMPUTED RESULTS:
{results_str}

Provide:
1. Key findings from the data
2. Trends or patterns observed
3. Business insights or recommendations
4. Any notable observations

Keep your response concise but informative (3-5 bullet points)."""

            # Call LLM for insights
            from langchain_openai import ChatOpenAI
            from ..config import settings
            llm = ChatOpenAI(
                model=settings.llm_model, 
                temperature=0.3,
                api_key=settings.openai_api_key
            )
            
            response = llm.invoke(insights_prompt)
            insights = response.content.strip()
            
            return f"## 💡 Insights\n\n{insights}"
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return ""
    
    def clear_data(self) -> Tuple[str, str, List]:
        """
        Clear all uploaded data
        
        Returns:
            Tuple of (status, preview, chat history)
        """
        self.vector_store.delete()
        self.current_file_data = None
        self.uploaded_files = []
        
        # Recreate empty vector store
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,
            index_path=settings.vector_db_dir / "index"
        )
        
        # Recreate retriever and RAG chain
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=settings.top_k_retrieval,
            score_threshold=settings.similarity_threshold
        )
        
        # Reset RAG chain (will be recreated on next query)
        self.rag_chain = None
        
        return "🗑️ All data cleared. Upload a new file to start.", "", []
    
    def clear_chart(self) -> None:
        """Clear the chart output"""
        return None
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with modern DataCrux branding"""
        
        with gr.Blocks(
            title="DataCrux - The Essential Core of Your Data's Intelligence"
        ) as interface:
            
            # Custom CSS for gradient background and styling
            gr.HTML("""
                <style>
                .gradio-container {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                }
                h1, h2, h3, p, span {
                    font-family: 'Inter', sans-serif;
                }
                .preview-container {
                    background: white;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                </style>
            """)
            
            # Header
            gr.HTML("""
                <div style="text-align: center; background: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.15);">
                    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">✦ DataCrux ✦</h1>
                    <p style="color: #6b7280; font-size: 1.2rem; font-style: italic; margin-top: 0.5rem;">"The Essential Core of Your Data's Intelligence."</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Data Upload")
                    
                    file_input = gr.File(
                        label="Upload Excel File (.xlsx, .xls, .csv)",
                        file_types=[".xlsx", ".xls", ".csv"]
                    )
                    
                    # Using explicit upload button as requested by user flow
                    upload_btn = gr.Button("📤 Process File", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear All Data", variant="stop")
                    
                    status_output = gr.Markdown(label="Status")
                    
                    gr.Markdown("### � Data Preview")
                    # Using HTML for preview to support the scrollable table from upload_file
                    preview_output = gr.HTML(
                        label="Preview",
                        elem_classes="preview-container"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Ask Questions About Your Data")
                    
                    chatbot = gr.Chatbot(type="tuples",
                        label="DataCrux Assistant",
                        height=400
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask me anything about your data... (e.g., 'Show average sales by region')",
                            scale=4
                        )
                        submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)
                    
                    gr.Markdown("### 📈 Visualization")
                    with gr.Row():
                        chart_output = gr.Plot(label="Visualization")
                    with gr.Row():
                        clear_chart_btn = gr.Button("🗑️ Clear Chart", size="sm", variant="secondary")
            
            # Footer
            gr.HTML("""
                <div style="text-align: center; padding: 1.5rem; color: white; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 1rem;">
                    <p style="margin: 0; font-size: 0.95rem;">Powered by RAG + LangChain + OpenAI + HuggingFace + Gradio</p>
                </div>
            """)
            
            # Event handlers
            # Auto-upload when file is selected/dropped
            file_input.upload(
                fn=self.upload_file,
                inputs=[file_input],
                outputs=[status_output, preview_output],
                api_name="upload_file"
            )
            
            # Manual upload button (alternative)
            upload_btn.click(
                fn=self.upload_file,
                inputs=[file_input],
                outputs=[status_output, preview_output],
                api_name="upload_file_manual"
            )
            
            submit_btn.click(
                fn=self.process_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, chart_output],
                api_name="chat"
            )
            
            query_input.submit(
                fn=self.process_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, chart_output]
            )
            
            clear_btn.click(
                fn=self.clear_data,
                inputs=[],
                outputs=[status_output, preview_output, chatbot]
            )
            
            clear_chart_btn.click(
                fn=self.clear_chart,
                inputs=[],
                outputs=[chart_output]
            )
        
        return interface
    
    def launch(self):
        """Launch the Gradio app"""
        interface = self.create_interface()
        
        # Enable queueing for better performance
        interface.queue()
        
        # Launch using defaults - Hugging Face handles the rest
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            allowed_paths=[str(settings.data_dir.resolve())] if settings.data_dir.exists() else []
        )
