"""
Gradio UI for Excel RAG Pipeline - Clean Insight Edition
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
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                if df[col].dtype == 'object':
                    clean_series = df[col].astype(str).str.replace(r'[$,€\s]', '', regex=True)
                    numeric_series = pd.to_numeric(clean_series, errors='coerce')
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
            file_path = Path(file.name)
            dest_path = settings.upload_dir / file_path.name
            shutil.copy(file.name, dest_path)
            logger.info(f"Processing uploaded file: {dest_path.name}")

            documents = []
            data_path = dest_path.parent
            csv_files = list(data_path.glob("**/*.csv"))
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for csv_file in csv_files:
                for encoding in encodings:
                    try:
                        loader = CSVLoader(str(csv_file), encoding=encoding)
                        loaded = loader.load()
                        documents.extend(loaded)
                        break
                    except Exception as e:
                        if encoding == encodings[-1]:
                            logger.error(f"All encodings failed for {csv_file}")

            xlsx_files = list(data_path.glob("**/*.xlsx"))
            for xlsx_file in xlsx_files:
                try:
                    df_excel = pd.read_excel(str(xlsx_file))
                    from langchain_core.documents import Document
                    loaded = []
                    for i, row in df_excel.iterrows():
                        content = "\n".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
                        meta = {"source": str(xlsx_file), "row": i}
                        loaded.append(Document(page_content=content, metadata=meta))
                    documents.extend(loaded)
                except Exception as e:
                    logger.error(f"Failed to load Excel {xlsx_file}: {e}")

            chunks = self.chunker.chunk_documents(documents)
            self.vector_store.add_chunks(chunks)
            self.uploaded_files.append(file_path.name)

            try:
                if dest_path.suffix.lower() == '.csv':
                    df = None
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(dest_path, encoding=encoding)
                            break
                        except: continue
                else:
                    df = pd.read_excel(dest_path)
                
                df = self._clean_and_normalize_df(df)
                self.current_file_data = df
                preview_html = f"""
                <div style="max-height: 400px; overflow-y: auto; overflow-x: auto;">
                    {df.head(10).to_html(index=False, classes="preview-table")}
                </div>
                """
            except Exception as e:
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
        import plotly.express as px
        import numpy as np
        try:
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        cleaned = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned, errors='ignore')
                    except: pass
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                x_col, y_col = categorical_cols[0], numeric_cols[0]
                if len(df[x_col].unique()) > 50: df = df.head(50)
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                fig.update_layout(template="plotly_white", showlegend=False)
                return fig
            elif len(numeric_cols) >= 2:
                fig = px.scatter(df.head(1000), x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                fig.update_layout(template="plotly_white")
                return fig
            return None
        except Exception as e:
            logger.error(f"Error in chart creation: {e}")
            return None
    
    def process_query(self, query: str, chat_history: List) -> Tuple[List, Optional[gr.Plot]]:
        """Process user query and hide technical details from chat history"""
        if not query.strip(): return chat_history, None
        if not settings.openai_api_key:
            error_msg = "⚠️ **OpenAI API key not configured!**"
            chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, None
        
        if self.vector_store.get_stats().get("status") == "empty":
            chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": "⚠️ Please upload an Excel file first!"})
            return chat_history, None
        
        chart_fig, computed_data, analysis = None, None, ""
        
        try:
            if self.rag_chain is None:
                self.rag_chain = RAGChain(retriever=self.retriever, model=settings.llm_model, api_key=settings.openai_api_key)
            
            schema_desc = "Schema not available"
            if self.current_file_data is not None:
                schema_desc = "Available Columns:\n" + "\n".join([f"- {col}" for col in self.current_file_data.columns])
                schema_desc += "\nSample Data:\n" + self.current_file_data.head(5).to_string(index=False)

            result = self.rag_chain.query(user_query=query, schema=schema_desc)
            if not result.get("success"):
                chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": f"❌ Error: {result.get('error')}"})
                return chat_history, None
            
            analysis = result["response"].get("analysis", "")
            code = result["response"].get("code")
            
            if code and self.current_file_data is not None:
                from ..execution.code_executor import CodeExecutor
                executor = CodeExecutor(timeout_seconds=10)
                execution_result = executor.execute(code, self.current_file_data)
                
                if execution_result["success"]:
                    chart_fig = execution_result.get("fig")
                    computed_data = execution_result.get("result_data")
                    
                    # Generate insights using the STATED PROMPT
                    insights_text = self._generate_insights(query=query, analysis=analysis, result_data=computed_data, context=result.get("raw_response", ""))
                    
                    response_parts = ["# 📊 Analysis Results\n"]
                    if insights_text: 
                        response_parts.append(f"{insights_text}\n")
                    else:
                        response_parts.append(f"**Analysis complete.** Based on the data, I have processed your request.")
                    
                    if chart_fig: 
                        response_parts.append("\n## 📈 Visualization\nSee the chart below.")
                    
                    response_text = "\n".join(response_parts)
                else:
                    response_text = f"⚠️ Analysis failed to execute. Please try rephrasing your question."
            else:
                response_text = f"# 📊 Analysis\n{analysis}"

            if chart_fig is None and isinstance(computed_data, pd.DataFrame) and len(computed_data) > 0:
                chart_fig = self._create_simple_chart_from_data(computed_data, query)
                if chart_fig and "📈 Visualization" not in response_text: 
                    response_text += "\n\n## 📈 Visualization\nSee the chart below."
            
            chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": response_text})
            if chart_fig is not None: 
                chart_fig.update_layout(autosize=True, margin=dict(l=50, r=50, t=50, b=50))
            return chat_history, chart_fig
        except Exception as e:
            chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            return chat_history, None
    
    def _generate_insights(
        self,
        query: str,
        analysis: str,
        result_data: Any,
        context: str
    ) -> str:
        """
        STRICTLY USE THE PROMPT PROVIDED: Generate narrative insights based on computed results
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
            
            # Create insights prompt STRICTLY AS REQUESTED
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
        STRICTLY USE THE LOGIC PROVIDED: Clear all uploaded data
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
        return None
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with Cosmic Nebula UI"""
        
        with gr.Blocks(title="DataCrux - Data Intelligence") as interface:
            gr.HTML("""
                <style>
                /* Cosmic Nebula Application Background */
                .gradio-container {
                    background: 
                        radial-gradient(circle at 15% 25%, rgba(79, 70, 229, 0.25) 0%, transparent 45%),
                        radial-gradient(circle at 85% 75%, rgba(45, 212, 191, 0.2) 0%, transparent 50%),
                        linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%) !important;
                    background-attachment: fixed !important;
                    position: relative;
                }
                .gradio-container::before {
                    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                    background-image: url('https://www.transparenttextures.com/patterns/stardust.png');
                    opacity: 0.5; pointer-events: none; z-index: 0;
                }
                h1, h2, h3, p, span { font-family: 'Inter', sans-serif; }
                .preview-container { background: rgba(255, 255, 255, 0.95); padding: 10px; border-radius: 8px; }
                .gradio-container > * { position: relative; z-index: 1; }
                
                /* Cosmic Nebula Header Style */
                .header-box {
                    text-align: center; 
                    background: 
                        radial-gradient(circle at 30% 20%, rgba(79, 70, 229, 0.4) 0%, transparent 40%),
                        radial-gradient(circle at 70% 80%, rgba(45, 212, 191, 0.3) 0%, transparent 50%),
                        linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #020617 100%);
                    background-image: 
                        url('https://www.transparenttextures.com/patterns/stardust.png'),
                        radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.15) 0%, transparent 80%),
                        linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #020617 100%);
                    padding: 3rem 2rem; 
                    border-radius: 24px; 
                    margin-bottom: 2rem; 
                    box-shadow: 0 20px 60px rgba(0,0,0,0.6), inset 0 0 20px rgba(56, 189, 248, 0.1); 
                    backdrop-filter: blur(20px); 
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    position: relative;
                    overflow: hidden;
                }
                </style>
            """)
            
            # Header
            gr.HTML("""
                <div class="header-box">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 1rem;">
                        <span style="font-size: 3rem; color: #ba8dff; filter: drop-shadow(0 0 15px rgba(186, 141, 255, 0.8));">✦</span>
                        <h1 style="color: #ffffff; font-size: 3.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; text-shadow: 0px 4px 20px rgba(0,0,0,0.5), 0 0 30px rgba(255,255,255,0.1);">DataCrux</h1>
                        <span style="font-size: 3rem; color: #8dbeff; filter: drop-shadow(0 0 15px rgba(141, 190, 255, 0.8));">✦</span>
                    </div>
                    <p style="color: #cbd5e1; font-size: 1.25rem; font-style: italic; margin: 0; font-weight: 300; letter-spacing: 0.05em; text-shadow: 2px 2px 8px rgba(0,0,0,0.4);">The Essential Core of Your Data's Intelligence.</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #fef9c3; margin-bottom: 10px;'>📁 Data Upload</h3>")
                    file_input = gr.File(label="Upload Excel File (.xlsx, .xls, .csv)", file_types=[".xlsx", ".xls", ".csv"])
                    upload_btn = gr.Button("📤 Process File", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear All Data", variant="stop")
                    status_output = gr.Markdown(label="Status")
                    
                    gr.HTML("<h3 style='color: #fef9c3; margin-top: 20px; margin-bottom: 10px;'>📋 Data Preview</h3>")
                    preview_output = gr.HTML(label="Preview", elem_classes="preview-container")
                
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='color: #fef9c3; margin-bottom: 10px;'>💬 Ask Questions About Your Data</h3>")
                    chatbot = gr.Chatbot(label="DataCrux Assistant", height=400)
                    with gr.Row():
                        query_input = gr.Textbox(label="Your Question", placeholder="Ask me anything about your data...", scale=4)
                        submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)
                    
                    gr.HTML("<h3 style='color: #fef9c3; margin-top: 20px; margin-bottom: 10px;'>📈 Visualization</h3>")
                    with gr.Row(): chart_output = gr.Plot(label="Visualization")
                    with gr.Row(): clear_chart_btn = gr.Button("🗑️ Clear Chart", size="sm", variant="secondary")
            
            # Footer
            gr.HTML("""
                <div style="text-align: center; padding: 1.5rem; background: rgba(42, 82, 122, 0.6); border-radius: 15px; margin-top: 1rem; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1);">
                    <p style="margin: 0; font-size: 0.95rem; color: #fef9c3; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">Powered by RAG + LangChain + OpenAI + HuggingFace + Gradio</p>
                </div>
            """)
            
            # Event handlers
            file_input.upload(fn=self.upload_file, inputs=[file_input], outputs=[status_output, preview_output])
            upload_btn.click(fn=self.upload_file, inputs=[file_input], outputs=[status_output, preview_output])
            submit_btn.click(fn=self.process_query, inputs=[query_input, chatbot], outputs=[chatbot, chart_output])
            query_input.submit(fn=self.process_query, inputs=[query_input, chatbot], outputs=[chatbot, chart_output])
            clear_btn.click(fn=self.clear_data, inputs=[], outputs=[status_output, preview_output, chatbot])
            clear_chart_btn.click(fn=self.clear_chart, inputs=[], outputs=[chart_output])
        
        return interface
    
    def launch(self):
        """Launch the Gradio app"""
        interface = self.create_interface()
        interface.queue()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            allowed_paths=[str(settings.data_dir.resolve())] if settings.data_dir.exists() else []
        )