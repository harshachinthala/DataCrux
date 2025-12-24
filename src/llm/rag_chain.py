"""
RAG Chain for LLM-based question answering
"""

from typing import Dict, Any, Optional
import logging
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage

from ..retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGChain:
    """LangChain-based RAG pipeline for Excel data Q&A"""
    
    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG chain
        
        Args:
            retriever: Retriever instance
            model: OpenAI model name
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key
        """
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        llm_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if api_key:
            llm_kwargs["openai_api_key"] = api_key
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        logger.info(f"RAG Chain initialized with model: {model}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for code generation"""
        
        from .prompts import CODE_GENERATION_PROMPT
        
        system_message = SystemMessagePromptTemplate.from_template(
            CODE_GENERATION_PROMPT + """

DATAFRAME SCHEMA:
{schema}

CONTEXT FROM VECTOR DB:
{context}

Generate Python code to answer the user's question using the DataFrame `df`.
"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template("{query}")
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        schema: str = "Schema not available"
    ) -> Dict[str, Any]:
        """
        Process a user query and return code to execute
        
        Args:
            user_query: User's question
            k: Number of chunks to retrieve
            filters: Metadata filters for retrieval
            schema: String description of data schema
            
        Returns:
            Dictionary with response and code to execute
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Retrieve relevant context
        context = self.retriever.retrieve_and_format(
            query=user_query,
            k=k,
            filters=filters,
            include_metadata=True
        )
        
        # Get retrieval stats
        results = self.retriever.retrieve(query=user_query, k=k, filters=filters)
        retrieval_stats = self.retriever.get_retrieval_stats(results)
        
        # Get LLM response
        try:
            # Create the chain: prompt | llm
            chain = self.prompt | self.llm
            
            # Invoke with input dictionary
            response = chain.invoke({
                "query": user_query,
                "context": context,
                "schema": schema
            })
            
            response_text = response.content
            
            # Parse response
            parsed_response = self._parse_code_response(response_text)
            
            return {
                "success": True,
                "query": user_query,
                "response": parsed_response,
                "raw_response": response_text,
                "retrieval_stats": retrieval_stats,
                "model": self.model
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "retrieval_stats": retrieval_stats
            }
    
    def _parse_code_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse code and analysis from LLM response
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed response dictionary with 'analysis' and 'code'
        """
        parsed = {
            "analysis": "",
            "code": None
        }
        
        # Extract ANALYSIS section
        analysis_match = re.search(r"##\s*ANALYSIS\s*\n(.*?)(?=##|```|\Z)", response_text, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            parsed["analysis"] = analysis_match.group(1).strip()
        
        # Extract CODE section (from code block)
        code_match = re.search(r"```python\s*\n(.*?)\n```", response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            parsed["code"] = code
            logger.info(f"Extracted code ({len(code)} chars)")
        else:
            logger.warning("No code block found in response")
        
        return parsed
    
    def _parse_chart_spec(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse chart specification from response
        
        Args:
            content: Chart suggestion section content
            
        Returns:
            Parsed chart spec or None
        """
        # Check if no chart is suggested
        if "none" in content.lower() and len(content) < 50:
            return None
        
        # Try to extract JSON from markdown code block
        # Matches ```json ... ``` or just ``` ... ```
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly (starting with { and ending with })
            # This regex looks for the outermost braces
            json_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                logger.warning("Could not extract chart specification JSON")
                return None
        
        # Clean up JSON string
        json_str = json_str.strip()
        
        # Parse JSON
        try:
            chart_spec = json.loads(json_str)
            logger.info(f"Parsed chart specification: {chart_spec.get('type', 'unknown')}")
            return chart_spec
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing chart JSON: {e}")
            return None

    def format_response_for_display(self, result: Dict[str, Any]) -> str:
        """
        Format response for user-friendly display
        
        Args:
            result: Query result dictionary
            
        Returns:
            Formatted string for display
        """
        if not result.get("success"):
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        response = result["response"]
        
        parts = [
            "# 📊 Analysis Results\n",
            response["answer"]
        ]
        
        if response.get("chart_suggestion"):
            chart_type = response["chart_suggestion"].get("type", "unknown")
            chart_title = response["chart_suggestion"].get("title", "Untitled")
            parts.append(f"\n## 📈 Visualization")
            parts.append(f"**Chart Type**: {chart_type.capitalize()}")
            parts.append(f"**Title**: {chart_title}")
        
        # Add retrieval stats
        stats = result.get("retrieval_stats", {})
        if stats.get("count", 0) > 0:
            parts.append(f"\n---\n*Retrieved {stats['count']} relevant chunks from {stats.get('unique_sources', 0)} file(s)*")
        
        return "\n\n".join(parts)
