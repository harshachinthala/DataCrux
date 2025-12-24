"""
Safe code executor for LLM-generated data analysis code
"""

import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Safely execute LLM-generated Python code for data analysis
    """
    
    def __init__(self, timeout_seconds: int = 10):
        """
        Initialize code executor
        
        Args:
            timeout_seconds: Maximum execution time (not enforced due to threading limitations)
        """
        self.timeout_seconds = timeout_seconds
        logger.info(f"CodeExecutor initialized")
    
    def execute(
        self,
        code: str,
        dataframe: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute code safely with the provided DataFrame
        
        Args:
            code: Python code to execute
            dataframe: DataFrame to analyze
            
        Returns:
            Dictionary with:
                - success: bool
                - result_data: DataFrame or Series (if successful)
                - fig: Plotly figure (if successful)
                - error: str (if failed)
        """
        logger.info(f"Executing code (length: {len(code)} chars)")
        
        try:
            # Create safe execution environment
            safe_globals = self._create_safe_globals(dataframe)
            
            # Execute code
            exec(code, safe_globals)
            
            # Extract results
            result_data = safe_globals.get('result_data')
            fig = safe_globals.get('fig')
            
            logger.info(f"Code executed successfully. Has result_data: {result_data is not None}, Has fig: {fig is not None}")
            
            return {
                "success": True,
                "result_data": result_data,
                "fig": fig,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Code execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result_data": None,
                "fig": None,
                "error": str(e)
            }
    
    def _create_safe_globals(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a safe globals dictionary for code execution
        
        Args:
            dataframe: User's DataFrame
            
        Returns:
            Dictionary of allowed globals
        """
        # Work on a copy to prevent modifications
        df_copy = dataframe.copy()
        
        # Define safe globals with only necessary libraries
        safe_globals = {
            # Data manipulation
            'pd': pd,
            'np': np,
            'df': df_copy,
            
            # Visualization
            'px': px,
            'go': go,
            
            # Results placeholders
            'result_data': None,
            'fig': None,
            
            # Disable dangerous built-ins
            '__builtins__': {
                # Allow only safe built-ins
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'print': print,  # For debugging
            }
        }
        
        return safe_globals
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for dangerous operations
        
        Args:
            code: Code to validate
            
        Returns:
            (is_valid, error_message)
        """
        # List of forbidden operations
        forbidden_keywords = [
            'import',  # No dynamic imports
            'open',    # No file operations
            '__',      # No dunder methods
            'eval',    # No eval
            'exec',    # No nested exec
            'compile', # No compile
            'globals', # No globals access
            'locals',  # No locals access
        ]
        
        code_lower = code.lower()
        
        for keyword in forbidden_keywords:
            if keyword in code_lower:
                return False, f"Forbidden keyword '{keyword}' found in code"
        
        return True, None
