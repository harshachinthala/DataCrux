"""
Chart generation from LLM specifications using Plotly
"""

from typing import Dict, Any, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate Plotly charts from JSON specifications"""
    
    def __init__(self):
        self.chart_type_map = {
            "bar": self._create_bar_chart,
            "line": self._create_line_chart,
            "scatter": self._create_scatter_chart,
            "pie": self._create_pie_chart,
            "histogram": self._create_histogram
        }
    
    def generate_chart(
        self,
        chart_spec: Dict[str, Any],
        data: Optional[pd.DataFrame] = None
    ) -> Optional[go.Figure]:
        """
        Generate a Plotly chart from specification
        
        Args:
            chart_spec: Chart specification dictionary
            data: Optional DataFrame with data (if not in spec)
            
        Returns:
            Plotly Figure object or None if generation fails
        """
        if not chart_spec:
            logger.warning("No chart specification provided")
            return None
        
        chart_type = chart_spec.get("type", "").lower()
        
        if chart_type not in self.chart_type_map:
            logger.error(f"Unknown chart type: {chart_type}")
            return None
        
        logger.info(f"Generating {chart_type} chart: {chart_spec.get('title', 'Untitled')}")
        
        try:
            # Always use the provided DataFrame if available
            if data is not None:
                df = data.copy()
            elif "data" in chart_spec and chart_spec["data"]:
                try:
                    df = pd.DataFrame(chart_spec["data"])
                except Exception as e:
                    logger.error(f"Failed to create DataFrame from chart_spec data: {e}")
                    return None
            else:
                logger.error("No data available for chart generation")
                return None
            
            # Normalize DataFrame columns
            df.columns = df.columns.astype(str).str.strip()
            
            # Clean data (convert currency/strings to numbers)
            df = self._clean_dataframe(df)
            
            # Try to generate the requested chart
            try:
                # Validate columns exist
                self._validate_columns(chart_spec, df)
                
                # Generate chart using appropriate method
                chart_func = self.chart_type_map.get(chart_type)
                if chart_func:
                    fig = chart_func(chart_spec, df)
                    if fig:
                        return fig
                else:
                    logger.warning(f"Unsupported chart type: {chart_type}")
            
            except Exception as e:
                logger.warning(f"Failed to generate {chart_type} chart: {e}")
            
            # FALLBACK: Create a simple chart with available data
            logger.info("Attempting fallback chart generation...")
            return self._create_fallback_chart(df, chart_spec)
        
        except Exception as e:
            logger.error(f"Error generating chart: {e}", exc_info=True)
            return None
    
    def _create_fallback_chart(self, df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a simple fallback chart when the requested chart fails
        
        Args:
            df: DataFrame
            spec: Original chart spec (for title)
            
        Returns:
            Simple bar or scatter chart
        """
        try:
            # Get variable columns (exclude index-like)
            exclude_patterns = ["index", "level_", "unnamed", "id"]
            
            # Helper to check if column should be excluded
            def is_excluded(col_name):
                col_lower = str(col_name).lower()
                return any(p in col_lower for p in exclude_patterns)
            
            # Get numeric columns
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns 
                if not is_excluded(c)
            ]
            
            # Get categorical columns
            categorical_cols = [
                c for c in df.select_dtypes(include=['object', 'category', 'string']).columns 
                if not is_excluded(c)
            ]
            
            # If no "clean" numeric columns found, try all numeric columns as last resort
            if not numeric_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logger.error("No numeric columns available for fallback chart")
                return None
            
            # Strategy 1: Categorical vs Numeric (Bar Chart)
            # This is the best default for "Show average earnings by city" type queries
            if categorical_cols and numeric_cols:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                logger.info(f"Fallback: Creating bar chart with x={x_col}, y={y_col}")
                
                fig = px.bar(
                    df.head(20),
                    x=x_col,
                    y=y_col,
                    title=spec.get("title", f"{y_col} by {x_col}")
                )
                return fig
            
            # Strategy 2: Numeric vs Numeric (Scatter)
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                logger.info(f"Fallback: Creating scatter plot with x={x_col}, y={y_col}")
                
                fig = px.scatter(
                    df.head(100),
                    x=x_col,
                    y=y_col,
                    title=spec.get("title", f"{y_col} vs {x_col}")
                )
                return fig
            
            # Strategy 3: Single Numeric (Histogram/Bar)
            y_col = numeric_cols[0]
            logger.info(f"Fallback: Creating distribution chart for {y_col}")
            fig = px.bar(
                df.head(20),
                y=y_col,
                title=spec.get("title", f"Distribution of {y_col}")
            )
            return fig
            
        except Exception as e:
            logger.error(f"Fallback chart generation failed: {e}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by converting string numbers to actual numbers
        """
        df = df.copy()
        
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Try to convert object/string columns to numeric
            try:
                # Remove common currency symbols and separators
                if df[col].dtype == 'object':
                    # Create a temporary series for cleanup
                    clean_series = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                    # Attempt conversion to numeric
                    df[col] = pd.to_numeric(clean_series, errors='ignore')
            except Exception as e:
                # If conversion fails, keep as is
                logger.debug(f"Could not convert column {col} to numeric: {e}")
                pass
                
        return df

    def _validate_columns(self, spec: Dict[str, Any], df: pd.DataFrame):
        """
        Validate and fix column names in spec by matching with DataFrame columns
        """
        df_cols = set(df.columns)
        df_cols_lower = {col.lower(): col for col in df_cols}
        
        for key in ["x_column", "y_column", "color_by", "labels_column", "values_column"]:
            if key not in spec or not spec[key]:
                continue
                
            col_name = str(spec[key]).strip()
            
            # Direct match
            if col_name in df_cols:
                continue
            
            # Case-insensitive match
            if col_name.lower() in df_cols_lower:
                corrected = df_cols_lower[col_name.lower()]
                logger.info(f"Corrected column '{col_name}' to '{corrected}'")
                spec[key] = corrected
                continue
            
            # Intelligent mapping for common patterns
            col_lower = col_name.lower()
            
            # Handle "Age Group" -> "Age"
            if "age" in col_lower and "group" in col_lower:
                if "age" in df_cols_lower:
                    spec[key] = df_cols_lower["age"]
                    logger.info(f"Mapped '{col_name}' to 'Age'")
                    continue
            
            # Handle "Average Income" -> "Income" with aggregation
            if "average" in col_lower or "avg" in col_lower:
                # Extract the actual column name
                for df_col in df_cols:
                    if df_col.lower() in col_lower:
                        spec[key] = df_col
                        # Set aggregation if not already set
                        if "aggregation" not in spec or spec["aggregation"] == "none":
                            spec["aggregation"] = "avg"
                        logger.info(f"Mapped '{col_name}' to '{df_col}' with avg aggregation")
                        break
                if spec[key] != col_name:  # If we found a match
                    continue
            
            # Try to find any column that contains the key word
            col_words = col_lower.split()
            for df_col in df_cols:
                df_col_lower = df_col.lower()
                # Check if any significant word from the requested column is in the actual column
                for word in col_words:
                    if len(word) > 3 and word in df_col_lower:  # Only match words longer than 3 chars
                        spec[key] = df_col
                        logger.info(f"Fuzzy matched '{col_name}' to '{df_col}'")
                        break
                if spec[key] != col_name:  # If we found a match
                    break
            
            if spec[key] == col_name:  # Still no match found
                logger.warning(f"Column '{col_name}' not found in DataFrame columns: {df_cols}")
    
    def _create_bar_chart(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> go.Figure:
        """Create bar chart"""
        x_col = spec.get("x_column")
        y_col = spec.get("y_column")
        title = spec.get("title", "Bar Chart")
        orientation = spec.get("orientation", "v")
        color_by = spec.get("color_by")
        aggregation = spec.get("aggregation", "none")
        
        # Apply aggregation if specified
        if aggregation and aggregation != "none":
            df = self._apply_aggregation(df, x_col, y_col, aggregation)
        
        # Create chart
        if orientation == "h":
            fig = px.bar(
                df,
                x=y_col,
                y=x_col,
                title=title,
                color=color_by,
                orientation="h"
            )
        else:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color=color_by
            )
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        return fig
    
    def _create_line_chart(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> go.Figure:
        """Create line chart"""
        x_col = spec.get("x_column")
        y_col = spec.get("y_column")
        title = spec.get("title", "Line Chart")
        color_by = spec.get("color_by")
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color=color_by,
            markers=True
        )
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        return fig
    
    def _create_scatter_chart(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> go.Figure:
        """Create scatter plot"""
        x_col = spec.get("x_column")
        y_col = spec.get("y_column")
        title = spec.get("title", "Scatter Plot")
        color_by = spec.get("color_by")
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color=color_by
        )
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        return fig
    
    def _create_pie_chart(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> go.Figure:
        """Create pie chart"""
        labels_col = spec.get("labels_column") or spec.get("x_column")
        values_col = spec.get("values_column") or spec.get("y_column")
        title = spec.get("title", "Pie Chart")
        
        fig = px.pie(
            df,
            names=labels_col,
            values=values_col,
            title=title
        )
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        return fig
    
    def _create_histogram(
        self,
        spec: Dict[str, Any],
        df: pd.DataFrame
    ) -> go.Figure:
        """Create histogram"""
        x_col = spec.get("x_column")
        title = spec.get("title", "Histogram")
        color_by = spec.get("color_by")
        
        fig = px.histogram(
            df,
            x=x_col,
            title=title,
            color=color_by
        )
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        return fig
    
    def _apply_aggregation(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        agg_func: str
    ) -> pd.DataFrame:
        """Apply aggregation function"""
        agg_map = {
            "sum": "sum",
            "avg": "mean",
            "count": "count",
            "min": "min",
            "max": "max"
        }
        
        if agg_func not in agg_map:
            return df
        
        pandas_func = agg_map[agg_func]
        
        result = df.groupby(group_col)[value_col].agg(pandas_func).reset_index()
        return result
