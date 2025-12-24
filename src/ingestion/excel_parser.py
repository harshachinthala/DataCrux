"""
Excel file parser for extracting data from various Excel formats
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SheetData:
    """Container for parsed sheet data"""
    name: str
    data: pd.DataFrame
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]


@dataclass
class ExcelFileData:
    """Container for parsed Excel file data"""
    filename: str
    sheets: Dict[str, SheetData]
    sheet_names: List[str]
    total_rows: int
    total_columns: int


class ExcelParser:
    """Parse Excel files (.xls, .xlsx, .csv) and extract structured data"""
    
    def __init__(self):
        self.supported_extensions = {'.xlsx', '.xls', '.csv'}
    
    def parse_file(self, file_path: Union[str, Path]) -> ExcelFileData:
        """
        Parse an Excel file and return structured data
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            ExcelFileData object containing all sheets and metadata
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {self.supported_extensions}"
            )
        
        logger.info(f"Parsing file: {file_path.name}")
        
        # Handle CSV files
        if file_path.suffix.lower() == '.csv':
            return self._parse_csv(file_path)
        
        # Handle Excel files
        return self._parse_excel(file_path)
    
    def _parse_csv(self, file_path: Path) -> ExcelFileData:
        """Parse CSV file"""
        try:
            # Try UTF-8 first
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 (ISO-8859-1) which accepts all byte values
                logger.warning(f"UTF-8 decoding failed, trying latin-1 encoding for {file_path.name}")
                df = pd.read_csv(file_path, encoding='latin-1')
            
            sheet_data = self._create_sheet_data("Sheet1", df)
            
            return ExcelFileData(
                filename=file_path.name,
                sheets={"Sheet1": sheet_data},
                sheet_names=["Sheet1"],
                total_rows=len(df),
                total_columns=len(df.columns)
            )
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise
    
    def _parse_excel(self, file_path: Path) -> ExcelFileData:
        """Parse Excel file (multi-sheet support)"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            total_rows = 0
            total_cols = 0
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheet_data = self._create_sheet_data(sheet_name, df)
                sheets[sheet_name] = sheet_data
                total_rows += len(df)
                total_cols = max(total_cols, len(df.columns))
            
            return ExcelFileData(
                filename=file_path.name,
                sheets=sheets,
                sheet_names=excel_file.sheet_names,
                total_rows=total_rows,
                total_columns=total_cols
            )
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            raise
    
    def _create_sheet_data(self, sheet_name: str, df: pd.DataFrame) -> SheetData:
        """Create SheetData object from DataFrame"""
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Get data types
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return SheetData(
            name=sheet_name,
            data=df,
            row_count=len(df),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
            dtypes=dtypes
        )
    
    def get_preview(
        self,
        excel_data: ExcelFileData,
        sheet_name: Optional[str] = None,
        n_rows: int = 5
    ) -> pd.DataFrame:
        """
        Get preview of data from a specific sheet
        
        Args:
            excel_data: Parsed Excel file data
            sheet_name: Name of sheet to preview (default: first sheet)
            n_rows: Number of rows to preview
            
        Returns:
            DataFrame with preview data
        """
        if sheet_name is None:
            sheet_name = excel_data.sheet_names[0]
        
        if sheet_name not in excel_data.sheets:
            raise ValueError(f"Sheet '{sheet_name}' not found")
        
        return excel_data.sheets[sheet_name].data.head(n_rows)
    
    def get_summary(self, excel_data: ExcelFileData) -> str:
        """
        Generate a text summary of the Excel file
        
        Args:
            excel_data: Parsed Excel file data
            
        Returns:
            Formatted summary string
        """
        summary_parts = [
            f"File: {excel_data.filename}",
            f"Total Sheets: {len(excel_data.sheets)}",
            f"Total Rows: {excel_data.total_rows}",
            f"Total Columns: {excel_data.total_columns}",
            "\nSheet Details:"
        ]
        
        for sheet_name, sheet_data in excel_data.sheets.items():
            summary_parts.append(
                f"  - {sheet_name}: {sheet_data.row_count} rows, "
                f"{sheet_data.column_count} columns"
            )
            summary_parts.append(f"    Columns: {', '.join(sheet_data.columns)}")
        
        return "\n".join(summary_parts)
