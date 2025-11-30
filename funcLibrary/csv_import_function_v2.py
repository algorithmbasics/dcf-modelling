# file: csv_import.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌

import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Union

# file = "https://docs.google.com/spreadsheets/d/1NYbkIPEGmbqMxDnQSsTNWnLns1JxgTdwZXsf51QgEQM/edit?usp=sharing"
# ----------------------------------------------| SHEET ID                                   |-----------------

def import_csv(
        path: Optional[Union[str, Path]] = None, 
        sheet_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> Optional[pd.DataFrame]:

    if path is None and sheet_id is None:
        if logger is not None:
            logger.error(f"❌ Path and Sheet ID are not specified by the user: {path}, {sheet_id}")
        return None
    
    if sheet_id is not None:
        # Construct Google Sheets CSV export URL
        path: str = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        if logger is not None:
            logger.info(f"⚙️ Constructed Google Sheets URL: {path}")
    
    # Check if it's a URL
    is_url = isinstance(path, str) and (path.startswith('http://') or path.startswith('https://'))
             
    if is_url:
        if logger is not None:
            logger.info(f"🌐 Detected URL: {path}")
        path_str = path
    else:
        # Handle local file path
        if isinstance(path, str):
            if logger is not None:
                logger.warning(f"⚙️ Converting string path '{path}' to Path object.")
            path = Path(path)

        if not isinstance(path, Path):
            if logger is not None:
                logger.error(f"❌ Invalid path object. User probably entered a number.")
            return None

        # Check if file exists (only for local paths)
        if path.exists():
            if logger is not None:
                logger.info(f"✅ File exists in {path}.")
        else:
            if logger is not None:
                logger.error(f"❌ File does not exist in {path}.")
            return None
        
        path_str = str(path)

    # Read CSV from either URL or local path
    df: pd.DataFrame = pd.read_csv(path_str)
    if logger is not None:
        logger.info(f"✅ Loaded data from {path_str}")
    
    # Check if dataset is empty
    if df.empty:
        if logger is not None:
            logger.error(f"❌ File is empty.")
        return None
    
    if logger is not None:
        logger.info("✅ Successful import - file exists and not empty.")

    return df