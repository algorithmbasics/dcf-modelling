# file: dfcol2numpy.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌
#                info: ℹ️, debug: 🐞, processing: 🔄

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

def list2numpy(
    list_seq: list,
    logger: Optional[logging.Logger] = None
    ) -> Optional[np.ndarray]:
    # did user enter a list?
    if not isinstance(list_seq, list) or len(list_seq) == 0:
        if logger is not None:
            logger.error(f"❌ User did not input a valid list: Does not exist or Empty List.")
        return None

    # is the list  - a list of numbers (int or float)? NEW CONCEPT
    for item in list_seq:
        if not isinstance(item, (int, float)):
            if logger is not None:
                logger.error(f"❌ User did not input a valid list: Not Integers or Float.")
            return None

    return np.array(list_seq, dtype=float)

# sales_as_list: List[int] = [42905, 65225, 108249, 156508, 170910, 182795, 233715, 215091, 228594, 265595, 260174, 274515, 365817, 394328, 383285, 391035, 416161]
# logger.info(sales_as_list)

# sales_as_array: np.ndarray = list2numpy(list_seq=sales_as_list)
# logger.info(sales_as_array)


# fetch dataframe column as numpy array
def dfcol2numpy(varname: str, df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    
    if logger is not None:
        logger.info(f"🔄 Converting '{varname}' column to numpy array.")
    
    if df is None:
        if logger is not None:
            logger.error("❌ No DataFrame provided.")
        return None
    
    if not isinstance(df, pd.DataFrame):
        if logger is not None:
            logger.error("❌ Provided data is not a valid DataFrame.")
        return None

    if not isinstance(varname, str) or varname.strip() == "":
        if logger is not None:
            logger.error("❌ Variable name must be a non-empty string.")
        return None
    
    if df.empty:
        if logger is not None:
            logger.error("❌ DataFrame is empty.")
        return None
    
    if varname not in df.columns:
        if logger is not None:
            logger.error(f"❌ Variable '{varname}' not found in DataFrame columns.")
        return None

    list_seq: List[float] = df[varname].tolist()
    numpy_array: Optional[np.ndarray] = list2numpy(list_seq)

    if numpy_array is None:
        if logger is not None:
            logger.error(f"❌ Conversion of variable '{varname}' to numpy array failed.")
        return None

    if logger is not None:
        logger.info(f"✅ Successfully converted '{varname}' column to numpy array with shape {numpy_array.shape}.")

    return numpy_array