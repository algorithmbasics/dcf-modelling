# file: delta_numpy.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌
#                info: ℹ️, debug: 🐞, processing: 🔄

import logging
import numpy as np
from typing import Optional

def delta_numpy(
    array: np.ndarray, 
    first_value: Optional[float], 
    logger: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:

    """
    Create an array of first differences:
        delta[i] = array[i] - array[i - 1]
    with delta[0] = np.nan.
    """
    
    if logger is not None:
        logger.info("🔄 Computing first differences (delta array).")
    
    # Validate input
    if array is None:
        if logger is not None:
            logger.error("❌ No array provided.")
        return None

    if not isinstance(array, np.ndarray):
        if logger is not None:
            logger.error("❌ Input must be a numpy array.")
        return None
    
    if array.ndim != 1:
        if logger is not None:
            logger.error("❌ Input array must be 1-dimensional.")
        return None
    
    if array.size == 0:
        if logger is not None:
            logger.error("❌ Input array is empty.")
        return None
    
    # Ensure numeric
    if not np.issubdtype(array.dtype, np.number):
        if logger is not None:
            logger.error("❌ Array must contain numeric values.")
        return None
    
    # Allocate output
    delta = np.empty_like(array, dtype=float)
    delta[0] = np.nan if first_value is None else 0.0
    delta[1:] = array[1:] - array[:-1]
    
    if logger is not None:
        logger.info("✅ Delta array successfully computed.")
    
    return delta
