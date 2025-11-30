# file: dfcol2numpy.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌
#                info: ℹ️, debug: 🐞, processing: 🔄

import logging
import numpy as np
import pandas as pd
from typing import Optional, Literal

METHODS_SUPPORTED = Literal["average", "last"]

def extend_array(
    arr: np.ndarray, # historical array
    forecast_periods: int, # number of forecast periods
    # method: how to fill the numbers -- either based on average or last value
    method: Optional[METHODS_SUPPORTED] = None,
    # once you start with the average or last value 
    # -- should it increase/decrease by a STEP value 
    STEP: Optional[float] = None,
    # what should be the starting value?
    START_VALUE: Optional[float] = None,
    # what should be the ending value?
    END_VALUE: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:
    
    if logger is not None:
        logger.info(f"🔄 Extending array '{arr}', with method='{method}', forecast_periods={forecast_periods}, STEP={STEP}" )

    # if input array is not valid, return None
    if not isinstance(arr, np.ndarray):
        if logger is not None:
            logger.error("❌ Input must be a numpy ndarray.")
        return None
    
    # if method is invalid, return None
    if method is not None and method not in ["average", "last"]:
        if logger is not None:
            logger.error(f"❌ Unsupported method '{method}'. Use 'average' or 'last'.")
        return None
    
    # if input array is not 1D, return None
    if arr.ndim != 1:
        if logger is not None:
            logger.error("❌ Input array must be 1D.")
        return None

    # if forecast_periods is negative, return None        
    if forecast_periods < 0:
        if logger is not None:
            logger.error("❌ forecast_periods must be non-negative.")
        return None
    
    # Case 1: Both start and end values provided
    if START_VALUE is not None and END_VALUE is not None:
        base_value = START_VALUE
        STEP = (END_VALUE - START_VALUE) / forecast_periods

    # Case 2: Otherwise, starting value and STEP
    else:
        STEP = 0.0 if STEP is None else float(STEP)
        base_value = (
            float(np.nanmean(arr)) if method == "average" else float(arr[-1])
        )

    # default steps array
    steps = np.array(range(1, forecast_periods + 1), dtype=float)

    # Prepare extension array
    extension = base_value + steps * STEP

    return np.concatenate((arr, extension))
