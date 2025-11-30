# file: apply_growth.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌, 
#                add: ➕, subtract: ➖, multiply: ✖️, divide: ➗

import numpy as np
import logging
from typing import Optional

def apply_growth(
    initial_value: float, 
    growth_rates: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:

    # ---- Type validation ----
    if not isinstance(initial_value, (int, float)):
        if logger is not None:
            logger.error(f"❌ Invalid operation specifed: empty or not string.")
        return None

    if not isinstance(growth_rates, np.ndarray):
        if logger is not None:
            logger.error(
                f"❌ Invalid operation specifed: growth_rates must be a numpy.ndarray."
            )
        return None

    # ---- Value validation ----
    if growth_rates.size == 0:
        if logger is not None:
            logger.error("❌ growth_rates array cannot be empty.")
        return None

    # Compute cumulative growth multipliers
    multipliers = np.cumprod(1 + growth_rates)

    # Insert the initial value at the beginning and scale the multipliers
    return np.concatenate(([initial_value], initial_value * multipliers))
