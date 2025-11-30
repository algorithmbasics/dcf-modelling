# file: array_arithmetic.py
import numpy as np
import logging
from typing import Optional

# constants
ADD_LIST = ["add", "addition", "sum", "plus", "+"]
SUB_LIST = ["subtract", "sub", "minus", "-"]
MULT_LIST = ["multiply", "times", "*"]
DIV_LIST = ["divide", "div", "/"]

OP_SET = set(ADD_LIST) | set(SUB_LIST) | set(MULT_LIST) | set(DIV_LIST)

def array_arithmetic(
        arr1: np.ndarray, 
        arr2: np.ndarray, 
        operation: str,
        logger: Optional[logging.Logger] = None
    ) -> Optional[np.ndarray]:

    # 1. validate operation
    if not operation or not isinstance(operation, str):
        if logger:
            logger.error("❌ Invalid operation specified: empty or not a string.")
        return None

    op = operation.lower()

    # 2. validate arrays
    if arr1 is None or arr2 is None:
        if logger:
            logger.error("❌ User did not input one of the arrays.")
        return None

    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        if logger:
            logger.error("❌ Invalid arrays input.")
        return None

    if arr1.shape != arr2.shape:
        if logger:
            logger.error("❌ Invalid arrays input: arrays must have identical shapes.")
        return None

    # 3. validate operation string
    if op not in OP_SET:
        if logger:
            logger.error("❌ Invalid operation input.")
        return None

    # Operations
    if op in ADD_LIST:
        if logger:
            logger.info(f"➕ Adding arrays with shape {arr1.shape}")
        return np.add(arr1, arr2)

    elif op in SUB_LIST:
        if logger:
            logger.info(f"➖ Subtracting arrays with shape {arr1.shape}")
        return np.subtract(arr1, arr2)

    elif op in MULT_LIST:
        if logger:
            logger.info(f"✖️ Multiplying arrays with shape {arr1.shape}")
        return np.multiply(arr1, arr2)

    elif op in DIV_LIST:
        if logger:
            logger.info(f"➗ Dividing arrays with shape {arr1.shape}")
        arr1f, arr2f = arr1.astype(float), arr2.astype(float)
        return np.divide(arr1f, arr2f, out=np.full_like(arr1f, np.nan), where=(arr2f != 0))

    return None
