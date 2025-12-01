# file: dcf.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌
#                info: ℹ️, debug: 🐞, processing: 🔄

# python libraries
import sys
import numpy as np
import pandas as pd
import logging
from typing import Optional

# user libraries
from funcLibrary.csv_import_function_v2 import import_csv
from funcLibrary.dfcol2numpy import dfcol2numpy
from funcLibrary.array_arithmetic_v2 import array_arithmetic
from funcLibrary.extend_array_function import extend_array
from funcLibrary.apply_growth import apply_growth
from funcLibrary.delta_numpy import delta_numpy
from funcLibrary.yahoo_finance_price_fetcher_v2 import yahoo_finance_price_fetcher_v2


# print options
np.set_printoptions(linewidth=200)


# Set up configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATEFMT: str = "%Y-%m-%d %H:%M:%S"
SHEET_ID: str = "18FqhJ8iB6AZwbjUjGWFiWPBiPebjigGIwAvmmQDd1FA"
CURRENT_YEAR: int = 2025
N_HISTORICAL_YEARS: int = 5
N_FORECAST_YEARS: int = 5
HISTORY_START_YEAR: int = CURRENT_YEAR - N_HISTORICAL_YEARS + 1
HISTORY_END_YEAR: int = CURRENT_YEAR
TERMINAL_GROWTH_RATE: float = 0.03
TERMINAL_COGS_MARGIN_RATE: float = 0.60
TERMINAL_SGA_MARGIN_RATE: float = 0.30
TERMINAL_RD_MARGIN_RATE: float = 0.10
TERMINAL_DEPR_TO_CAPEX_RATE: float = 1.0
WACC: float = 0.1016
TICKER_SYMBOL: str = "AIT"
BUY_SELL_THRESHOLD: float = 0.08 # + or - 8%


# variable settings
DATES_VAR: str = "DATE"
SALES_VAR: str = "SALES_REV_TURN"
COGS_VAR: str = "IS_COG_AND_SERVICES_SOLD"
GROSS_PROFIT_VAR: str = "GROSS_PROFIT"
OTHER_OPER_INC_VAR: str = "IS_OTHER_OPER_INC"
OPER_EXP_VAR: str = "IS_OPERATING_EXPN"
SGA_EXP_VAR: str = "IS_SGA_EXPENSE"
RD_EXP_VAR: str = "IS_OPERATING_EXPENSES_RD"
OTHER_OPER_EXP_VAR: str = "IS_OTHER_OPERATING_EXPENSES"
OPER_INC_VAR: str = "IS_OPER_INC"
PRETAX_INC: str = "PRETAX_INC"
TAX_EXP_VAR: str = "IS_INC_TAX_EXP"
EFFECTIVE_TAX_RATE_VAR: str = "EFF_TAX_RATE"
CAPEX_VAR: str = "CAPITAL_EXPEND"
CAPEX_TO_SALES_VAR: str = "CAP_EXPEND_TO_SALES"
EBITDA_VAR: str = "EBITDA"
EBITA_VAR: str = "EBITA"
DEPRECIATION_EXPENSE_VAR: str = "IS_DEPR_EXP"
TOTAL_CURRENT_ASSETS_VAR: str = "BS_CUR_ASSET_REPORT"
TOTAL_CURRENT_LIABILITIES_VAR: str = "BS_CUR_LIAB"
SHORT_TERM_DEBT: str = "BS_ST_BORROW"
SHORT_TERM_PORTION_OF_LONG_TERM_DEBT: str = "BS_CURR_PORTION_LT_DEBT"
LONG_TERM_DEBT: str = "BS_LT_BORROW"
PREFERRED_EQUITY: str = "PFD_EQTY_HYBRID_CAPITAL"
MINORITY_INTEREST: str = "MINORITY_NONCONTROLLING_INTEREST"
CASH_AND_EQUIVALENTS: str = "CASH_CASH_EQTY_STI_DETAILED"
N_SHARES_OUTSTANDING: str = "BS_SH_OUT"


# logging config
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT,
    datefmt=LOGGING_DATEFMT
)
logger = logging.getLogger(__name__)
LOGGING_OUTPUT = None # None or logger

#----------------------------------------------------------------------------------------------------------
# STEP 1: IMPORT DATA FROM GOOGLE SHEET
# STEP 2: Keep only data within historical window
#         window: HISTORY_START_YEAR, HISTORY_END_YEAR
# STEP 3: convert relevant columns to numpy arrays: dfcol2numpy()
# STEP 4: check constructed variables
# STEP 5: compute margin arrays using array_arithmetic()
# STEP 6: projections using extend_array()
# STEP 7: fill up forecasted values using apply_growth()
# STEP 8: construct FCF
# STEP 9: Present value calculations
# STEP 10: Enterprise value
# STEP 11: Equity value
# STEP 12: Decision
#----------------------------------------------------------------------------------------------------------

# STEP 1: IMPORT DATA FROM GOOGLE SHEET

# Load DCF data
dcf_df: Optional[pd.DataFrame] = import_csv(path=None, sheet_id=SHEET_ID, logger=LOGGING_OUTPUT)

# Check if data was loaded successfully
if dcf_df is None:
    logger.error("❌ Failed to load DCF data. Stopping execution.")
    sys.exit()

logger.info(f"⚙️ DCF DataFrame columns: {dcf_df.columns.tolist()}")
logger.info(f"⚙️ DCF DataFrame preview:\n{dcf_df}")

# sys.exit()  # STOP <1>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING



# STEP 2: Keep only data within historical window
#         window: HISTORY_START_YEAR, HISTORY_END_YEAR

# process dates: is data variable in the input dataset
if DATES_VAR not in dcf_df.columns:
    logger.error(f"❌ Variable '{DATES_VAR}' not found in DataFrame columns. Stopping execution.")
    sys.exit()

# convert date strings to datetime objects
dcf_df.loc[:, "formatted_date"] = pd.to_datetime(dcf_df[DATES_VAR], format="%m/%d/%Y", errors='coerce')
dcf_df.drop(columns=[DATES_VAR], inplace=True)
logger.info(f"ℹ️ Formatted date column added: {dcf_df.shape}")

# extract year from date
dcf_df.loc[:, "year"] = dcf_df["formatted_date"].dt.year
logger.info(f"ℹ️ Year column added: {dcf_df.shape}")

# sort dataframe by date
dcf_df = dcf_df.sort_values(by=["year"], ascending=[True]).reset_index(drop=True)

# actuals dataset
actuals_condition: bool = (dcf_df["year"] >= HISTORY_START_YEAR) & (dcf_df["year"] <= HISTORY_END_YEAR)
dcf_actuals: pd.DataFrame = dcf_df[actuals_condition].reset_index(drop=True)
logger.info(f"ℹ️ DataFrame reduced between years '{HISTORY_START_YEAR}' and '{HISTORY_END_YEAR}': {dcf_actuals.shape}")

# sales growth column in dataframe
# using pct_change()
dcf_actuals.loc[:, "sales_growth"] = dcf_actuals[SALES_VAR].pct_change(periods=1)
logger.info(f"ℹ️ Sales growth column added: {dcf_actuals.shape}")

# Log success message
logger.info("✅ DCF data loaded.")
logger.info(f"ℹ️ DCF DataFrame preview:\n{dcf_actuals[["formatted_date", "year", SALES_VAR, "sales_growth"]]}")

# sys.exit()  # STOP <2>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING




# STEP 3: convert relevant columns to numpy arrays: dfcol2numpy()

# sales array
sales_arr: Optional[np.ndarray] = dfcol2numpy(varname=SALES_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)
if sales_arr is None:
    logger.error(f"❌ Failed to convert '{sales_arr}' column to numpy array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Sales numpy array: {sales_arr}")

# sys.exit()  # STOP <3>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# sales growth array
sales_growth_arr: Optional[np.ndarray] = dfcol2numpy(varname="sales_growth", df=dcf_actuals, logger=LOGGING_OUTPUT)

# cogs array
cogs_arr: Optional[np.ndarray] = dfcol2numpy(varname=COGS_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# gross profit array
gp_arr: Optional[np.ndarray] = dfcol2numpy(varname=GROSS_PROFIT_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# "other operating income/expense" typically includes:
#   - Restructuring charges, Asset impairments/write-offs, Gains/losses on asset sales,
#       Litigation settlements, Foreign exchange gains/losses, One-time items

# other operating income array
other_oper_inc_arr: Optional[np.ndarray] = dfcol2numpy(varname=OTHER_OPER_INC_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# other operating expense array
other_oper_exp_arr: Optional[np.ndarray] = dfcol2numpy(varname=OTHER_OPER_EXP_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# SG&A expense array
sga_exp_arr: Optional[np.ndarray] = dfcol2numpy(varname=SGA_EXP_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# R&D expense array
rd_exp_arr: Optional[np.ndarray] = dfcol2numpy(varname=RD_EXP_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# operating expense array
oper_exp_arr: Optional[np.ndarray] = dfcol2numpy(varname=OPER_EXP_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# operating income array
oper_inc_arr: Optional[np.ndarray] = dfcol2numpy(varname=OPER_INC_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# pretax income array
pretax_inc_arr: Optional[np.ndarray] = dfcol2numpy(varname=PRETAX_INC, df=dcf_actuals, logger=LOGGING_OUTPUT)

# tax expense array
tax_exp_arr: Optional[np.ndarray] = dfcol2numpy(varname=TAX_EXP_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# effective tax rate array
effective_tax_rate_arr: Optional[np.ndarray] = dfcol2numpy(varname=EFFECTIVE_TAX_RATE_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT) / 100.0

# capital expenditure array
capital_expenditure_arr: Optional[np.ndarray] = dfcol2numpy(varname=CAPEX_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# ebitda array
ebitda_arr: Optional[np.ndarray] = dfcol2numpy(varname=EBITDA_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# ebita array
ebita_arr: Optional[np.ndarray] = dfcol2numpy(varname=EBITA_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# depreciation array
depreciation_arr: Optional[np.ndarray] = dfcol2numpy(varname=DEPRECIATION_EXPENSE_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# amortization array
amortization_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=ebita_arr, arr2=oper_inc_arr, operation="subtract", logger=LOGGING_OUTPUT)

# total current assets array
total_current_assets_arr: Optional[np.ndarray] = dfcol2numpy(varname=TOTAL_CURRENT_ASSETS_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# total current liabilities array
total_current_liabilities_arr: Optional[np.ndarray] = dfcol2numpy(varname=TOTAL_CURRENT_LIABILITIES_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT)

# working capital array
working_capital_arr: Optional[np.ndarray] = array_arithmetic(
        arr1=total_current_assets_arr, arr2=total_current_liabilities_arr, operation="subtract", logger=LOGGING_OUTPUT)

# short term debt array
st_debt_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=SHORT_TERM_DEBT, df=dcf_actuals, logger=LOGGING_OUTPUT)

# short term portion of long term debt array
stlt_debt_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=SHORT_TERM_PORTION_OF_LONG_TERM_DEBT, df=dcf_actuals, logger=LOGGING_OUTPUT)

# long term debt array
lt_debt_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=LONG_TERM_DEBT, df=dcf_actuals, logger=LOGGING_OUTPUT)

# total debt array = BS_ST_BORROW + BS_CURR_PORTION_LT_DEBT + BS_LT_BORROW
total_debt_arr: Optional[np.ndarray] =  (
    array_arithmetic(
        arr1=st_debt_arr, 
        arr2=array_arithmetic(
            arr1=stlt_debt_arr, arr2=lt_debt_arr, operation="add", logger=LOGGING_OUTPUT), 
        operation="add", 
        logger=LOGGING_OUTPUT
    )
)

# preferred equity array
preferred_equity_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=PREFERRED_EQUITY, df=dcf_actuals, logger=LOGGING_OUTPUT)

# minority interest array
minority_interest_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=MINORITY_INTEREST, df=dcf_actuals, logger=LOGGING_OUTPUT)

# cash and equivalents array
cash_and_equivalents_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=CASH_AND_EQUIVALENTS, df=dcf_actuals, logger=LOGGING_OUTPUT)

# number of shares outstanding array
shares_out_arr: Optional[np.ndarray] = dfcol2numpy(
    varname=N_SHARES_OUTSTANDING, df=dcf_actuals, logger=LOGGING_OUTPUT)


# sys.exit()  # STOP <4>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING




# STEP 4: check constructed variables

# gross profit check: SALES - COGS
calculated_gross_profit_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_arr, arr2=cogs_arr, operation="subtract", logger=LOGGING_OUTPUT)
if calculated_gross_profit_arr is None:
    logger.error("❌ Failed to compute gross profit array for checking. Stopping execution.")
    sys.exit()

check_gross_margin_arr: np.ndarray = np.allclose(gp_arr, calculated_gross_profit_arr, rtol=1e-5, atol=1e-5)
logger.info(f"⚙️ CHECK PASSED? Gross profit same as Calculated Gross profit: {check_gross_margin_arr}")


# operating expense check: SG&A + R&D + Misc Expenses
calculated_oper_exp_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=array_arithmetic(arr1=sga_exp_arr, arr2=rd_exp_arr, operation="add", logger=LOGGING_OUTPUT), 
        arr2=other_oper_exp_arr, 
        operation="add", 
        logger=LOGGING_OUTPUT)
    )
if calculated_oper_exp_arr is None:
    logger.error("❌ Failed to compute operating expense array for checking. Stopping execution.")
    sys.exit()

check_oper_exp_arr: np.ndarray = np.allclose(oper_exp_arr, calculated_oper_exp_arr, rtol=1e-5, atol=1e-5)
logger.info(f"⚙️ CHECK PASSED? Operating expense same as Calculated Operating expense: {check_oper_exp_arr}")


# operating income check: EBIT = GROSS PROFIT + Other Income - Operating Expense
calculated_oper_inc_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=gp_arr,
        arr2=array_arithmetic(arr1=other_oper_inc_arr, arr2=oper_exp_arr, operation="subtract", logger=LOGGING_OUTPUT), 
        operation="add", 
        logger=LOGGING_OUTPUT)
    )
if calculated_oper_inc_arr is None:
    logger.error("❌ Failed to compute operating income array for checking. Stopping execution.")
    sys.exit()

check_oper_inc_arr: np.ndarray = np.allclose(oper_inc_arr, calculated_oper_inc_arr, rtol=1e-5, atol=1e-5)
logger.info(f"⚙️ CHECK PASSED? Operating income same as Calculated Operating income: {check_oper_inc_arr}")


# effective tax rate check
calculated_effective_tax_rate_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=tax_exp_arr, arr2=pretax_inc_arr, operation="divide", logger=LOGGING_OUTPUT)
if calculated_effective_tax_rate_arr is None:
    logger.error("❌ Failed to compute effective tax rate array for checking. Stopping execution.")
    sys.exit()

logger.info(f"📊 Calculated Effective tax rate numpy array: {calculated_effective_tax_rate_arr}")

check_effective_tax_rate_arr: np.ndarray = np.allclose(effective_tax_rate_arr, calculated_effective_tax_rate_arr, rtol=1e-5, atol=1e-5)
logger.info(f"⚙️ CHECK PASSED? Effective tax rate same as Calculated Effective tax rate: {check_effective_tax_rate_arr}")


# depreciation check
calculated_depreciation_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=ebitda_arr, arr2=ebita_arr, operation="subtract", logger=LOGGING_OUTPUT)
if calculated_depreciation_arr is None:
    logger.error("❌ Failed to compute operating income array for checking. Stopping execution.")
    sys.exit()

check_depreciation_arr: np.ndarray = np.allclose(depreciation_arr, calculated_depreciation_arr, rtol=1e-5, atol=1e-5)
logger.info(f"⚙️ CHECK PASSED? Depreciation same as Calculated Depreciation: {check_depreciation_arr}")

# sys.exit()  # STOP <5>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING




# STEP 5: compute margin arrays using array_arithmetic()

# cogs margin array
cogs_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=cogs_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if cogs_margin_arr is None:
    logger.error("❌ Failed to compute COGS margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 COGS margin numpy array: {cogs_margin_arr}")


# other operating income margin array
other_oper_inc_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=other_oper_inc_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if other_oper_inc_margin_arr is None:
    logger.error("❌ Failed to compute Other Operating Income margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Other Operating Income margin numpy array: {other_oper_inc_margin_arr}")


# other operating expense array
other_oper_exp_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=other_oper_exp_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if other_oper_exp_margin_arr is None:
    logger.error("❌ Failed to compute Other Operating Expense margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Other Operating Expense margin numpy array: {other_oper_exp_margin_arr}")


# SG&A margin array
sga_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sga_exp_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if sga_margin_arr is None:
    logger.error("❌ Failed to compute SG&A margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 SG&A margin numpy array: {sga_margin_arr}")


# R&D margin array
rd_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=rd_exp_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if rd_margin_arr is None:
    logger.error("❌ Failed to compute R&D margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 R&D margin numpy array: {rd_margin_arr}")


# capital expenditure margin array
capex_margin_arr: Optional[np.ndarray] = dfcol2numpy(varname=CAPEX_TO_SALES_VAR, df=dcf_actuals, logger=LOGGING_OUTPUT) / 100.0
if capex_margin_arr is None:
    logger.error(f"❌ Failed to convert '{CAPEX_TO_SALES_VAR}' column to numpy array. Stopping execution.")
    sys.exit()

logger.info(f"📊 CAPITAL EXPENDITURE margin numpy array: {capex_margin_arr}")


# Depreciation-to-capital expenditure margin array
depr_capex_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=depreciation_arr, 
        arr2=array_arithmetic(arr1=capex_margin_arr, arr2=sales_arr, operation="multiply", logger=LOGGING_OUTPUT), 
        operation="divide", 
        logger=LOGGING_OUTPUT)
    )
if depr_capex_arr is None:
    logger.error("❌ Failed to compute Depreciation-to-capital expenditure margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Depreciation-to-capital expenditure margin numpy array: {depr_capex_arr}")


# amortization margin array
amortization_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=amortization_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if amortization_margin_arr is None:
    logger.error("❌ Failed to compute Amortization margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 AMORTIZATION margin numpy array: {amortization_margin_arr}")


# working capital margin array
working_capital_margin_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=working_capital_arr, arr2=sales_arr, operation="divide", logger=LOGGING_OUTPUT)
if working_capital_margin_arr is None:
    logger.error("❌ Failed to compute Working Capital margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 WORKING CAPITAL margin numpy array: {working_capital_margin_arr}")

# sys.exit()  # STOP <6>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING




# STEP 6: projections using extend_array()

# sales growth projections
sales_growth_arr_extended: Optional[np.ndarray] = extend_array(
    arr=sales_growth_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    START_VALUE=float(np.nanmean(sales_growth_arr)), 
    END_VALUE=TERMINAL_GROWTH_RATE,
    logger=LOGGING_OUTPUT
)
if sales_growth_arr_extended is None:
    logger.error("❌ Failed to extend sales growth array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended sales growth numpy array: {sales_growth_arr_extended}")

# sys.exit()  # STOP <7>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# cogs margin projections
cogs_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=cogs_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from current historical average
    START_VALUE=float(np.nanmean(cogs_margin_arr)), 
    # Default assumption: maintain your advantage (or approach terminal rate if higher)
    END_VALUE=min(float(np.nanmean(cogs_margin_arr)), TERMINAL_COGS_MARGIN_RATE),
    logger=LOGGING_OUTPUT
)
if cogs_margin_arr_extended is None:
    logger.error("❌ Failed to extend COGS margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended COGS margin numpy array: {cogs_margin_arr_extended}")

# sys.exit()  # STOP <8>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# other operating income margin projections
other_oper_inc_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=other_oper_inc_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(other_oper_inc_margin_arr[-1]),
    # Default assumption: goes to ZERO in the long term
    END_VALUE=0.0,
    logger=LOGGING_OUTPUT
)
if other_oper_inc_margin_arr_extended is None:
    logger.error("❌ Failed to extend Other Operating Income margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Other Operating Income margin numpy array: {other_oper_inc_margin_arr_extended}")

# sys.exit()  # STOP <9>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# SG&A margin projections
sga_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=sga_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(sga_margin_arr[-1]), 
    # Default assumption: maintain your advantage (or approach terminal rate if higher)
    END_VALUE=min(float(np.nanmean(sga_margin_arr)), TERMINAL_SGA_MARGIN_RATE),
    logger=LOGGING_OUTPUT
)
if sga_margin_arr_extended is None:
    logger.error("❌ Failed to extend SG&A margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended SG&A margin numpy array: {sga_margin_arr_extended}")

# sys.exit()  # STOP <10>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# R&D margin projections
rd_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=rd_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(rd_margin_arr[-1]), 
    # Default assumption: maintain your advantage (or approach terminal rate if lower)
    END_VALUE=max(float(np.nanmean(rd_margin_arr)), TERMINAL_RD_MARGIN_RATE),
    logger=LOGGING_OUTPUT
)
if rd_margin_arr_extended is None:
    logger.error("❌ Failed to extend R&D margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended R&D margin numpy array: {rd_margin_arr_extended}")

# sys.exit()  # STOP <11>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# other operating expense margin projections
other_oper_exp_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=other_oper_exp_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(other_oper_exp_margin_arr[-1]),
    # Default assumption: goes to ZERO in the long term
    END_VALUE=0.0,
    logger=LOGGING_OUTPUT
)
if other_oper_exp_margin_arr_extended is None:
    logger.error("❌ Failed to extend Other Operating Expense margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Other Operating Expense margin numpy array: {other_oper_exp_margin_arr_extended}")

# sys.exit()  # STOP <12>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# effective tax rate projections
effective_tax_rate_arr_extended: Optional[np.ndarray] = extend_array(
    arr=calculated_effective_tax_rate_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(calculated_effective_tax_rate_arr[-1]),
    # Default assumption: and stay there
    END_VALUE=float(calculated_effective_tax_rate_arr[-1]),
    logger=LOGGING_OUTPUT
)
if effective_tax_rate_arr_extended is None:
    logger.error("❌ Failed to extend Effective Tax Rate array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Effective Tax Rate numpy array: {effective_tax_rate_arr_extended}")

# sys.exit()  # STOP <13>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# capex margin projections
capex_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=capex_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from historical average
    START_VALUE=float(np.nanmean(capex_margin_arr)),
    # Default assumption: and stay there (for maintenance capex)
    END_VALUE=float(np.nanmean(capex_margin_arr)),
    logger=LOGGING_OUTPUT
)
if capex_margin_arr_extended is None:
    logger.error("❌ Failed to extend Capex Margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Capex Margin numpy array: {capex_margin_arr_extended}")

# sys.exit()  # STOP <14>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# Depreciation-to-capex projections
depr_capex_arr_extended: Optional[np.ndarray] = extend_array(
    arr=depr_capex_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from last value
    START_VALUE=float(depr_capex_arr[-1]),
    # Default assumption: and get to TERMINAL_DEPR_TO_CAPEX_RATE=1.0 in the long term
    END_VALUE=TERMINAL_DEPR_TO_CAPEX_RATE,
    logger=LOGGING_OUTPUT
)
if depr_capex_arr_extended is None:
    logger.error("❌ Failed to extend Depreciation-to-Capex array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Depreciation-to-Capex numpy array: {depr_capex_arr_extended}")

# sys.exit()  # STOP <15>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# amortization margin projections
amortization_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=amortization_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from historical average
    START_VALUE=float(np.nanmean(amortization_margin_arr)),
    # Default assumption: stay there
    END_VALUE=float(np.nanmean(amortization_margin_arr)),
    logger=LOGGING_OUTPUT
)
if amortization_margin_arr_extended is None:
    logger.error("❌ Failed to extend Amortization Margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Amortization Margin numpy array: {amortization_margin_arr_extended}")

# sys.exit()  # STOP <16>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# working capital margin projections
working_capital_margin_arr_extended: Optional[np.ndarray] = extend_array(
    arr=working_capital_margin_arr, 
    forecast_periods=N_FORECAST_YEARS, 
    method=None, 
    STEP=None,
    # Default assumption: start from historical average
    START_VALUE=float(np.nanmean(working_capital_margin_arr)),
    # Default assumption: stay there
    END_VALUE=float(np.nanmean(working_capital_margin_arr)),
    logger=LOGGING_OUTPUT
)
if working_capital_margin_arr_extended is None:
    logger.error("❌ Failed to extend Working Capital Margin array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Extended Working Capital Margin numpy array: {working_capital_margin_arr_extended}")

# sys.exit()  # STOP <17>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING




# STEP 7: fill up forecasted values using apply_growth()

# sales forecast projections
sales_forecast_arr: Optional[np.ndarray] = apply_growth(
    initial_value=sales_arr[0], 
    growth_rates=sales_growth_arr_extended[1:], 
    logger=LOGGING_OUTPUT
)
if sales_forecast_arr is None:
    logger.error("❌ Failed to compute sales forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Sales forecast numpy array: {sales_forecast_arr}")

# sys.exit()  # STOP <18>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# cogs forecast projections
cogs_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=cogs_margin_arr_extended, operation="multiply", logger=LOGGING_OUTPUT)
if cogs_forecast_arr is None:
    logger.error("❌ Failed to compute cogs forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 COGS forecast numpy array: {cogs_forecast_arr}")

# sys.exit()  # STOP <19>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# gross profit forecast projections
gross_profit_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=cogs_forecast_arr, operation="subtract", logger=LOGGING_OUTPUT)
if gross_profit_forecast_arr is None:
    logger.error("❌ Failed to compute gross profit forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Gross profit forecast numpy array: {gross_profit_forecast_arr}")

# sys.exit()  # STOP <20>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# other operating income forecast projections
other_oper_inc_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=other_oper_inc_margin_arr_extended, operation="multiply", logger=LOGGING_OUTPUT)
if other_oper_inc_forecast_arr is None:
    logger.error("❌ Failed to compute Other Operating Income forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Other Operating Income forecast numpy array: {other_oper_inc_forecast_arr}")

# sys.exit()  # STOP <21>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# SG&A forecast projections
sga_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=sga_margin_arr_extended, operation="multiply", logger=LOGGING_OUTPUT)
if sga_forecast_arr is None:
    logger.error("❌ Failed to compute SG&A forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 SG&A forecast numpy array: {sga_forecast_arr}")

# sys.exit()  # STOP <22>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# R&D forecast projections
rd_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=rd_margin_arr_extended, operation="multiply", logger=LOGGING_OUTPUT)
if rd_forecast_arr is None:
    logger.error("❌ Failed to compute R&D forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 R&D forecast numpy array: {rd_forecast_arr}")

# sys.exit()  # STOP <23>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# other operating expense forecast projections
other_oper_exp_forecast_arr: Optional[np.ndarray] = array_arithmetic(
    arr1=sales_forecast_arr, arr2=other_oper_exp_margin_arr_extended, operation="multiply", logger=LOGGING_OUTPUT)
if other_oper_exp_forecast_arr is None:
    logger.error("❌ Failed to compute Other Operating Expense forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Other Operating Expense forecast numpy array: {other_oper_exp_forecast_arr}")

# sys.exit()  # STOP <24>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# operating expense forecast projections
oper_exp_forecast_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=array_arithmetic(arr1=sga_forecast_arr, arr2=rd_forecast_arr, operation="add", logger=LOGGING_OUTPUT), 
        arr2=other_oper_exp_forecast_arr, 
        operation="add", 
        logger=LOGGING_OUTPUT
    )
)
if oper_exp_forecast_arr is None:
    logger.error("❌ Failed to compute Operating Expense forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Operating Expense forecast numpy array: {oper_exp_forecast_arr}")

# sys.exit()  # STOP <25>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# EBIT forecast projections
ebit_forecast_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=gross_profit_forecast_arr,
        arr2=array_arithmetic(
            arr1=other_oper_inc_forecast_arr, 
            arr2=oper_exp_forecast_arr, 
            operation="subtract", 
            logger=LOGGING_OUTPUT
        ), 
        operation="add", 
        logger=LOGGING_OUTPUT
    )
)
if ebit_forecast_arr is None:
    logger.error("❌ Failed to compute EBIT forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 EBIT forecast numpy array: {ebit_forecast_arr}")

# sys.exit()  # STOP <26>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# NOPAT forecast projections
nopat_forecast_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=ebit_forecast_arr, 
        arr2=(1 - effective_tax_rate_arr_extended), 
        operation="multiply", 
        logger=LOGGING_OUTPUT
    )
)
if nopat_forecast_arr is None:
    logger.error("❌ Failed to compute NOPAT forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 NOPAT forecast numpy array: {nopat_forecast_arr}")

# sys.exit()  # STOP <27>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# capital expenditure forecast projections
capex_forecast_arr: Optional[np.ndarray] = (
        array_arithmetic(
            arr1=sales_forecast_arr, 
            arr2=capex_margin_arr_extended, 
            operation="multiply", 
            logger=LOGGING_OUTPUT
        )
    )
if capex_forecast_arr is None:
    logger.error("❌ Failed to compute Capex forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Capex forecast numpy array: {capex_forecast_arr}")

# sys.exit()  # STOP <28>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# depreciation forecast projections
depreciation_forecast_arr: Optional[np.ndarray] = (
        array_arithmetic(
            arr1=capex_forecast_arr, 
            arr2=depr_capex_arr_extended, 
            operation="multiply", 
            logger=LOGGING_OUTPUT
        )
    )
if depreciation_forecast_arr is None:
    logger.error("❌ Failed to compute Depreciation forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Depreciation forecast numpy array: {depreciation_forecast_arr}")

# sys.exit()  # STOP <29>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# amortization forecast projections
amortization_forecast_arr: Optional[np.ndarray] = (
        array_arithmetic(
            arr1=sales_forecast_arr, 
            arr2=amortization_margin_arr_extended, 
            operation="multiply", 
            logger=LOGGING_OUTPUT
        )
    )
if amortization_forecast_arr is None:
    logger.error("❌ Failed to compute Amortization forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Amortization forecast numpy array: {amortization_forecast_arr}")

# sys.exit()  # STOP <30>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# working capital forecast projections
working_capital_forecast_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=sales_forecast_arr, 
        arr2=working_capital_margin_arr_extended, 
        operation="multiply", 
        logger=LOGGING_OUTPUT
    )
)
if working_capital_forecast_arr is None:
    logger.error("❌ Failed to compute Working Capital forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Working Capital forecast numpy array: {working_capital_forecast_arr}")

# sys.exit()  # STOP <31>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# compute change in working capital forecast projections
change_in_working_capital_forecast_arr: Optional[np.ndarray] = delta_numpy(
    array=working_capital_forecast_arr, first_value=0.0, logger=LOGGING_OUTPUT
)
if change_in_working_capital_forecast_arr is None:
    logger.error("❌ Failed to compute Change in Working Capital forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 Change in Working Capital forecast numpy array: {change_in_working_capital_forecast_arr}")

# sys.exit()  # STOP <32>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# STEP 8: construct FCF
# FCF = nopat_forecast_arr
#       + depreciation_forecast_arr + amortization_forecast_arr
#       - capex_forecast_arr - change_in_working_capital_forecast_arr

logger.info(f"[RESULT] {nopat_forecast_arr=}")
logger.info(f"[RESULT] {depreciation_forecast_arr=}")
logger.info(f"[RESULT] {amortization_forecast_arr=}")
logger.info(f"[RESULT] {capex_forecast_arr=}")
logger.info(f"[RESULT] {change_in_working_capital_forecast_arr=}")


# fcf forecast projections
fcf_forecast_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=array_arithmetic(
            arr1=array_arithmetic(
                arr1=nopat_forecast_arr, 
                arr2=array_arithmetic(
                    arr1=depreciation_forecast_arr, 
                    arr2=amortization_forecast_arr, 
                    operation="add", 
                    logger=LOGGING_OUTPUT
                ), 
                operation="add", 
                logger=LOGGING_OUTPUT
            ), 
            arr2=capex_forecast_arr, 
            operation="subtract", 
            logger=LOGGING_OUTPUT
        ),
        arr2=change_in_working_capital_forecast_arr, 
        operation="subtract", 
        logger=LOGGING_OUTPUT 
    )
)
if fcf_forecast_arr is None:
    logger.error("❌ Failed to compute Operating Expense forecast array. Stopping execution.")
    sys.exit()

logger.info(f"[RESULT] FCF forecast numpy array: {fcf_forecast_arr}")

# fcf_forecast_arr current & projected years only
fcf_forecast_arr: np.ndarray = fcf_forecast_arr[N_HISTORICAL_YEARS-1:]
logger.info(f"[RESULT] FCF forecast numpy array (current & projected years only): {fcf_forecast_arr}")

# sys.exit()  # STOP <33>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# wacc array
wacc_constant_arr: np.ndarray = np.full(N_FORECAST_YEARS, WACC)
# logger.info(f"{wacc_constant_arr=}")

wacc_arr = apply_growth(initial_value=1.0, growth_rates=wacc_constant_arr, logger=LOGGING_OUTPUT)
# logger.info(f"{wacc_arr=}")

# sys.exit()  # STOP <34>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# STEP 9: Present value calculations

# present value of forecasted FCF
pv_forecasted_fcf_arr: Optional[np.ndarray] = (
    array_arithmetic(
        arr1=fcf_forecast_arr, 
        arr2=wacc_arr, 
        operation="divide", 
        logger=LOGGING_OUTPUT
    )
)
if pv_forecasted_fcf_arr is None:
    logger.error("❌ Failed to compute PV FCF forecast array. Stopping execution.")
    sys.exit()

logger.info(f"📊 PV FCF forecast numpy array: {pv_forecasted_fcf_arr}")


# PV FCF forecast
pv_forecasted_fcf_sum: float = float(pv_forecasted_fcf_arr.sum())
logger.info(f"📊 PV FCF forecast: {pv_forecasted_fcf_sum}")

# sys.exit()  # STOP <35>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING



# PV Terminal Value
# Terminal Value = Last FCF * (1 + g)/(wacc - g)
terminal_value: float = float(
    ((fcf_forecast_arr[-1])*(1+TERMINAL_GROWTH_RATE))/(WACC-TERMINAL_GROWTH_RATE))
# Discount Terminal Value to present term
pv_terminal_value: float = float(terminal_value/((1+WACC)**N_FORECAST_YEARS))
logger.info(f"📊 PV Terminal Value: {pv_terminal_value}")


# STEP 10: Enterprise value

# Enterprise Value
enterprise_value = pv_forecasted_fcf_sum+pv_terminal_value
logger.info(f"📊 Enterprise Value: {enterprise_value}")

# sys.exit()  # STOP <36>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# STEP 11: Equity value

# Equity Value = Enterprise Value
#                – Total Debt
#                – Preferred Equity
#                – Minority Interest
#                + Cash & Cash Equivalents
equity_value = (
    enterprise_value
    - total_debt_arr[-1]
    - preferred_equity_arr[-1]
    - minority_interest_arr[-1]
    + cash_and_equivalents_arr[-1]
)
logger.info(f"📊 Equity Value: {equity_value}")

# share value = equity value / number of outstanding shares
share_value: float = equity_value / (shares_out_arr[-1])
logger.info(f"📊 Share Value: {share_value}")

# sys.exit()  # STOP <37>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING


# price check
price: float = yahoo_finance_price_fetcher_v2(symbol=TICKER_SYMBOL)
logger.info(f"📊 Price: ${price}")


# STEP 12: Decision

# decision:
is_sell: bool = share_value < (1 - BUY_SELL_THRESHOLD) * price
if is_sell:
    logger.info(f"[DECISION] SELL {TICKER_SYMBOL}")
    sys.exit()

is_buy: bool = share_value < (1 + BUY_SELL_THRESHOLD) * price
if is_buy:
    logger.info(f"[DECISION] BUY {TICKER_SYMBOL}")
    sys.exit()

logger.info(f"[DECISION] HOLD {TICKER_SYMBOL}")

# sys.exit()  # STOP <38>   # TEMPORARY EXIT TO AVOID RUNNING THE REST OF THE CODE DURING TESTING
