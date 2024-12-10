import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
#import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib as mpl
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint


# Function to plot different types of time-series
def plot_series(df=None, column=None, series=pd.Series([]),
                label=None, ylabel=None, title=None, start=0, end=None):
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=20)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=30)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=30)
    if label:
        ax.legend(fontsize=20)
    if title:
        ax.set_title(title, fontsize=25)
    return ax

# FUNCTION FOR ARIMA MODEL EXTRACTION AND ANALYSIS
def analyze_order(order, results_dict):
    if order in results_dict:
        # Extract the summary and residuals
        summary = results_dict[order]["summary"]
        residuals = results_dict[order]["residuals"]
        
        # Print the summary
        print(f"Summary for ARIMA order {order}:\n")
        print(summary)
        
        # Plot the residuals
        plt.figure(figsize=(10, 6))
        plt.plot(residuals, label="Residuals", color="blue")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Residuals for ARIMA order {order}")
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the autocorrelation of residuals
        plt.figure(figsize=(10, 6))
        plot_acf(residuals, lags=24)
        plt.title(f"Autocorrelation of Residuals for ARIMA order {order}")
        plt.show()
        
    else:
        print(f"Order {order} not found in the results dictionary.")



# FUNCTION TO FIND THE MODEL WITH LOWEST RESIDUALS
def find_best_model(results_dict):
    # Dictionary to store the RMSE for each order
    rmse_values = {}

    # Calculate RMSE for each model
    for order, results in results_dict.items():
        residuals = results["residuals"]
        rmse = np.sqrt(np.mean(residuals**2))  # Root Mean Squared Error
        rmse_values[order] = rmse

    # Find the order with the lowest RMSE
    best_order = min(rmse_values, key=rmse_values.get)
    best_rmse = rmse_values[best_order]

    print(f"Best ARIMA order: {best_order} with RMSE: {best_rmse}")
    return best_order, best_rmse

    

# FUNCTION TO APPLY LJUNG-BOX TEST FOR A SPECIFIC ORDER
def ljung_box_test(order, results_dict, lags=24):
    if order in results_dict:
        # Extract residuals
        residuals = results_dict[order]["residuals"]
        
        # Perform Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)

        print(f"Ljung-Box Test Results for ARIMA order {order}:\n")
        #print(lb_test)

        # Interpret results
        if any(lb_test['lb_pvalue'] < 0.05):
            print("\nResiduals are autocorrelated at some lag(s) (p-value < 0.05).")
        else:
            print("\nResiduals show no significant autocorrelation (p-value >= 0.05).")
        
        return lb_test
    else:
        print(f"Order {order} not found in the results dictionary.")
        return None
    
    # Function to perform Johansen test

def analyze_cointegration(ts1, ts2, max_lags=0):
    """
    Analyze cointegration between two time series with options for Johansen parameters.
    
    Parameters:
    - ts1, ts2: Time series (arrays or pandas Series)
    - max_lags: Number of lags to include in differencing for Johansen test (default is 0)
    
    Returns:
    - Dictionary with results for Johansen test and Engle-Granger test
    """
    results = {}

    # Johansen Test for Cointegration
    johansen_results = {}
    det_orders = [-1, 0, 1]  # Possible deterministic trend orders

    """ 
    - det_order: Deterministic trend order for Johansen test (0 for no trend, 1 for linear trend, etc.)
    """
    
    for det_order in det_orders:
        johansen_test = coint_johansen(
            endog=np.column_stack((ts1, ts2)), det_order=det_order, k_ar_diff=max_lags
        )
        trace_stats = johansen_test.lr1  # Trace statistics
        critical_values = johansen_test.cvt  # Critical values (90%, 95%, 99%)
        johansen_results[f"det_order={det_order}"] = {
            "trace_stats": trace_stats,
            "critical_values": critical_values,
            "cointegration_rank": sum(trace_stats > critical_values[:, 1])  # Compare with 95% level
        }
    
    results['Johansen Test'] = johansen_results

    # Engle-Granger Test for Cointegration
    score, p_value, _ = sm.tsa.coint(ts1, ts2)
    engle_granger_results = {
        "score": score,
        "p_value": p_value,
        "cointegration": p_value < 0.05  # Null hypothesis: no cointegration
    }
    results['Engle-Granger Test'] = engle_granger_results

    return results

# function to perform a simple ols regression and retrieve the results
def ols_reg(y, x):

    """ 
    y = your dependent time series (arrays or pandas Series)
    x = your covariates/independent variables (arrays or pandas Series or Dataframe)
    """

    Y = y
    X = x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    ols_results = model.fit()
    return ols_results
