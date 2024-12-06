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

def analyze_cointegration(df):

    """
    Perform Cointegration test using both the Johansen and Engle-Granger test

    Parameters: df (pd.DataFrame): A DataFrame with two time series columns.

    Returns: dict: A dictionary with Johansen rank results, Engle-Granger score, and p-value.
    
    """
     # Johansen Test
    johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)
    trace_stat = johansen_result.lr1  # Trace statistics
    critical_values = johansen_result.cvt  # Critical values (90%, 95%, 99%)

    #for rank in [0, 1]:
    #    print(f"Rank {rank}:")
    #    print(f"Trace Statistic: {trace_stat[rank]:.4f}")
    #    print(f"Critical Value at 95%: {critical_values[rank][1]:.4f}")  # 95% level
    #    print()

    # Determine Johansen rank (0 or 1)
    rank = None
    if trace_stat[0] < critical_values[0][1]:
        rank = 0
    elif trace_stat[1] < critical_values[1][1]:
        rank = 1
    else:
        rank = ">1"

    # Engle-Granger Test
    score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])

    # Prepare Results
    results = {
        "Johansen": {
            "Trace Statistics": trace_stat.tolist(),
            "Critical Values": critical_values.tolist(),
            "Rank": rank
        },
        "Engle-Granger": {
            "Score": score,
            "P-Value": pvalue
        }
    }

    return results
