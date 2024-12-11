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
from scipy.stats import f


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

    for det_order, stats in results['Johansen Test'].items():
        print(f"\nDeterministic Order: {det_order}")
        print(f"Trace Statistics: {stats['trace_stats']}")
        print(f"Critical Values (90%, 95%, 99%): \n{stats['critical_values']}")
        print(f"Cointegration Rank: {stats['cointegration_rank']}")

    # Display Engle-Granger results
    print("\nEngle-Granger Test Results:")
    print(f"Score: {results['Engle-Granger Test']['score']}")
    print(f"P-value: {results['Engle-Granger Test']['p_value']}")
    if results['Engle-Granger Test']['cointegration'] == False:
        print("Conclusion: No Cointegration")

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

# Function to detect structural break in the data by doing the Chow Test
def sbreak_test(X, Y, last_index, first_index=None, significance=0.05):
    """
    Perform a Chow test for a structural break at a specified breakpoint or breakpoint range.
    
    Args:
        X (array-like): The independent variable(s) (explanatory variable(s)).
        Y (array-like): The dependent variable (response variable).
        last_index (int): The index of the last data point before the breakpoint.
        first_index (int, optional): The index of the first data point after the breakpoint (for ranges). Defaults to None.
        significance (float, optional): The significance level for the test (default: 0.05).
        
    Returns:
        f_stat (float): The F-statistic value.
        p_value (float): The p-value for the Chow test.
    """
    # Ensure X is a 2D array and add a constant for regression
    X = sm.add_constant(X)

    # Determine the range for splitting the data
    if first_index is None:
        first_index = last_index + 1  # Default to single-point break

    # Split the data
    X1, X2 = X[:last_index], X[first_index:]
    Y1, Y2 = Y[:last_index], Y[first_index:]

    # Fit separate regressions for the two segments
    model1 = sm.OLS(Y1, X1).fit()
    model2 = sm.OLS(Y2, X2).fit()

    # Fit the full model
    model_full = sm.OLS(Y, X).fit()

    # Compute the residual sum of squares (SSR) for each model
    SSR_full = model_full.ssr  # Full model
    SSR1 = model1.ssr  # First segment
    SSR2 = model2.ssr  # Second segment

    # Calculate the number of parameters (including constant)
    k = X.shape[1]

    # Sample sizes for each segment
    n1, n2 = len(Y1), len(Y2)

    # Compute the F-statistic
    numerator = (SSR_full - (SSR1 + SSR2)) / k
    denominator = (SSR1 + SSR2) / (n1 + n2 - 2 * k)
    f_stat = numerator / denominator

    # Calculate the p-value using the F-distribution
    p_value = 1 - f.cdf(f_stat, dfn=k, dfd=n1 + n2 - 2 * k)

    # Interpret the results based on the significance level
    if p_value < significance:
        print(f"Structural break detected at the breakpoint (p-value = {p_value:.4f})")
    else:
        print(f"No structural break detected at the breakpoint (p-value = {p_value:.4f})")

    return f_stat, p_value
