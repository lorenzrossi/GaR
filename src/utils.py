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
def ljung_box_test(order, results_dict, lags=12):
    if order in results_dict:
        # Extract residuals
        residuals = results_dict[order]["residuals"]
        
        # Perform Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)

        print(f"Ljung-Box Test Results for ARIMA order {order}:\n")
        print(lb_test)

        # Interpret results
    #    if any(lb_test < 0.05):
    #        print("\nResiduals are autocorrelated at some lag(s) (p-value < 0.05).")
    #    else:
    #        print("\nResiduals show no significant autocorrelation (p-value >= 0.05).")
    #    
    #    return lb_test
    #else:
    #    print(f"Order {order} not found in the results dictionary.")
    #    return None
