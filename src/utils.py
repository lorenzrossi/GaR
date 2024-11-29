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
        
    else:
        print(f"Order {order} not found in the results dictionary.")