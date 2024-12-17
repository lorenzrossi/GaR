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
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import lagmat
from scipy.linalg import eig
from statsmodels.tsa.stattools import adfuller



# Funzione per plottare time-series
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

# Funzione per plottare medie rolling
def plot_rolling_mean(df, column, window_sizes, ylabel=None, title=None, start=0, end=None):
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=30)
    plt.grid(True)
    
    # Plot degli actual data
    ax.plot(
        df[column][start:end],
        linestyle='-',
        linewidth=2,
        label=f'{column} (2004-2023) - Monthly Values, Actual Data',
        color='black'
    )
    
    # Plot delle medie rolling per le finestre temporali scelte
    for window_size in window_sizes:
        rolling_mean = df[column].rolling(window_size, center=True).mean()
        ax.plot(
            rolling_mean[start:end],
            linestyle='-',
            linewidth=2,
            label=f'{column}, {window_size}-period rolling mean',
            #color = 'orange'
        )
    
    if title:
        ax.set_title(title, fontsize=25)
    ax.legend(fontsize=20)
    return ax

# Funzione per plottare le varianze rolling
def plot_rolling_variance(df, column, window_sizes, ylabel=None, title=None, start=0, end=None):
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=30)
    
    for window_size in window_sizes:
        rolling_var = df[column].rolling(window_size, center=True).var()
        ax.plot(
            rolling_var[start:end],
            linestyle='--',
            linewidth=2,
            label=f'{column}: {window_size}-period rolling variance'
        )
    if title:
        ax.set_title(title, fontsize=25)
    ax.legend(fontsize=20)
    return ax

# Funzione per plottare la seasonal decomposition di una time series 
def plot_decomposition(df, column, window_sizes, model):

    """
        Argomenti della funzione:
        df (pd.Dataframe): dataframe della time series.
        column (str): nome della varibile/time series da estrarre dal dataframe.
        window_sizes: lista con i valori dei periodi coi quali fare la decomposizione (1 = annual, 4 = quarterly, 12 = monthly, 365 = daily etc)
        model (str): "additive" o "multiplicative".
    """

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 16

    #
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'

    res = sm.tsa.seasonal_decompose(df, model=model, period=window_sizes)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 15))
    res.observed.plot(ax=ax1, title='Raw')
    res.seasonal.plot(ax=ax2, title='Seasonal')
    res.trend.plot(ax=ax3, title='Trend')
    res.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    plt.show()


# Funzione per fittare modello/i ARIMA (p, d, q) ad una time series
def fit_arima_models(data, max_p=0, max_q=0, d=0, iterative=True):
    """
    Argomenti della funzione:
        data (pd.Series): time series su cui verranno fittati i modelli ARIMA.
        max_p (int): Valore massimo per l'ordine AR (p). Il valore predefinito è 0.
        max_q (int): Valore massimo per l'ordine MA (q). Il valore predefinito è 0.
        d (int): Ordine di differenziazione (d). Il valore predefinito è 0.
        iterative (Bool): parametro per iterare il fit del modello ARIMA 'FINO A' max p/q. Se si vuole fittare un singolo modello ARIMA, scrivere False. Default = True. 

    Restituisce:
        - order_results: dizionario contenente il modello come chiave. 
        Gli items (i valori) di ogni chiave sono il modello fittato (per accedere alle ulteriori componenti, come il test dell'autocorrelazione dei reidui e dell'eteroschedasticità), 
        summary, residui, log likelihood e information criteria, cov matrix dei parametri e la vrinza dei residui.
        - warning_orders: lista contenente i modelli che generano errori quando fittati. 
    """

    warning_message="Non-invertible starting MA parameters found"

    order_results = {}
    warning_orders = []

    # Se iterative==True, si fitta solo il modello iterando per ogni combinazione...
    if iterative:
        orders = [(p, d, q) for p in range(max_p + 1) for q in range(max_q + 1)]
    else:
        # ...altrimenti, si fitta il modello relativo agli ordini p,d,q specificati
        orders = [(max_p, d, max_q)]  # single-item list

    # Fitting
    for order in orders:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always") 

                # Fit del modello ARIMA
                model = ARIMA(data, order=order)
                model_fit = model.fit()

                # Inseriscoi risultati nel dizionario
                order_results[order] = {
                    "model_fit": model_fit,             # Full model_fit 
                    "summary": model_fit.summary(),     # Summary
                    "residuals": model_fit.resid,       # Residui
                    "fittedvalues": model_fit.fittedvalues,  # In-sample fitted values
                    "log_likelihood": model_fit.llf,    # Log-likelihood
                    "AIC": model_fit.aic,               # Akaike Information Criterion
                    "BIC": model_fit.bic,               # Bayesian Information Criterion
                    "HQIC": model_fit.hqic,             # Hannan-Quinn Information Criterion
                    "cov_params": model_fit.cov_params(),  # Covariance matrix dei parametri
                    #"sigma2": model_fit.sigma2          # Varianza dei residui
                }

                # Check for specific warnings
                for warning in w:
                    if warning_message in str(warning.message):
                        warning_orders.append(order)

        except Exception as e:
            print(f"Failed to fit model for order {order}: {e}")

    return order_results, warning_orders

# FUNCTION FOR ARIMA MODEL EXTRACTION AND ANALYSIS
def analyze_order(order, results_dict):

    """
    Argomenti:
    order (tuple): L'ordine ARIMA (p, d, q) da analizzare.
    results_dict (dict): Dizionario contenente i risultati dei modelli ARIMA. 
    """
    if order in results_dict:
        # Extract the model_fit object and additional details
        model_fit = results_dict[order]["model_fit"]
        residuals = results_dict[order]["residuals"]
        fitted_values = results_dict[order]["fittedvalues"]

        # Print the summary
        print(f"Summary for ARIMA order {order}:\n")
        print(model_fit.summary())

        # Print Information Criteria
        print("\nInformation Criteria:")
        print(f"AIC: {model_fit.aic:.2f}")
        print(f"BIC: {model_fit.bic:.2f}")
        print(f"HQIC: {model_fit.hqic:.2f}")

        # Print Covariance Matrix
        print("\nCovariance Matrix of Parameter Estimates:")
        print(model_fit.cov_params())

        ## Print Residual Variance
        #print("\nVariance of Residuals (Sigma^2):")
        #print(model_fit.sigma2)

        ## Perform Heteroskedasticity Test
        #heteroskedasticity_test = model_fit.test_heteroskedasticity()
        #print("\nHeteroskedasticity Test Results:")
        #print(heteroskedasticity_test)

        ## Perform Serial Correlation Test
        #serial_correlation_test = model_fit.test_serial_correlation()
        #print("\nSerial Correlation Test Results:")
        #print(serial_correlation_test)

        # Plot Residuals
        plt.figure(figsize=(10, 6))
        plt.plot(residuals, label="Residuals", color="blue")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Residuals for ARIMA order {order}")
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Autocorrelation of Residuals
        plt.figure(figsize=(10, 6))
        plot_acf(residuals, lags=24)
        plt.title(f"Autocorrelation of Residuals for ARIMA order {order}")
        plt.show()

        # Plot Fitted Values Against True Values
        plt.figure(figsize=(10, 6))
        plt.plot(model_fit.data.endog, label="True Values", color="blue", alpha=0.6)
        plt.plot(fitted_values, label="Fitted Values", color="orange", alpha=0.8)
        plt.title(f"Fitted vs True Values for ARIMA order {order}")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print(f"Order {order} not found in the results dictionary.")



# Funzione per trovare il "modello migliore" a seconda dell'infromation criterion desiderato
def find_best_model(results_dict, ic):

    """
    Argomenti:
    results_dict: Dizionario contenente i risultati dei modelli ARIMA. 
    ic (str): Information criterion (AIC, BIC...)"""
    # Dizionario per i valori degli info criterion
    ic_values = {}

    # Estraggo il valore del IC selezionato per ogni modello ARIMA
    for order, results in results_dict.items():
        info_crit = results[ic] 
        ic_values[order] = info_crit 

    # Seleziono il valore dell'IC minore tra i modelli del dizionario
    best_order = min(ic_values, key=ic_values.get)
    best_ic = ic_values[best_order]

    print(f"Best ARIMA order: {best_order} with RMSE: {best_ic}")
    return best_order, best_ic

    

# Funzione per applicare il Ljiung Box test sulla correlazione dei residui
def ljung_box_test(order, results_dict, lags=24):

    """
    Argomenti:
    order (tuple): L'ordine ARIMA (p, d, q) da analizzare.
    results_dict (dict): Dizionario contenente i risultati dei modelli ARIMA, con ogni ordine come chiave e un dizionario con residui e riepilogo come item.
    lags (int, opzionale): Numero di lag da considerare nel test di Ljung-Box. Il valore predefinito è 24 (due anni).

    Restituisce:
        pd.DataFrame o None: Un DataFrame con i risultati del test di Ljung-Box (statistica e p-value) se l'ordine è presente nel dizionario; altrimenti, restituisce None.

    """


    if order in results_dict:

        # Se l'ordine ARIMA specificato è presente nel dizionario, estrae i residui e applica il test di Ljung-Box.
        residuals = results_dict[order]["residuals"] 
        
        lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)

        print(f"Ljung-Box Test Results for ARIMA order {order}:\n")
        #print(lb_test)

        # Print dei risultati
        if any(lb_test['lb_pvalue'] < 0.05):
            print("\nResiduals are autocorrelated at some lag(s) (p-value < 0.05).")
        else:
            print("\nResiduals show no significant autocorrelation (p-value >= 0.05).")
        
        # Infine, restituisce un DataFrame con i risultati del test per ulteriori analisi.
        return lb_test
    else:
        print(f"Order {order} not found in the results dictionary.")
        return None
    
    
# Funzione per analizzare la cointegrazione tra due serie temporali con opzioni per i parametri di Johansen e confronto con test di Engle-Granger

def analyze_cointegration(ts1, ts2, max_lags=0):
    """    
    Argomenti:
    - ts1, ts2: Serie temporali (array o pandas Series)
    - max_lags: Numero di lag da includere nella differenziazione per il test di Johansen (predefinito è 0)
    
    Restituisce:
    - Dizionario con i risultati del test di Johansen e del test di Engle-Granger
    """
    results = {}  # Dizionario per salvare i risultati finali
    johansen_results = {}  # Dizionario per i risultati del test di Johansen
    det_orders = [-1, 0, 1]  # Ordini deterministici: -1 (nessuna costante), 0 (costante), 1 (costante + trend)
    best_det_order = None  # Variabile per salvare il miglior ordine deterministico
    best_rank = 0  # Variabile per salvare il rank di cointegrazione più alto trovato
    
    # TEST DI JOHANSEN PER LA COINTEGRAZIONE
    print(" Johansen Test Results ")
    for det_order in det_orders:  # Itera su ciascun ordine deterministico
        # Applico il test di Johansen
        johansen_test = coint_johansen(
            endog=np.column_stack((ts1, ts2)), det_order=det_order, k_ar_diff=max_lags
        )
        trace_stats = johansen_test.lr1  # Statistiche Trace
        critical_values = johansen_test.cvt  # Valori critici (90%, 95%, 99%)

        # Calcolo del rank di cointegrazione confrontando le statistiche con i valori critici al 95%
        rank = 0
        for i, stat in enumerate(trace_stats):
            if stat > critical_values[i, 2]:  # Confronta con il valore critico al 99%
                rank += 1
        
        # Salvo i risultati per ogni ordine deterministico
        johansen_results[f"det_order={det_order}"] = {
            "trace_stats": trace_stats,
            "critical_values": critical_values,
            "rank": rank
        }

        # Stampa i risultati intermedi
        print(f"\nDeterministic Order: {det_order}")
        print(f"Trace Statistics: {trace_stats}")
        print(f"Critical Values (90%, 95%, 99%):\n{critical_values}")
        print(f"Cointegration Rank: {rank}")

        # Aggiorno il miglior ordine deterministico se trovo un rank più alto
        if rank > best_rank:
            best_rank = rank
            best_det_order = det_order

    # Salvo il miglior ordine deterministico e il rank di cointegrazione nel dizionario dei risultati
    results['Johansen Test'] = johansen_results
    results['Best Deterministic Order'] = best_det_order
    results['Best Rank'] = best_rank

    # Stampo il risultato finale del test di Johansen
    print("\n Best Cointegration Results ")
    print(f"Best Deterministic Order: {best_det_order}")
    print(f"Best Cointegration Rank: {best_rank}")

    # TEST DI ENGLE-GRANGER PER LA COINTEGRAZIONE
    score, p_value, _ = sm.tsa.coint(ts1, ts2)
    engle_granger_results = {
        "score": score,               # Statistica del test di Engle-Granger
        "p_value": p_value,           # P-value associato al test
        "cointegration": p_value < 0.05  # Verifica se c'è cointegrazione (p-value < 0.05)
    }
    results['Engle-Granger Test'] = engle_granger_results

    # Stampa i risultati del test di Engle-Granger
    print("\n Engle-Granger Test Results ")
    print(f"Score: {score}")
    print(f"P-value: {p_value}")
    if not engle_granger_results['cointegration']:
        print("Conclusion: No Cointegration")  # Conclusione se non c'è cointegrazione
    else:
        print("Conclusion: Cointegration Found")  # Conclusione se c'è cointegrazione

    return results

# function to perform a simple ols regression and retrieve the results
#def ols_reg(y, x):
#
#    """ 
#    y = your dependent time series (arrays or pandas Series)
#    x = your covariates/independent variables (arrays or pandas Series or Dataframe)
#    """
#
#    Y = y
#    X = x
#    X = sm.add_constant(X)
#    model = sm.OLS(Y,X)
#    ols_results = model.fit()
#    return ols_results

# Function to detect structural break in the data by doing the Chow Test
#def sbreak_test(X, Y, last_index, first_index=None, significance=0.05):
#    """
#    Perform a Chow test for a structural break at a specified breakpoint or breakpoint range.
#    
#    Argomenti:
#        X: Le variabili indipendenti/explanatory.
#        Y: La variabile dipendente.
#        last_index (int): L'indice dell'ultimo punto prima del breakpoint.
#        first_index (int, opzionale): L'indice del primo punto dopo il breakpoint (per intervalli). Predefinito è None.
#        significance (float, opzionale): Il livello di significatività per il test (predefinito: 0.05).
#        
#    Restituisce:
#        f_stat (float): F statistic.
#        p_value (float): Il p-value per Chow test.
#    """
#    # Aggiungo la costante per la regression
#    X = sm.add_constant(X)
#
#    # Split dei dati in base all'indice di divisione dei due periodi
#    if first_index is None:
#        first_index = last_index + 1 
#
#    # Split 
#    X1, X2 = X[:last_index], X[first_index:]
#    Y1, Y2 = Y[:last_index], Y[first_index:]
#
#    # Fitto due regressioni per i due periodi
#    model1 = sm.OLS(Y1, X1).fit()
#    model2 = sm.OLS(Y2, X2).fit()
#
#    # Fitto il modello completo
#    model_full = sm.OLS(Y, X).fit()
#
#    # Compute the residual sum of squares (SSR) for each model
#    SSR_full = model_full.ssr  # Full model
#    SSR1 = model1.ssr  # First segment
#    SSR2 = model2.ssr  # Second segment
#
#    # Calculate the number of parameters (including constant)
#    k = X.shape[1]
#
#    # Sample sizes for each segment
#    n1, n2 = len(Y1), len(Y2)
#
#    # Compute the F-statistic
#    numerator = (SSR_full - (SSR1 + SSR2)) / k
#    denominator = (SSR1 + SSR2) / (n1 + n2 - 2 * k)
#    f_stat = numerator / denominator
#
#    # Calculate the p-value using the F-distribution
#    p_value = 1 - f.cdf(f_stat, dfn=k, dfd=n1 + n2 - 2 * k)
#
#    # Interpret the results based on the significance level
#    if p_value < significance:
#        print(f"Structural break detected at the breakpoint (p-value = {p_value:.4f})")
#    else:
#        print(f"No structural break detected at the breakpoint (p-value = {p_value:.4f})")
#
#    return f_stat, p_value

def johansen_test_manual(series1, series2, det_order=0, k_ar_diff=1):
    """
    Implementazione manuale del test di cointegrazione di Johansen.

    Argomenti:
        series1 (array-like): Prima serie temporale (es. Y).
        series2 (array-like): Seconda serie temporale (es. X).
        det_order (int): Ordine deterministico: 
                         -1 = nessun costante, 
                          0 = costante, 
                          1 = costante e trend.
        k_ar_diff (int): Numero di lag da includere nei termini di differenziazione.

    
    Restituisce:
        dict: Statistiche Trace, valori critici e rank di cointegrazione.
    """
    # STEP 1: Combina le serie in un array 2D
    data = np.column_stack((series1, series2))

    # STEP 1: Trasforma le serie in First Difference
    delta_data = np.diff(data, axis=0)
    n_obs = delta_data.shape[0]  # Numero di osservazioni dopo differenziazione

    # STEP 2: Costruzione delle matrici di lag
    X_lag = lagmat(data[:-1], maxlag=k_ar_diff, trim="both", original="ex")
    Y = delta_data[k_ar_diff:]

    # STEP 3: Regressione per stimare il modello VECM
    Z = np.hstack([X_lag, np.ones((Y.shape[0], 1))]) if det_order >= 0 else X_lag
    beta = np.linalg.lstsq(Z, Y, rcond=None)[0]
    residuals = Y - Z @ beta

    # STEP 4: Calcolo delle matrici della varianza-covarianza
    S11 = residuals.T @ residuals / n_obs  # Covarianza dei residui
    S00 = np.cov(data[:-1], rowvar=False, bias=True)

    # STEP 5: Matrice Pi e decomposizione agli autovalori
    Pi = np.linalg.inv(S00) @ S11
    eigenvalues, _ = eig(Pi)  # Calcolo degli autovalori

    # STEP 6: Calcolo delle statistiche trace
    eigenvalues = np.real(eigenvalues)  # Autovalori reali
    eigenvalues = eigenvalues[eigenvalues > 0]  # Considera solo autovalori positivi
    trace_stats = -n_obs * np.cumsum(np.log(1 - eigenvalues))

    # Valori critici per il test
    critical_values = np.array([
        [10.4741, 12.3212, 16.364],  # Rank 0
        [2.9762, 4.1296, 6.9406]     # Rank 1
    ])

    # Determina il rank
    rank = sum(trace_stats > critical_values[:, 1])  # Confronto con valori critici al 95%

    # Risultati
    results = {
        "eigenvalues": eigenvalues,
        "trace_stats": trace_stats,
        "critical_values": critical_values,
        "rank": rank
    }

    print("Eigenvalues:", results["eigenvalues"])
    print("Trace Stats:", results["trace_stats"])
    print("Critical Values (90%, 95%, 99%):\n", results["critical_values"])
    print("Cointegration Rank:", results["rank"])
    
    return results


def engle_granger_test_manual(Y, X):
    """
    Implementazione manuale del test di Engle-Granger per la cointegrazione.

    Argomenti:
        Y (array-like): Serie temporale dipendente.
        X (array-like): Serie temporale indipendente.

    Restituisce:
        dict: Risultati del test con statistiche, p-value e conclusione.
    """
    # STEP 1: Stima della regressione Y = alpha + beta * X + residui
    X = sm.add_constant(X)  # Aggiungo una costante (intercetta)
    model = sm.OLS(Y, X).fit()  # Regressione tramite OLS
    residuals = model.resid     # Residui della regressione

    # STEP 2: Test di stazionarietà sui residui (ADF Test)
    adf_result = adfuller(residuals, regression='c', autolag='AIC')
    test_statistic, p_value, used_lags, n_obs, critical_values, ic_best = adf_result

    # STEP 3: Interpretazione del risultato
    conclusion = "Cointegration Found" if p_value < 0.05 else "No Cointegration"

    # Risultati in un dizionario
    results = {
        "ADF Statistic": test_statistic,
        "P-Value": p_value,
        "Used Lags": used_lags,
        "Number of Observations": n_obs,
        "Critical Values": critical_values,
        "Residuals": residuals,
        "Conclusion": conclusion
    }

    # Stampa dei risultati
    print("### Engle-Granger Test Results ###")
    print(f"ADF Statistic: {test_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")
    print(f"Conclusion: {conclusion}")

    return results
