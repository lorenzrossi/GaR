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
from scipy.stats import f, chi2
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import lagmat
from scipy.linalg import eig, inv
from statsmodels.tsa.stattools import adfuller
from numpy.polynomial.polynomial import Polynomial



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


def ts_decomposition(time_series, period, trend_method="moving_average", poly_degree=2):
    """
    Scompone la time series in trend, stagionalità e residui. 

    Argomenti:
        time_series (pd.Series): la time series.
        period (int): la finestra temporale (e.g., 12 per dati mensili).
        trend_method (str): Metodo per estrarre il trend ("moving_average" o "polynomial").
        poly_degree (int): grado del polinomio per l'estrazione del trend (se usato "polynomial").

    Risultato:
        dict: contiene trend, stagionalità e residui.
    """
    # Step 1: Compute the trend
    if trend_method == "moving_average":
        trend = time_series.rolling(window=period, center=True).mean()
    elif trend_method == "polynomial":
        # Fit a polynomial to the time series index and values
        x = np.arange(len(time_series))
        mask = ~np.isnan(time_series)  # Ignore NaNs in the polynomial fit
        coefs = Polynomial.fit(x[mask], time_series[mask], poly_degree).convert().coef
        trend = np.polyval(np.flip(coefs), x)
        trend = pd.Series(trend, index=time_series.index)
    else:
        raise ValueError("Invalid trend_method. Choose 'moving_average' or 'polynomial'.")

    # Step 2: Detrend the series (original series - trend)
    detrended = time_series - trend

    # Step 3: Estimate seasonality by averaging over periods
    seasonal = detrended.groupby(time_series.index % period).transform("mean")

    # Step 4: Compute residuals (original series - trend - seasonality)
    residuals = time_series - trend - seasonal

    # Step 5: Plot the decomposition
    plt.figure(figsize=(20, 8))

    # Original Series
    plt.subplot(4, 1, 1)
    plt.plot(time_series, label="Original", color="blue")
    plt.title("Original Time Series")
    plt.legend()

    # Trend
    plt.subplot(4, 1, 2)
    plt.plot(trend, label=f"Trend ({trend_method})", color="orange")
    plt.title("Trend")
    plt.legend()

    # Seasonality
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label="Seasonality", color="green")
    plt.title("Seasonality")
    plt.legend()

    # Residuals
    plt.subplot(4, 1, 4)
    plt.plot(residuals, label="Residuals", color="red")
    plt.title("Residuals")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Return components as a dictionary
    return {
        "trend": trend,
        "seasonal": seasonal,
        "residuals": residuals
    }


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
        plt.figure(figsize=(25, 8))
        plt.plot(residuals, label="Residuals", color="blue")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Residuals for ARIMA order {order}")
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Autocorrelation of Residuals
        plt.figure(figsize=(25, 6))
        plot_acf(residuals, lags=24)
        plt.title(f"Autocorrelation of Residuals for ARIMA order {order}")
        plt.show()

        # Plot Fitted Values Against True Values
        plt.figure(figsize=(25, 8))
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


def compute_critical_values(n_vars, rank, alpha):
    """
    Compute critical values dynamically using the Chi-squared distribution.
    
    Parameters:
        n_vars (int): Number of variables in the dataset.
        rank (int): Current rank being tested.
        alpha (float): Significance level (e.g., 0.05, 0.01).

    Returns:
        tuple: (critical_value_trace, critical_value_maxeig)
    """
    # Degrees of freedom for the trace test
    dof_trace = (n_vars - rank) * (n_vars - rank - 1) / 2
    # Degrees of freedom for the max eigenvalue test
    dof_maxeig = 1

    # Compute critical values using the Chi-squared inverse CDF
    critical_value_trace = chi2.ppf(1 - alpha, dof_trace)
    critical_value_maxeig = chi2.ppf(1 - alpha, dof_maxeig)

    return critical_value_trace, critical_value_maxeig


def jcitest(data, max_lags=1, alpha=0.05):
    """
    Esegui il Johansen Cointegration Test simile a quello di MATLAB.

    Args:
        data (np.ndarray o pd.DataFrame): Le time series (n_obs x n_vars).
        max_lags (int): Numero massimo di lag inclusi nel modello.
        alpha (float): Livello di significatività per i valori critici (0.10, 0.05 o 0.01).

    Results:
        dict: Risultati per ogni assunzione sul trend deterministico (-1, 0, 1).
    """
    # Controlla se i dati sono in formato Pandas e convertili in array NumPy
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.to_numpy()  # Converte oggetti Pandas in array NumPy

    # Verifica che i dati siano bidimensionali
    if data.ndim != 2:
        raise ValueError("Input data must be bidimensional (n_obs x n_vars).")

    # Ottieni il numero di osservazioni e variabili
    n_obs, n_vars = data.shape

    # Funzione per creare la matrice delle variabili laggate
    def create_lagged_matrix(data, lags):
        """Crea le variabili laggate per la rappresentazione VECM."""
        lagged_data = []
        for lag in range(1, lags + 1):  # Per ogni lag fino al massimo consentito
            lagged_data.append(data[lags - lag:-lag])  # Crea la matrice laggata
        return np.column_stack(lagged_data)  # Combina le colonne in un'unica matrice

    # Calcola le differenze prime dei dati
    delta_data = np.diff(data, axis=0)  # Differenza prima lungo le righe
    Y = delta_data[max_lags:]  # Allinea i dati differenziati per i lag
    X_lag = create_lagged_matrix(delta_data, max_lags)  # Crea le variabili laggate
    Z_lag = data[max_lags:-1]  # Crea la matrice dei livelli laggati

    # Mappa dei valori critici per diversi livelli di significatività
    critical_values_map = {
        0.10: [10.49, 2.71],  # Valori critici al 90%
        0.05: [12.32, 4.13],  # Valori critici al 95%
        0.01: [16.36, 6.94],  # Valori critici al 99%
    }

    # Controlla se il valore di alpha è valido
    if alpha not in critical_values_map:
        raise ValueError("Alpha value not valid. Use 0.10, 0.05 or 0.01.")

    # Seleziona i valori critici appropriati in base ad alpha
    critical_values = critical_values_map[alpha]

    # Dizionario per salvare i risultati
    results = {}

    # Ciclo su tutte le assunzioni sul trend deterministico
    for det_order in [-1, 0, 1]:
        # Creazione delle componenti deterministiche
        if det_order == -1:
            X = X_lag  # Nessun intercetta o trend
        elif det_order == 0:
            X = np.hstack([X_lag, np.ones((X_lag.shape[0], 1))])  # Solo intercetta
        elif det_order == 1:
            trend = np.arange(1, X_lag.shape[0] + 1).reshape(-1, 1)  # Crea il trend lineare
            X = np.hstack([X_lag, np.ones((X_lag.shape[0], 1)), trend])  # Intercetta e trend

        # Esegui la regressione di Y su X e calcola i residui
        beta_X = np.linalg.lstsq(X, Y, rcond=None)[0]  # Coefficienti della regressione
        residuals_Y = Y - X @ beta_X  # Residui di Y su X

        # Esegui la regressione di Z su X e calcola i residui
        beta_Z = np.linalg.lstsq(X, Z_lag, rcond=None)[0]
        residuals_Z = Z_lag - X @ beta_Z

        # Calcola le matrici di covarianza
        S11 = residuals_Y.T @ residuals_Y / residuals_Y.shape[0]  # Covarianza dei residui YY
        S00 = residuals_Z.T @ residuals_Z / residuals_Z.shape[0]  # Covarianza dei residui ZZ
        S01 = residuals_Y.T @ residuals_Z / residuals_Y.shape[0]  # Covarianza dei residui YZ
        S10 = residuals_Z.T @ residuals_Y / residuals_Y.shape[0]  # Covarianza dei residui ZY

        # Calcola la matrice degli autovalori
        eig_matrix = inv(S00) @ S01 @ inv(S11) @ S10
        eigenvalues, eigenvectors = eig(eig_matrix)  # Autovalori e autovettori
        eigenvalues = np.real(eigenvalues[eigenvalues > 0])  # Considera solo gli autovalori reali positivi

        # Calcola le statistiche trace e max eigenvalue
        n_obs_eff = n_obs - max_lags  # Numero effettivo di osservazioni
        trace_stats = -n_obs_eff * np.cumsum(np.log(1 - eigenvalues[::-1]))[::-1]  # Statistica trace
        max_eigen_stats = -n_obs_eff * np.log(1 - eigenvalues)  # Statistica max eigenvalue

        # Funzione per calcolare i p-value usando la distribuzione Chi-quadro
        def p_value_chi2(stat, dof):
            return 1 - chi2.cdf(stat, dof)

        # Calcola i p-value per le statistiche trace e max eigenvalue
        trace_p_values = [p_value_chi2(stat, n_vars - i) for i, stat in enumerate(trace_stats)]
        max_eigen_p_values = [p_value_chi2(stat, 1) for stat in max_eigen_stats]

        # Determina il rango basato sui valori critici
        rank_trace = sum(stat > critical_values[0] for stat in trace_stats)  # Rango per trace
        rank_maxeig = sum(stat > critical_values[1] for stat in max_eigen_stats)  # Rango per max eigenvalue

        # Crea una tabella dei risultati formattata
        results_table = "\n".join([
            f"Deterministic Trend Assumption: {det_order}",
            f"Significance Level: {alpha:.2f}",
            "r | h  |  stat    | cValue   | pValue  | eigVal",
            "-" * 60,
            *(f"{i} | {stat > critical_values[0]} | {stat:.4f} | {critical_values[0]:.4f} | {p_value:.4f} | {eigenvalue:.4f}"
              for i, (stat, p_value, eigenvalue) in enumerate(zip(trace_stats, trace_p_values, eigenvalues))),
            f"\nRank: {rank_trace}",
            #f"Rango MaxEig: {rank_maxeig}\n",
            "=" * 60
        ])

        # Salva i risultati per il det_order corrente
        results[f"det_order_{det_order}"] = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "trace_stats": trace_stats,
            "trace_p_values": trace_p_values,
            "max_eigen_stats": max_eigen_stats,
            "max_eigen_p_values": max_eigen_p_values,
            "critical_values": critical_values,
            "rank_trace": rank_trace,
            "rank_maxeig": rank_maxeig,
            "results_table": results_table
        }

    return results




def egcitest(Y, X, max_lags=0):
    """
    Implementazione manuale del test di Engle-Granger per la cointegrazione.

    Argomenti:
        Y (array-like): Serie temporale dipendente.
        X (array-like): Serie temporale indipendente.
        max_lags (int): max_lags (int): Numero massimo di lag inclusi nel ADF test (default = 0).

    Restituisce:
        dict: Risultati del test con statistiche, p-value e conclusione.
    """

    if not isinstance(X, (pd.Series, pd.DataFrame)):
        X = pd.Series(X)
    if not isinstance(Y, (pd.Series, pd.DataFrame)):
        Y = pd.Series(Y)

    # Regressione OLS di Y su X
    #if "const" not in X.columns: # Controllo se X ha già una colonna di costante; in caso contrario, la aggiungo.
    X = sm.add_constant(X)  # Aggiunge un termine costante al modello
    model = sm.OLS(Y, X).fit()  # Esegue la regressione OLS
    residuals = model.resid  # Estrae i residui dal modello

    # Test ADF sui residui per valutare la stazionarietà
    adf_result = adfuller(residuals, maxlag=max_lags, autolag=None if max_lags > 0 else "AIC")
    adf_statistic = adf_result[0]  # Statistica del test ADF
    p_value = adf_result[1]  # p-value del test ADF
    critical_values = adf_result[4]  # Valori critici del test ADF

    # Calcolo dell'errore standard residuo
    # L'errore standard residuo tiene conto dei gradi di libertà
    residual_std_error = np.sqrt(np.sum(residuals**2) / (len(residuals) - len(model.params)))

    # Conclusione sulla cointegrazione
    conclusion = "Cointegrated" if p_value < 0.05 else "Not Cointegrated"

    # Stampiamo i risultati del test
    print("\nEngle-Granger Test Results")
    print("OLS Results:")
    print(f"  Coefficients: {model.params}")  # Coefficienti stimati dal modello
    print(f"  Residual SE: {residual_std_error:.4f}")  # Errore standard residuo
    print("\nADF Test on the Residuals:")
    print(f"  Stat: {adf_statistic:.4f}")  # Statistica del test ADF
    print(f"  P-value: {p_value:.4f}")  # p-value del test
    print("  Critical Values:")
    for key, value in critical_values.items():
        print(f"    {key}: {value:.4f}")  # Stampiamo i valori critici formattati
    print(f"Conclusion: {conclusion}")  # Stampiamo la conclusione

    # Restituiamo i risultati come dizionario
    return {
        "coefficients": model.params,  # Coefficienti del modello OLS
        "residuals": residuals,  # Residui del modello
        "adf_statistic": adf_statistic,  # Statistica del test ADF
        "p_value": p_value,  # p-value del test ADF
        "critical_values": critical_values,  # Valori critici del test ADF
        "conclusion": conclusion,  # Conclusione sulla cointegrazione
    }



def granger_causality_test(y, x, max_lag):
    """
    Implementazione manuale del test di causalità di Granger.

    Argomenti:
        y (np.ndarray): Variabile dipendente (serie target).
        x (np.ndarray): Variabile indipendente (serie predittore).
        max_lag (int): Numero massimo di ritardi da includere.

    Risultato:
        dict: Contiene F-statistic, p-value e conclusione del test.
    """
    # Se x è un array numpy, convertilo in una Serie Pandas
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    # Se y è un array numpy, convertilo in una Serie Pandas
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Controllo della lunghezza delle serie
    if len(x) != len(y):
        raise ValueError("Input time series must have the same lenght.")
    # Verifica che il numero massimo di lag sia compatibile con la lunghezza della serie
    if max_lag >= len(y):
        raise ValueError("max_lag must be less than the lenght of input time series.")

    # Funzione per creare una matrice di ritardi per una serie temporale
    def create_lagged_matrix(series, lags):
        # Costruisce una matrice con colonne che rappresentano ritardi successivi della serie
        return np.column_stack([series.shift(i).to_numpy() for i in range(1, lags + 1)])[lags:]

    # Creazione delle matrici di ritardi per Y e X
    lagged_y = create_lagged_matrix(y, max_lag)  # Matrice dei ritardi per Y
    lagged_x = create_lagged_matrix(x, max_lag)  # Matrice dei ritardi per X
    trimmed_y = y[max_lag:].to_numpy()  # Allinea Y con le matrici di ritardo

    # Modello ristretto: regressione di Y sui propri ritardi
    restricted_model_data = sm.add_constant(lagged_y)  # Aggiunge una costante per la regressione
    restricted_model = sm.OLS(trimmed_y, restricted_model_data).fit()  # Esegue la regressione
    ssr_restricted = restricted_model.ssr  # Somma dei residui quadrati (modello ristretto)

    # Modello non ristretto: regressione di Y sui ritardi di Y e X
    unrestricted_model_data = sm.add_constant(np.hstack([lagged_y, lagged_x]))  # Aggiunge X ai predittori
    unrestricted_model = sm.OLS(trimmed_y, unrestricted_model_data).fit()  # Esegue la regressione
    ssr_unrestricted = unrestricted_model.ssr  # Somma dei residui quadrati (modello non ristretto)

    # Calcolo della statistica F per il test di causalità di Granger
    n_lags = max_lag  # Gradi di libertà per il numeratore
    df_resid = len(trimmed_y) - unrestricted_model_data.shape[1]  # Gradi di libertà per il denominatore
    f_stat = ((ssr_restricted - ssr_unrestricted) / n_lags) / (ssr_unrestricted / df_resid)  # Formula della statistica F

    # Calcolo del p-value utilizzando la distribuzione F
    p_value = 1 - f.cdf(f_stat, n_lags, df_resid)  # Calcola il p-value dalla cdf della distribuzione F

    # Step 4: Conclusion
    conclusion = "X Granger-causes Y" if p_value < 0.05 else "No Granger causality from X to Y"

    # Print Results
    print("\nGranger Causality Test Results")
    print(f"Restricted SSR: {ssr_restricted:.4f}")
    print(f"Unrestricted SSR: {ssr_unrestricted:.4f}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Conclusion: {conclusion}")

    # Return results as a dictionary
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "ssr_restricted": ssr_restricted,
        "ssr_unrestricted": ssr_unrestricted,
        "conclusion": conclusion,
    }

# Funzione per plottare la seasonal decomposition di una time series 
#def plot_decomposition(df, column, window_sizes, model):
#
#    """
#        Argomenti della funzione:
#        df (pd.Dataframe): dataframe della time series.
#        column (str): nome della varibile/time series da estrarre dal dataframe.
#        window_sizes: lista con i valori dei periodi coi quali fare la decomposizione (1 = annual, 4 = quarterly, 12 = monthly, 365 = daily etc)
#        model (str): "additive" o "multiplicative".
#    """
#
#    mpl.rcParams['font.family'] = 'Arial'
#    mpl.rcParams['font.size'] = 16
#
#    #
#    mpl.rcParams['figure.facecolor'] = 'white'
#    mpl.rcParams['axes.facecolor'] = 'white'
#    mpl.rcParams['savefig.facecolor'] = 'white'
#
#    res = sm.tsa.seasonal_decompose(df, model=model, period=window_sizes)
#    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 15))
#    res.observed.plot(ax=ax1, title='Raw')
#    res.seasonal.plot(ax=ax2, title='Seasonal')
#    res.trend.plot(ax=ax3, title='Trend')
#    res.resid.plot(ax=ax4, title='Residual')
#    plt.tight_layout()
#    plt.show()

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
