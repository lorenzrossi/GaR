import pandas as pd
import numpy as np
import os
#import scipy
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import f, chi2
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import lagmat
from scipy.linalg import inv
from statsmodels.tsa.stattools import adfuller
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from scipy.interpolate import interp1d
import statsmodels.api as sm




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
        plt.plot(residuals[1:], label="Residuals", color="blue")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Residuals for ARIMA order {order}")
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Autocorrelation of Residuals
        plt.figure(figsize=(25, 6))
        plot_acf(residuals[1:], lags=24)
        plt.title(f"Autocorrelation of Residuals for ARIMA order {order}")
        plt.show()

        # Plot Fitted Values Against True Values
        plt.figure(figsize=(25, 8))
        plt.plot(model_fit.data.endog, label="True Values", color="blue", alpha=0.6)
        plt.plot(fitted_values[1:], label="Fitted Values", color="orange", alpha=0.8)
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
    

def jcitest(endog,  det_order, k_ar_diff):
    if det_order not in [-1, 0, 1]:
        warnings.warn(
            "Critical values are only available for a det_order of -1, 0, or 1.",
            stacklevel=2,
        )
    if endog.shape[1] > 12:
        warnings.warn(
            "Critical values are only available for time series with 12 variables at most.",
            stacklevel=2,
        )

    def compute_p_value(stat, crit_vals):
        """
        Compute approximate p-value for Johansen test statistic using interpolation.

        Parameters
        ----------
        stat : float
            The test statistic value (trace or max eigenvalue).
        crit_vals : array-like
            Critical values for the test at different significance levels (90%, 95%, 99%).

        Returns
        -------
        float
            Approximate p-value.
        """
        significance_levels = np.array([0.10, 0.05, 0.01])  # Corresponding significance levels

        if stat < crit_vals[0]:
            return 1.0  # Very high p-value (no evidence of cointegration)
        elif stat > crit_vals[-1]:
            return 0.01  # Very low p-value (strong evidence of cointegration)
        else:
            return np.interp(stat, crit_vals, significance_levels[::-1])

    def detrend(y, order):
        if order == -1:
            return y
        return OLS(y, np.vander(np.linspace(-1, 1, len(y)), order + 1)).fit().resid

    def resid(y, x):
        if x.size == 0:
            return y
        return y - np.dot(x, np.dot(np.linalg.pinv(x), y))

    endog = np.asarray(endog)
    nobs, neqs = endog.shape
    f = 0 if det_order > -1 else det_order

    endog = detrend(endog, det_order)
    dx = np.diff(endog, 1, axis=0)
    z = lagmat(dx, k_ar_diff)
    z = z[k_ar_diff:]
    z = detrend(z, f)
    dx = dx[k_ar_diff:]
    dx = detrend(dx, f)
    r0t = resid(dx, z)
    lx = endog[: (endog.shape[0] - k_ar_diff)][1:]
    dx = detrend(lx, f)
    rkt = resid(dx, z)
    skk = np.dot(rkt.T, rkt) / rkt.shape[0]
    sk0 = np.dot(rkt.T, r0t) / rkt.shape[0]
    s00 = np.dot(r0t.T, r0t) / r0t.shape[0]
    sig = np.dot(sk0, np.dot(inv(s00), sk0.T))
    tmp = inv(skk)
    au, du = np.linalg.eig(np.dot(tmp, sig))
    temp = inv(np.linalg.cholesky(np.dot(du.T, np.dot(skk, du))))
    dt = np.dot(du, temp)
    auind = np.argsort(au)
    aind = np.flipud(auind)
    a = au[aind]
    d = dt[:, aind]
    non_zero_d = d.flat != 0
    if np.any(non_zero_d):
        d *= np.sign(d.flat[non_zero_d][0])

    lr1 = np.zeros(neqs)
    lr2 = np.zeros(neqs)
    cvm = np.zeros((neqs, 3))
    cvt = np.zeros((neqs, 3))
    p_values_trace = np.zeros(neqs)
    p_values_max_eig = np.zeros(neqs)
    iota = np.ones(neqs)
    t = rkt.shape[0]

    for i in range(neqs):
        lr1[i] = -t * np.sum(np.log(iota - a)[i:])
        lr2[i] = -t * np.log(1 - a[i])

        cvm[i, :] = c_sja(neqs - i, det_order)
        cvt[i, :] = c_sjt(neqs - i, det_order)
        aind[i] = i

        # Compute p-values
        p_values_trace[i] = compute_p_value(lr1[i], cvt[i, :])
        p_values_max_eig[i] = compute_p_value(lr2[i], cvm[i, :])

    
    # Display all values of the result
    result_dict = {
        "Eigenvalues": a,
        "Eigenvectors": d,
        "Trace Test Statistics": lr1,
        "Max Eigenvalue Statistics": lr2,
        "Critical Values (Trace)": cvt,
        "Critical Values (Max Eigenvalue)": cvm,
        "P-values (Trace)": p_values_trace,
        "P-values (Max Eigenvalue)": p_values_max_eig,
    }

    return result_dict


#def compute_critical_values(rank, alpha, det_order):
#    """
#    Calcola i valori critici per il test di Johansen basati su tabelle predefinite.
#
#    Args:
#        n_vars (int): Numero di variabili nel dataset (es. 2).
#        rank (int): Rango attualmente testato (es. 0 o 1).
#        alpha (float): Livello di significatività (es. 0.10, 0.05, 0.01).
#        det_order (int): Assunzione sul trend deterministico (-1, 0, 1).
#
#    Returns:
#        tuple: (valore_critico_trace, valore_critico_maxeig)
#    """
#    # Tabelle dei valori critici (esempio per n_vars = 2)
#    johansen_critical_values = {
#        -1: {  # Nessuna intercetta o trend
#            0: {0.10: 13.43, 0.05: 15.67, 0.01: 20.04},
#            1: {0.10: 2.71, 0.05: 3.84, 0.01: 6.65},
#        },
#        0: {  # Solo intercetta
#            0: {0.10: 18.17, 0.05: 19.22, 0.01: 25.32},
#            1: {0.10: 7.52, 0.05: 9.16, 0.01: 12.25},
#        },
#        1: {  # Intercetta e trend lineare
#            0: {0.10: 23.15, 0.05: 23.78, 0.01: 30.45},
#            1: {0.10: 10.55, 0.05: 12.39, 0.01: 15.72},
#        },
#    }
#
#    # Recupera i valori critici basati su det_order, rank e alpha
#    try:
#        critical_value_trace = johansen_critical_values[det_order][rank][alpha]
#        critical_value_maxeig = johansen_critical_values[det_order][rank][alpha]
#    except KeyError:
#        raise ValueError("Valori critici non disponibili per i parametri specificati.")
#
#    return critical_value_trace, critical_value_maxeig
#
#def jcitest(data, max_lags=1, alphas=0.05):
#    """
#    Esegui il Johansen Cointegration Test con valori critici personalizzabili.
#
#    Args:
#        data (np.ndarray o pd.DataFrame): Dataset delle serie temporali (n_obs x n_vars).
#        max_lags (int): Numero massimo di lag inclusi nel modello.
#        alphas (tuple): Livelli di significatività da testare (es. (0.10, 0.05, 0.01)).
#        dist_type (str): Tipo di distribuzione per i valori critici ("chi2", "gamma", "skewed_t").
#        kwargs: Parametri addizionali per il calcolo dei valori critici.
#
#    Returns:
#        dict: Risultati per ogni assunzione sul trend deterministico (-1, 0, 1).
#    """
#    # Assicura che alphas sia una tupla
#    if isinstance(alphas, (float, int)):  # Se è un singolo numero
#        alphas = (alphas,)
#    # Se i dati sono un DataFrame Pandas o una Serie, convertili in un array NumPy
#    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
#        data = data.to_numpy()
#
#    # Controlla che i dati siano bidimensionali (n_obs x n_vars)
#    if data.ndim != 2:
#        raise ValueError("Input data must be 2D (n_obs x n_vars).")
#
#    # Determina il numero di osservazioni (n_obs) e variabili (n_vars) dai dati
#    n_obs, n_vars = data.shape
#
#    # Funzione interna per creare una matrice laggata
#    def create_lagged_matrix(data, lags):
#        """
#        Crea una matrice laggata concatenando i lag specificati.
#
#        Args:
#            data (np.ndarray): Dati delle serie temporali.
#            lags (int): Numero di lag.
#
#        Returns:
#            np.ndarray: Matrice laggata.
#        """
#        # Costruisce la matrice laggata concatenando le colonne corrispondenti ai lag
#        return np.column_stack([data[i:-(lags - i) or None] for i in range(lags)])
#
#    # Calcola le differenze prime dei dati per ottenere la serie stazionaria
#    delta_data = np.diff(data, axis=0)
#
#    # Inizializza un dizionario per salvare i risultati
#    results = {}
#
#    # Itera sulle assunzioni del trend deterministico (-1, 0, 1)
#    for det_order in [-1, 0, 1]:
#        #print(f"Processing deterministic trend assumption: det_order={det_order}")
#
#        # Crea la matrice laggata dalle differenze prime
#        X_lag = create_lagged_matrix(delta_data, max_lags)
#
#        # Aggiunge componenti deterministiche basate su det_order
#        if det_order == -1:
#            X = X_lag  # Nessuna intercetta o trend
#        elif det_order == 0:
#            # Aggiunge una colonna di 1 per rappresentare l'intercetta
#            X = np.hstack([X_lag, np.ones((X_lag.shape[0], 1))])
#        else:
#            # Aggiunge sia l'intercetta che un trend lineare
#            trend = np.arange(1, X_lag.shape[0] + 1).reshape(-1, 1)
#            X = np.hstack([X_lag, np.ones((X_lag.shape[0], 1)), trend])
#
#        # Allinea i dati differenziati con la matrice laggata
#        Y = delta_data[max_lags:]  # Rimuove i primi max_lags osservazioni per allineamento
#
#        # Esegue una regressione di Y su X per ottenere i residui di Y
#        beta_X = np.linalg.lstsq(X, Y, rcond=None)[0]  # Coefficienti della regressione
#        residuals_Y = Y - X @ beta_X  # Residui di Y calcolati come Y - Xβ
#
#        # Allinea i livelli originali (Z) con le variabili laggate
#        Z = data[max_lags:max_lags + X.shape[0]]  # Rimuove osservazioni iniziali
#        # Esegue una regressione di Z su X per ottenere i residui di Z
#        beta_Z = np.linalg.lstsq(X, Z, rcond=None)[0]  # Coefficienti della regressione
#        residuals_Z = Z - X @ beta_Z  # Residui di Z calcolati come Z - Xβ
#
#        # Calcola le matrici di covarianza usando i residui
#        S11 = residuals_Y.T @ residuals_Y / residuals_Y.shape[0]  # Covarianza di Y
#        S00 = residuals_Z.T @ residuals_Z / residuals_Z.shape[0]  # Covarianza di Z
#        S01 = residuals_Y.T @ residuals_Z / residuals_Y.shape[0]  # Covarianza incrociata Y-Z
#        S10 = residuals_Z.T @ residuals_Y / residuals_Y.shape[0]  # Covarianza incrociata Z-Y
#
#        # Costruisce la matrice per calcolare autovalori e autovettori
#        eig_matrix = inv(S00) @ S01 @ inv(S11) @ S10
#        eigenvalues, eigenvectors = eig(eig_matrix)  # Calcola autovalori e autovettori
#        eigenvalues = np.real(eigenvalues[eigenvalues > 0])  # Considera solo autovalori reali positivi
#
#        # Calcola le statistiche trace e max eigenvalue
#        n_obs_eff = n_obs - max_lags  # Numero effettivo di osservazioni
#        trace_stats = -n_obs_eff * np.cumsum(np.log(1 - eigenvalues[::-1]))[::-1]  # Statistica trace
#        max_eigen_stats = -n_obs_eff * np.log(1 - eigenvalues)  # Statistica max eigenvalue
#
#        # Inizializza un dizionario per salvare i risultati per ogni livello di significatività
#        results[f"det_order_{det_order}"] = {}
#
#        # Itera su ogni livello di significatività fornito
#        for alpha in alphas:
#            #print(f"Testing significance level: alpha={alpha}")
#
#            # Calcola i valori critici basati sulle tabelle di Johansen
#            trace_critical_values, _ = zip(
#                *[compute_critical_values(r, alpha, det_order) for r in range(len(eigenvalues))]
#            )
#
#            # Calcola i p-value per le statistiche trace e max eigenvalue
#            trace_p_values = [1 - chi2.cdf(stat, n_vars - i) for i, stat in enumerate(trace_stats)]
#            _ = [1 - chi2.cdf(stat, 1) for stat in max_eigen_stats]
#
#            # Determina il rango basato sulle statistiche trace
#            rank_trace = sum(stat > crit for stat, crit in zip(trace_stats, trace_critical_values))
#
#            # Crea una tabella formattata per visualizzare i risultati
#            results_table = "\n".join([
#                f"Deterministic trend assumption: {det_order}",
#                f"Significance level: {alpha:.2f}",
#                "r | h  |  stat    | cValue   | pValue  | eigVal",
#                "-" * 60,
#                *(f"{i} | {stat > trace_critical_values[i]} | {stat:.4f} | {trace_critical_values[i]:.4f} | {p_value:.4f} | {eigenvalue:.4f}"
#                  for i, (stat, p_value, eigenvalue) in enumerate(zip(trace_stats, trace_p_values, eigenvalues))),
#                f"\nRank: {rank_trace}",
#                "=" * 60
#            ])
#
#            # Salva i risultati per questo livello di significatività
#            results[f"det_order_{det_order}"][f"alpha_{alpha:.2f}"] = {
#                "eigenvalues": eigenvalues,  # Autovalori calcolati
#                "trace_stats": trace_stats,  # Statistiche trace
#                "trace_critical_values": trace_critical_values,  # Valori critici per trace
#                "trace_p_values": trace_p_values,  # P-value per le statistiche trace
#                "rank_trace": rank_trace,  # Rango stimato
#                "results_table": results_table,  # Tabella formattata
#            }
#
#    if isinstance(results, dict):  # Ensure `results` is a dictionary
#        for det_order, alpha_results in results.items():
#            #print(f"Processing det_order: {det_order}")
#            if isinstance(alpha_results, dict):  # Ensure `alpha_results` is a dictionary
#                for alpha, details in alpha_results.items():
#                    #print(f"Processing alpha: {alpha}")
#                    if "results_table" in details:  # Ensure key exists
#                        print(details["results_table"])
#                    else:
#                        print(f"Missing 'results_table' in details: {details}")
#            else:
#                print(f"Unexpected type for alpha_results: {type(alpha_results)}")
#    else:
#        print(f"Unexpected type for results: {type(results)}")
#
#    # Ritorna il dizionario dei risultati
#    return results
#
#
#
#def print_jcitest_results(results):
#    """
#    Prints the Johansen Cointegration Test results in a tabular format.
#
#    Parameters:
#        results (dict): The results from the `jcitest` function.
#    """
#    for det_order, result in results.items():
#        print(f"Deterministic Trend Assumption: {det_order}")
#        print(f"{'-' * 60}")
#        
#        table_data = []
#        headers = ["r", "h", "Trace Stat", "Critical Value", "p-Value", "Eigenvalue"]
#        
#        for i, (trace_stat, crit_value, p_value, eigenvalue) in enumerate(
#            zip(result["trace_stats"], result["trace_critical_values"], 
#                result["trace_p_values"], result["eigenvalues"])
#        ):
#            h = "True" if trace_stat > crit_value else "False"
#            table_data.append([i, h, f"{trace_stat:.4f}", f"{crit_value:.4f}", f"{p_value:.4f}", f"{eigenvalue:.4f}"])
#        
#        print(tabulate(table_data, headers=headers, tablefmt="grid"))
#        print(f"Rank (r): {result['rank_trace']}")
#        print(f"{'=' * 60}\n")


#def egcitest(data, creg='c', rreg='ADF', lags=0, test='t1', alpha=0.05,
#             response_variable=None, predictor_variables=None):
#    """
#    Test di cointegrazione Engle-Granger.
#    
#    Parametri:
#    - data: numpy.ndarray, pandas.DataFrame o pandas.Series (i dati di input)
#    - creg: str, tipo di regressione ('nc', 'c', 'ct', 'ctt')
#    - cvec: list o numpy.ndarray, coefficienti specificati dall'utente
#    - rreg: str, metodo per la regressione residua ('ADF' o 'PP')
#    - lags: int, numero di lag per il test residuo
#    - test: str, tipo di test ('t1', 't2')
#    - alpha: float, livello di significatività
#    - response_variable: str, variabile di risposta (se data è un DataFrame)
#    - predictor_variables: list, variabili predittive (se data è un DataFrame)
#    
#    Ritorna:
#    - results: dict, contiene statistiche del test, p-value e valori critici
#    """
#
#    # Converti i dati in array numpy se sono un DataFrame o una Series
#    if isinstance(data, pd.DataFrame):
#        if response_variable:  # Se è specificata una variabile di risposta
#            y = data[response_variable].values  # Estrai la variabile di risposta
#        else:
#            y = data.iloc[:, 0].values  # Per default usa la prima colonna come risposta
#        if predictor_variables:  # Se sono specificate variabili predittive
#            x = data[predictor_variables].values  # Estrai le variabili predittive
#        else:
#            x = data.drop(columns=[data.columns[0]]).values  # Usa tutte le altre colonne
#    elif isinstance(data, (np.ndarray, pd.Series)):  # Se i dati sono un array numpy o una Series
#        y = data[:, 0] if data.ndim > 1 else data  # La prima colonna è la variabile di risposta
#        x = data[:, 1:] if data.ndim > 1 else None  # Le altre colonne sono predittori
#    else:
#        raise ValueError("I dati devono essere un array numpy, un DataFrame pandas o una Series.")
#
#    # Controlla che ci siano abbastanza predittori
#    if x is None or x.shape[1] < 1:
#        raise ValueError("I dati devono avere almeno una variabile predittiva.")
#    # Verifica che i lag siano validi
#    if not isinstance(lags, int) or lags < 0:
#        raise ValueError("Il numero di lag deve essere un intero non negativo.")
#    # Controlla che il tipo di regressione sia valido
#    if creg not in ['nc', 'c', 'ct', 'ctt']:
#        raise ValueError("Valore di 'creg' non valido. Usa ['nc', 'c', 'ct', 'ctt'].")
#    # Controlla che il metodo di regressione residua sia valido
#    if rreg not in ['ADF', 'PP']:
#        raise ValueError("Valore di 'rreg' non valido. Usa ['ADF', 'PP'].")
#    # Controlla che il tipo di test sia valido
#    if test not in ['t1', 't2']:
#        raise ValueError("Valore di 'test' non valido. Usa ['t1', 't2'].")
#    # Controlla che alpha sia compreso tra 0 e 1
#    if not (0 < alpha < 1):
#        raise ValueError("Alpha deve essere compreso tra 0 e 1.")
#
#    # Step 1: Esegui la regressione di cointegrazione
#    if creg == 'nc':  # Nessuna costante
#        x_design = x
#    elif creg == 'c':  # Solo costante
#        x_design = np.hstack((np.ones((x.shape[0], 1)), x))  # Aggiungi una colonna di 1
#    elif creg == 'ct':  # Costante e trend
#        trend = np.arange(1, x.shape[0] + 1).reshape(-1, 1)  # Crea una colonna con i numeri da 1 a n
#        x_design = np.hstack((np.ones((x.shape[0], 1)), trend, x))  # Aggiungi trend e costante
#    elif creg == 'ctt':  # Costante, trend e trend quadratico
#        trend = np.arange(1, x.shape[0] + 1).reshape(-1, 1)  # Crea una colonna di trend
#        trend2 = trend ** 2  # Crea una colonna di trend quadratico
#        x_design = np.hstack((np.ones((x.shape[0], 1)), trend, trend2, x))  # Aggiungi tutto
#
#    # Esegui la regressione OLS
#    coeffs = np.linalg.lstsq(x_design, y, rcond=None)[0]  # Calcola i coefficienti con OLS
#    residuals = y - x_design @ coeffs  # Calcola i residui della regressione
#
#    # Step 2: Esegui il test di radice unitaria sui residui
#    if rreg == 'ADF':  # Usa il test ADF
#        adf_result = adfuller(residuals, maxlag=lags, regression='c' if test == 't1' else 'ct')
#        test_statistic, p_value, critical_values = adf_result[0], adf_result[1], adf_result[4]
#    elif rreg == 'PP':  # Placeholder per il test PP (non implementato)
#        raise NotImplementedError("Il test PP non è ancora implementato.")
#
#    # Interpola i valori critici se necessario (non implementato in questo esempio)
#
#    # Step 3: Prepara i risultati
#    results = {
#        'test_statistic': test_statistic,  # Statistica del test
#        'p_value': p_value,  # P-value del test
#        'critical_values': critical_values,  # Valori critici
#        'coefficients': coeffs,  # Coefficienti della regressione
#        'residuals': residuals  # Residui della regressione
#    }
#
#    # Stampa i risultati
#    print("\nEngle-Granger Cointegration Test Results")
#    print(f"Statistic: {test_statistic:.4f}")
#    print(f"P-Value: {p_value:.4f}")
#    print("Critical Values:")
#    for key, value in critical_values.items():
#        print(f"  {key}: {value:.4f}")
#
#    return results

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

def gctest(data, num_lags=1, integration=0, constant=True, trend=False, x=None, alpha=0.05,
           test_type='chi-square', cause_variables=None, effect_variables=None, condition_variables=None,
           predictor_variables=None):
    """
    Test di causalità di Granger e blocco di esogeneità.
    
    Parametri:
    - data: numpy.ndarray o pandas.DataFrame (dati delle serie temporali)
    - num_lags: int, numero di lag per il modello VAR
    - integration: int, grado di integrazione per differenziare le serie
    - constant: bool, includere una costante nel modello VAR
    - trend: bool, includere un termine di trend nel modello VAR
    - x: numpy.ndarray, variabili esogene
    - alpha: float, livello di significatività
    - test_type: str, tipo di test ('chi-square' o 'f')
    - cause_variables: list, variabili ipotizzate come cause
    - effect_variables: list, variabili ipotizzate come effetti
    - condition_variables: list, variabili di condizionamento
    - predictor_variables: list, ulteriori predittori esogeni
    
    Ritorna:
    - results: dict o pandas.DataFrame (statistiche del test, p-value, valori critici)
    """

    # Valida e preelabora i dati di input
    if isinstance(data, pd.DataFrame):  # Se i dati sono un DataFrame
        if effect_variables is None:  # Se non sono specificate variabili di effetto
            effect_variables = [data.columns[-1]]  # Per default usa l'ultima colonna come effetto
        y2 = data[effect_variables].to_numpy()  # Estrai la variabile di effetto come array numpy
        # Se non sono specificate variabili di causa, usa tutte le colonne tranne quelle di effetto
        cause_variables = data.columns.difference(effect_variables).tolist() if cause_variables is None else cause_variables
        y1 = data[cause_variables].to_numpy()  # Estrai le variabili di causa come array numpy
        # Variabili di condizionamento, se specificate
        y3 = data[condition_variables].to_numpy() if condition_variables else np.empty((len(data), 0))
        # Se sono specificati predittori, estraili
        if predictor_variables:
            x = data[predictor_variables].to_numpy()
    elif isinstance(data, np.ndarray):  # Se i dati sono un array numpy
        y1, y2 = data[:, :-1], data[:, -1:]  # Considera l'ultima colonna come effetto
        y3 = np.empty((data.shape[0], 0))  # Nessuna variabile di condizionamento per default
    else:
        raise ValueError("I dati devono essere un pandas DataFrame o un array numpy.")

    # Converte tutte le serie in array numpy e controlla validità
    y1, y2, y3, x = [np.asarray(arr, dtype=float) for arr in [y1, y2, y3, x or np.empty((len(y1), 0))]]
    nobs = min(map(len, [y1, y2, y3, x]))  # Calcola il numero minimo di osservazioni tra le serie

    # Tronca i dati in modo che abbiano la stessa lunghezza
    y1, y2, y3, x = [arr[-nobs:] if len(arr) > nobs else arr for arr in [y1, y2, y3, x]]
    y123 = np.hstack((y1, y2, y3))  # Combina tutte le serie in un unico array

    # Controlla per valori mancanti
    mask = ~np.isnan(y123).any(axis=1) & ~np.isnan(x).any(axis=1)  # Maschera per righe senza valori NaN
    y123, x = y123[mask], x[mask]  # Filtra le righe con dati validi
    nobs -= num_lags + integration  # Aggiorna il numero di osservazioni considerando i lag e l'integrazione

    # Costruisce il modello VAR
    var_model = VAR(y123)  # Crea il modello VAR con i dati delle serie temporali
    lag_order = num_lags + integration  # Imposta il numero totale di lag
    # Costruisce la matrice esogena con costante e/o trend se specificati
    exog = np.hstack((np.ones((len(x), 1)), x)) if constant else x
    trend_term = np.arange(1, len(y123) + 1).reshape(-1, 1) if trend else None
    exog = np.hstack((exog, trend_term)) if trend_term is not None else exog

    # Adatta il modello VAR
    var_result = var_model.fit(lag_order, trend='c' if constant else 'n')
    coeff_matrix = var_result.params  # Matrice dei coefficienti stimati
    residuals = var_result.resid  # Residui del modello
    sigma_u = var_result.sigma_u  # Matrice di covarianza dei residui

    # Funzione per calcolare le statistiche del test di Granger
    def test_statistics(var_coeffs, var_cov, df1, df2):
        """Calcola le statistiche del test di Granger."""
        stat = var_coeffs.T @ np.linalg.inv(var_cov) @ var_coeffs  # Statistica del test
        if test_type == 'chi-square':  # Test chi-quadrato
            cval = chi2.ppf(1 - alpha, df1)  # Valore critico per chi-quadrato
            pval = 1 - chi2.cdf(stat, df1)  # p-value per chi-quadrato
        else:  # Test F
            stat /= df1
            cval = f.ppf(1 - alpha, df1, df2)  # Valore critico per F
            pval = 1 - f.cdf(stat, df1, df2)  # p-value per F
        return stat, cval, pval

    # Indici per le variabili di causa ed effetto
    n_y1, n_y2, n_y3 = y1.shape[1], y2.shape[1], y3.shape[1]
    cause_indices = np.arange(n_y1)  # Indici per variabili di causa
    effect_indices = np.arange(n_y1, n_y1 + n_y2)  # Indici per variabili di effetto

    # Estrai sottocampionamenti per test delle ipotesi
    restricted_cov = var_result.cov_params().iloc[cause_indices, cause_indices]  # Covarianza delle restrizioni
    unrestricted_coeffs = coeff_matrix.iloc[effect_indices, cause_indices].values.flatten()  # Coefficienti stimati

    # Gradi di libertà
    df1 = unrestricted_coeffs.size  # Gradi di libertà del numeratore
    df2 = nobs - coeff_matrix.shape[0]  # Gradi di libertà del denominatore

    # Calcola statistiche, valori critici e p-value
    stat, cval, pval = test_statistics(unrestricted_coeffs, restricted_cov, df1, df2)

    # Prepara i risultati
    results = {
        'test_statistic': stat,  # Statistica del test
        'critical_value': cval,  # Valore critico
        'p_value': pval,  # P-value
        'hypothesis': pval <= alpha  # Risultato del test
    }

    # Se i dati sono tabulari, restituisci un DataFrame
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame([results], index=["Granger Test"])

    return results

#def granger_causality_test(y, x, max_lag):
#    """
#    Implementazione manuale del test di causalità di Granger.
#
#    Argomenti:
#        y (np.ndarray): Variabile dipendente (serie target).
#        x (np.ndarray): Variabile indipendente (serie predittore).
#        max_lag (int): Numero massimo di ritardi da includere.
#
#    Risultato:
#        dict: Contiene F-statistic, p-value e conclusione del test.
#    """
#    # Se x è un array numpy, convertilo in una Serie Pandas
#    if isinstance(x, np.ndarray):
#        x = pd.Series(x)
#    # Se y è un array numpy, convertilo in una Serie Pandas
#    if isinstance(y, np.ndarray):
#        y = pd.Series(y)
#
#    # Controllo della lunghezza delle serie
#    if len(x) != len(y):
#        raise ValueError("Input time series must have the same lenght.")
#    # Verifica che il numero massimo di lag sia compatibile con la lunghezza della serie
#    if max_lag >= len(y):
#        raise ValueError("max_lag must be less than the lenght of input time series.")
#
#    # Funzione per creare una matrice di ritardi per una serie temporale
#    def create_lagged_matrix(series, lags):
#        # Costruisce una matrice con colonne che rappresentano ritardi successivi della serie
#        return np.column_stack([series.shift(i).to_numpy() for i in range(1, lags + 1)])[lags:]
#
#    # Creazione delle matrici di ritardi per Y e X
#    lagged_y = create_lagged_matrix(y, max_lag)  # Matrice dei ritardi per Y
#    lagged_x = create_lagged_matrix(x, max_lag)  # Matrice dei ritardi per X
#    trimmed_y = y[max_lag:].to_numpy()  # Allinea Y con le matrici di ritardo
#
#    # Modello ristretto: regressione di Y sui propri ritardi
#    restricted_model_data = sm.add_constant(lagged_y)  # Aggiunge una costante per la regressione
#    restricted_model = sm.OLS(trimmed_y, restricted_model_data).fit()  # Esegue la regressione
#    ssr_restricted = restricted_model.ssr  # Somma dei residui quadrati (modello ristretto)
#
#    # Modello non ristretto: regressione di Y sui ritardi di Y e X
#    unrestricted_model_data = sm.add_constant(np.hstack([lagged_y, lagged_x]))  # Aggiunge X ai predittori
#    unrestricted_model = sm.OLS(trimmed_y, unrestricted_model_data).fit()  # Esegue la regressione
#    ssr_unrestricted = unrestricted_model.ssr  # Somma dei residui quadrati (modello non ristretto)
#
#    # Calcolo della statistica F per il test di causalità di Granger
#    n_lags = max_lag  # Gradi di libertà per il numeratore
#    df_resid = len(trimmed_y) - unrestricted_model_data.shape[1]  # Gradi di libertà per il denominatore
#    f_stat = ((ssr_restricted - ssr_unrestricted) / n_lags) / (ssr_unrestricted / df_resid)  # Formula della statistica F
#
#    # Calcolo del p-value utilizzando la distribuzione F
#    p_value = 1 - f.cdf(f_stat, n_lags, df_resid)  # Calcola il p-value dalla cdf della distribuzione F
#
#    # Step 4: Conclusion
#    conclusion = "X Granger-causes Y" if p_value < 0.05 else "No Granger causality from X to Y"
#
#    # Print Results
#    print("\nGranger Causality Test Results")
#    print(f"Restricted SSR: {ssr_restricted:.4f}")
#    print(f"Unrestricted SSR: {ssr_unrestricted:.4f}")
#    print(f"F-statistic: {f_stat:.4f}")
#    print(f"P-value: {p_value:.4f}")
#    print(f"Conclusion: {conclusion}")
#
#    # Return results as a dictionary
#    return {
#        "f_stat": f_stat,
#        "p_value": p_value,
#        "ssr_restricted": ssr_restricted,
#        "ssr_unrestricted": ssr_unrestricted,
#        "conclusion": conclusion,
#    }
