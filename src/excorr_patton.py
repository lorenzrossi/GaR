import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
from scipy.stats import chi2

def pearson_corr(x, y):
    return np.cov(x, y, ddof=1)[0, 1] / (np.std(x, ddof=1) * np.std(y, ddof=1))

def exceedence_correl(X, Y, qc=[0.5], indic=1):
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    T = len(X)
    qc = sorted(set(qc))
    k = len(qc)

    out1 = []
    rhodiff = []
    xi = []

    if indic == 1:  # Quantile-based
        valid_qc = []

        for q in qc:
            x_low = np.quantile(X, 1 - q)
            y_low = np.quantile(Y, 1 - q)
            x_high = np.quantile(X, q)
            y_high = np.quantile(Y, q)

            low_ix = np.where((X <= x_low) & (Y <= y_low))[0]
            high_ix = np.where((X >= x_high) & (Y >= y_high))[0]

            if len(low_ix) >= 3 and len(high_ix) >= 3:
                valid_qc.append(q)

        qc1 = sorted(set(valid_qc + [1 - q for q in valid_qc]))
        q = qc1[len(qc1)//2:]
        k = len(q)
        out1 = np.zeros(2 * k)
        xi = np.zeros((T, k))
        rhodiff = np.zeros(k)

        for j in range(k):
            qj = q[j]
            x_low = np.quantile(X, 1 - qj)
            y_low = np.quantile(Y, 1 - qj)
            x_high = np.quantile(X, qj)
            y_high = np.quantile(Y, qj)

            low_ix = np.where((X <= x_low) & (Y <= y_low))[0]
            high_ix = np.where((X >= x_high) & (Y >= y_high))[0]

            rho_low = pearson_corr(X[low_ix], Y[low_ix])
            rho_high = pearson_corr(X[high_ix], Y[high_ix])

            out1[j] = rho_low
            out1[2 * k - 1 - j] = rho_high
            rhodiff[j] = rho_high - rho_low

            x1p = (X - np.mean(X[high_ix])) / np.std(X[high_ix], ddof=1)
            x2p = (Y - np.mean(Y[high_ix])) / np.std(Y[high_ix], ddof=1)
            x1m = (X - np.mean(X[low_ix])) / np.std(X[low_ix], ddof=1)
            x2m = (Y - np.mean(Y[low_ix])) / np.std(Y[low_ix], ddof=1)

            term_plus = (x1p * x2p - rho_high) * ((X >= x_high) & (Y >= y_high))
            term_minus = (x1m * x2m - rho_low) * ((X <= x_low) & (Y <= y_low))

            xi[:, j] = (T / len(high_ix)) * term_plus - (T / len(low_ix)) * term_minus

    else:  # Standard deviation-based
        X = (X - np.mean(X)) / np.std(X, ddof=1)
        Y = (Y - np.mean(Y)) / np.std(Y, ddof=1)
        valid_qc = []

        for c in qc:
            low_ix = np.where((X <= -c) & (Y <= -c))[0]
            high_ix = np.where((X >= c) & (Y >= c))[0]
            if len(low_ix) >= 3 and len(high_ix) >= 3:
                valid_qc.append(c)

        qc1 = sorted(set(valid_qc + [-c for c in valid_qc]))
        cvec = qc1[len(qc1)//2:]
        k = len(cvec)
        out1 = np.zeros(2 * k)
        xi = np.zeros((T, k))
        rhodiff = np.zeros(k)

        for j in range(k):
            cj = cvec[j]
            low_ix = np.where((X <= -cj) & (Y <= -cj))[0]
            high_ix = np.where((X >= cj) & (Y >= cj))[0]

            rho_low = pearson_corr(X[low_ix], Y[low_ix])
            rho_high = pearson_corr(X[high_ix], Y[high_ix])

            out1[j] = rho_low
            out1[2 * k - 1 - j] = rho_high
            rhodiff[j] = rho_high - rho_low

            x1p = (X - np.mean(X[high_ix])) / np.std(X[high_ix], ddof=1)
            x2p = (Y - np.mean(Y[high_ix])) / np.std(Y[high_ix], ddof=1)
            x1m = (X - np.mean(X[low_ix])) / np.std(X[low_ix], ddof=1)
            x2m = (Y - np.mean(Y[low_ix])) / np.std(Y[low_ix], ddof=1)

            term_plus = (x1p * x2p - rho_high) * ((X >= cj) & (Y >= cj))
            term_minus = (x1m * x2m - rho_low) * ((X <= -cj) & (Y <= -cj))

            xi[:, j] = (T / len(high_ix)) * term_plus - (T / len(low_ix)) * term_minus

    # Newey-West covariance estimation using statsmodels
    model = OLS(xi, np.ones((T, 1))).fit()
    omega_hat = cov_hac(model)
    teststat = T * rhodiff @ np.linalg.inv(omega_hat) @ rhodiff
    pval = 1 - chi2.cdf(teststat, df=k)

    return {
        'correlations': out1,
        'thresholds': qc1 if indic == 1 else qc1,
        'teststat': teststat,
        'pval': pval
    }