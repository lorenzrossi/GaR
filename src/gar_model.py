import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import t
from scipy.integrate import quad

def skewed_t_pdf(y, mu, sigma, alpha, nu):
    """PDF of a skewed t-distribution."""
    t_pdf = t.pdf((y - mu) / sigma, df=nu)
    cdf_component = t.cdf(alpha * (y - mu) / sigma * np.sqrt((nu + 1) / (nu + (y - mu)**2 / sigma**2)), df=nu + 1)
    return 2 * t_pdf * cdf_component / sigma

# Define conditional density function f(y_t | y_{t-1}, x)
def f_t(y, mu, sigma, alpha, nu):
    return skewed_t_pdf(y, mu, sigma, alpha, nu)


# Unconditional density (e.g., normal distribution for simplicity)
def g_t(y, mu_g, sigma_g):
    return (1 / (np.sqrt(2 * np.pi) * sigma_g)) * np.exp(-((y - mu_g)**2) / (2 * sigma_g**2))

# Entropy calculation
def downside_entropy(mu, sigma, alpha, nu, mu_g, sigma_g, median):
    """Calculate downside entropy."""
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (np.log(g_t(y, mu_g, sigma_g)) - np.log(f_t(y, mu, sigma, alpha, nu)))
    entropy, _ = quad(integrand, -np.inf, median)
    return -entropy

def upside_entropy(mu, sigma, alpha, nu, mu_g, sigma_g, median):
    """Calculate upside entropy."""
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (np.log(g_t(y, mu_g, sigma_g)) - np.log(f_t(y, mu, sigma, alpha, nu)))
    entropy, _ = quad(integrand, median, np.inf)
    return -entropy
