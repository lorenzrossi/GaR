import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import t
from scipy.integrate import quad

# Parameters
#np.random.seed(42)
#phi_g = [0.5, 0.3, -0.2]  # Coefficients for covariates
#mu_g = 2                  # Long-term mean of GDP growth
#sigma_epsilon_plus = 1    # Variance of positive shocks
#sigma_epsilon_minus = 2   # Variance of negative shocks
#p_plus = 0.5              # Probability of positive shocks

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
    epsilon = 1e-10  # Small value to avoid log(0)
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (
        np.log(g_t(y, mu_g, sigma_g) + epsilon) - np.log(f_t(y, mu, sigma, alpha, nu) + epsilon)
    )
    entropy, _ = quad(integrand, -np.inf, median, limit=100)
    return -entropy

def upside_entropy(mu, sigma, alpha, nu, mu_g, sigma_g, median):
    """Calculate upside entropy."""
    epsilon = 1e-10  # Small value to avoid log(0)
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (
        np.log(g_t(y, mu_g, sigma_g) + epsilon) - np.log(f_t(y, mu, sigma, alpha, nu) + epsilon)
    )
    entropy, _ = quad(integrand, median, np.inf, limit=100)
    return -entropy



############ NON NORMAL / MIXTURE OF NORMAL ######################

# Conditional density f_t(y)
#def f_t(y, covariates):
#    mu_t = mu_g + np.dot(phi_g, covariates)  # Linear combination of covariates
#    # Positive shocks
#    f_plus = p_plus * (1 / np.sqrt(2 * np.pi * sigma_epsilon_plus**2)) * np.exp(
#        -((y - mu_t)**2) / (2 * sigma_epsilon_plus**2)
#    ) * (y >= mu_t)
#    # Negative shocks
#    f_minus = (1 - p_plus) * (1 / np.sqrt(2 * np.pi * sigma_epsilon_minus**2)) * np.exp(
#        -((y - mu_t)**2) / (2 * sigma_epsilon_minus**2)
#    ) * (y < mu_t)
#    return f_plus + f_minus
#
## Unconditional density g(y)
#def g_t(y):
#    var_unconditional = sigma_epsilon_plus**2 * p_plus + sigma_epsilon_minus**2 * (1 - p_plus)
#    mean_unconditional = mu_g / (1 - np.sum(phi_g))  # Adjusted for multiple covariates
#    return (1 / np.sqrt(2 * np.pi * var_unconditional)) * np.exp(
#        -((y - mean_unconditional)**2) / (2 * var_unconditional)
#    )
#
## Calculate downside entropy
#def downside_entropy(covariates):
#    mu_t = mu_g + np.dot(phi_g, covariates)
#    def integrand(y):
#        return f_t(y, covariates) * (np.log(g_t(y)) - np.log(f_t(y, covariates)))
#    entropy, _ = quad(integrand, -np.inf, mu_t)
#    return -entropy
#
## Calculate upside entropy
#def upside_entropy(covariates):
#    mu_t = mu_g + np.dot(phi_g, covariates)
#    def integrand(y):
#        return f_t(y, covariates) * (np.log(g_t(y)) - np.log(f_t(y, covariates)))
#    entropy, _ = quad(integrand, mu_t, np.inf)
#    return -entropy
