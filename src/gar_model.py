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

#i didn't understand why they talk about the skewed t distribution and then they do an example with this conditional distribution 

# Unconditional density (e.g., normal distribution for simplicity)
def g_t(y, mu_g, sigma_g):
    return (1 / (np.sqrt(2 * np.pi) * sigma_g)) * np.exp(-((y - mu_g)**2) / (2 * sigma_g**2))

#(np.sqrt(1 - phi_g**2) * (1 / np.sqrt(2 * np.pi * sigma_g**2))) * np.exp(-((1 - phi_g**2) / (2 * sigma_g**2)) * ((y - (mu_g / (1 - phi_g)))**2))

# Entropy calculation
def downside_entropy(mu, sigma, alpha, nu, mu_g, sigma_g, median):
    """Calculate downside entropy."""
    epsilon = 1e-10  # Small value to avoid log(0)
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (
        np.log(g_t(y, mu_g, sigma_g) + epsilon) - np.log(f_t(y, mu, sigma, alpha, nu) + epsilon)
    )
    entropy, _ = quad(integrand, -np.inf, median, limit=1000)
    return -entropy

def upside_entropy(mu, sigma, alpha, nu, mu_g, sigma_g, median):
    """Calculate upside entropy."""
    epsilon = 1e-10  # Small value to avoid log(0)
    integrand = lambda y: f_t(y, mu, sigma, alpha, nu) * (
        np.log(g_t(y, mu_g, sigma_g) + epsilon) - np.log(f_t(y, mu, sigma, alpha, nu) + epsilon)
    )
    entropy, _ = quad(integrand, median, np.inf, limit=1000)
    return -entropy


class EntropyModel:
    def __init__(self, data, target_col, covariates, lag_col):
        """
        Initialize the model with data and specify target and covariates.
        :param data: DataFrame containing the time series.
        :param target_col: Name of the target column (y_t).
        :param covariates: List of covariate column names (including x).
        :param lag_col: Name of the lagged column (y_t-1).
        """
        self.data = data
        self.target_col = target_col
        self.covariates = covariates
        self.lag_col = lag_col
        self.quantile_models = {}
        self.predicted_quantiles = None

    def fit_quantile_regression(self, quantiles=[0.05, 0.5, 0.95]):
        """
        Fit quantile regression models for specified quantiles.
        :param quantiles: List of quantiles to fit.
        """
        for q in quantiles:
            model = QuantReg(self.data[self.target_col], self.data[self.covariates])
            result = model.fit(q=q)
            self.quantile_models[q] = result
            print(f"Quantile {q} coefficients:\n{result.params}")
        
        # Predict and store quantiles
        self.data['q_05'] = self.quantile_models[0.05].predict(self.data[self.covariates])
        self.data['q_50'] = self.quantile_models[0.5].predict(self.data[self.covariates])
        self.data['q_95'] = self.quantile_models[0.95].predict(self.data[self.covariates])
        self.predicted_quantiles = self.data[['q_05', 'q_50', 'q_95']]
    
    @staticmethod
    def skewed_t_pdf(y, mu, sigma, alpha, nu):
        """
        PDF of a skewed t-distribution.
        :param y: Value at which to evaluate the PDF.
        :param mu: Location parameter.
        :param sigma: Scale parameter.
        :param alpha: Skewness parameter.
        :param nu: Degrees of freedom.
        """
        t_pdf = t.pdf((y - mu) / sigma, df=nu)
        cdf_component = t.cdf(alpha * (y - mu) / sigma * np.sqrt((nu + 1) / (nu + (y - mu)**2 / sigma**2)), df=nu+1)
        return 2 * t_pdf * cdf_component / sigma

    def compute_entropy(self, row, alpha=0, nu=20):
        """
        Compute downside and upside entropy for a single row.
        :param row: Row of the DataFrame with quantile predictions.
        :param alpha: Skewness parameter for skewed t-distribution.
        :param nu: Degrees of freedom for skewed t-distribution.
        """
        mu = row['q_50']  # Median as location
        sigma = (row['q_95'] - row['q_05']) / 2  # Scale based on interquantile range
        median = mu  # Median of the conditional distribution
        
        # Unconditional density parameters
        mu_g = self.data[self.target_col].mean()
        sigma_g = self.data[self.target_col].std()
        
        # Define conditional and unconditional densities
        f_t = lambda y: self.skewed_t_pdf(y, mu, sigma, alpha, nu)
        g_t = lambda y: (1 / (np.sqrt(2 * np.pi) * sigma_g)) * np.exp(-((y - mu_g)**2) / (2 * sigma_g**2))
        
        # Calculate downside entropy
        def downside_entropy():
            integrand = lambda y: f_t(y) * (np.log(g_t(y)) - np.log(f_t(y)))
            entropy, _ = quad(integrand, -np.inf, median)
            return -entropy
        
        # Calculate upside entropy
        def upside_entropy():
            integrand = lambda y: f_t(y) * (np.log(g_t(y)) - np.log(f_t(y)))
            entropy, _ = quad(integrand, median, np.inf)
            return -entropy

        return downside_entropy(), upside_entropy()

    def compute_entropies_for_all(self):
        """
        Compute downside and upside entropy for all rows in the data.
        """
        entropies = self.data.apply(lambda row: self.compute_entropy(row), axis=1)
        self.data['downside_entropy'], self.data['upside_entropy'] = zip(*entropies)



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
