
## Quantile regression

# 
# This example page shows how to use ``statsmodels``' ``QuantReg`` class to replicate parts of the analysis published in 
# 
# * Koenker, Roger and Kevin F. Hallock. "Quantile Regressioin". Journal of Economic Perspectives, Volume 15, Number 4, Fall 2001, Pages 143–156
# 
# We are interested in the relationship between income and expenditures on food for a sample of working class Belgian households in 1857 (the Engel data). 
# 
# ## Setup
# 
# We first need to load some modules and to retrieve the data. Conveniently, the Engel dataset is shipped with ``statsmodels``.

from __future__ import print_function
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg

data = sm.datasets.engel.load_pandas().data
data.head()


# ## Least Absolute Deviation
# 
# The LAD model is a special case of quantile regression where q=0.5

mod = smf.quantreg('foodexp ~ income', data)
res = mod.fit(q=.5)
print(res.summary())


# ## Visualizing the results
# 
# We estimate the quantile regression model for many quantiles between .05 and .95, and compare best fit line from each of these models to Ordinary Least Squares results. 

# ### Prepare data for plotting
# 
# For convenience, we place the quantile regression results in a Pandas DataFrame, and the OLS results in a dictionary.

quantiles = np.arange(.05, .96, .1)


def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['income']] + res.conf_int().loc['income'].tolist()


models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

print(models)


# ### First plot
# 
# This plot compares best fit lines for 10 quantile regression models to the least squares fit. As Koenker and Hallock (2001) point out, we see that:
# 
# 1. Food expenditure increases with income
# 2. The *dispersion* of food expenditure increases with income
# 3. The least squares estimates fit low income observations quite poorly (i.e. the OLS line passes over most low income households)

x = np.arange(data.income.min(), data.income.max(), 50)
get_y = lambda a, b: a + b * x

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    plt.plot(x, y, linestyle='dotted', color='grey')
    

plt.scatter(data.income, data.foodexp, alpha=.2)
plt.xlim((240, 3000))
plt.ylim((240, 2000))
plt.legend()
plt.xlabel('Income')
plt.ylabel('Food expenditure')
plt.show()
