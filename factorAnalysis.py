# based on https://www.datacamp.com/community/tutorials/introduction-factor-analysis
# other good link : https://github.com/James-Thorson/spatial_factor_analysis/blob/master/R/Sim_Fn.R

import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance


# raw data
url = 'https://raw.githubusercontent.com/rkn2/factorAnalysisExample/master/bfi%20(1).csv'
df = pd.read_csv(url)
df.columns
unnecessaryColumns = ['gender', 'age', 'education']
df.drop(unnecessaryColumns, axis=1, inplace=True)
df.dropna(inplace=True)
numVars = df.shape[1] - len(unnecessaryColumns)

# regular fa
fa = FactorAnalyzer()
numFactors = 5
fa.analyze(df, numFactors, rotation=None)
L = np.array(fa.loadings)
headings = list( fa.loadings.transpose().keys() )
factor_threshold = 0.25
for i, factor in enumerate(L.transpose()):
  descending = np.argsort(np.abs(factor))[::-1]
  contributions = [(np.round(factor[x],2),headings[x]) for x in descending if np.abs(factor[x])>factor_threshold]
  print('Factor %d:'%(i+1),contributions)


# pre computed correlation matrix fa
fa = FactorAnalyzer()
numFactors = 5
x = (df-df.mean(0))/df.std(0)
corr = np.cov(x, rowvar=False, ddof=0)

# for spatial correlation, define spatial weights matrix
W = make_spatial_weights(xy)


self = fa
start = [0.5 for _ in range(corr.shape[0])]
objective = self._fit_uls_objective
res = minimize(objective,
               start,
               method='L-BFGS-B',
               bounds=None,
               options={'maxiter': 1000},
               args=(corr, numFactors))

# get factor column names
columns = ['Factor{}'.format(i) for i in range(1, numFactors + 1)]
loadings = self._normalize_wls(res.x, corr, numFactors)

