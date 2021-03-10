import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys=['Date'], inplace=True)

dataset.mean()
dataset.var()
dataset.std()

(dataset/dataset.shift(1)-1).dropna().cov()
(dataset/dataset.shift(1)-1).dropna().corr()

sns.heatmap(data=(dataset/dataset.shift(1)-1).dropna().corr(), annot=True)

# Risco
pesos1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
pesos1.sum()

# Retornos
taxas_retorno = (dataset/dataset.shift(1)-1).dropna()

# Anualizando
taxas_retorno.cov() * 246

# Periodo em cima do peso do portfolio
np.dot(taxas_retorno.cov() * 246, pesos1)

variancia_portfolio1 = np.dot(pesos1, np.dot(taxas_retorno.cov() * 246, pesos1))
volatilidade_portfolio = math.sqrt(variancia_portfolio1)