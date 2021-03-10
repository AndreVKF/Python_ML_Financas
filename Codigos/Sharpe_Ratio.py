import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys

sys.path.insert(1, 'C:/Users/André Viniciu/OneDrive/Pasta/Documentos/Python_ML_Financas/Codigos')

from Alocacao_Otimizacao import alocacao_ativos

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys='Date', inplace=True)

dtset, datas, acoes_pesos, soma_valor = alocacao_ativos(dataset, 5000)

# Sharpe Ratio s/ Risk Free Rate
dtset['Taxa_Retorno_Diario'].mean()/dtset['Taxa_Retorno_Diario'].std() * np.sqrt(252)