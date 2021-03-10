# Descreve as relações entre o retorno esperado e o risco, 
# comparando o portfolio com o mercado

# Ri = Rf + Bi * (Rm - Rf)
# Rm = Retorno esperado do portfolio
# Rf = Retorno sem riscos
# Beta = Comparacao entre a carteira e o mercado

import pandas as pd
import numpy as np
import plotly.express as px

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys=['Date'], inplace=True)

# Normalizando
dataset_normalizado = dataset/dataset.iloc[0]

# Retorno diário
dataset_taxa_retorno = dataset/dataset.shift(1) - 1
dataset_taxa_retorno.loc['2015-01-02'] = 0.00

# Retorno Médio Anualizado
dataset_taxa_retorno.mean() * 252

# Beta Por Regressão Linear
beta, alpha = np.polyfit(
    x=dataset_taxa_retorno['BOVA']
    ,y=dataset_taxa_retorno['MGLU']
    ,deg=1)

figura = px.scatter(dataset_taxa_retorno, x = 'BOVA', y = 'MGLU', title = 'BOVA x MGLU')
figura.add_scatter(x=dataset_taxa_retorno['BOVA'], y=beta * dataset_taxa_retorno['BOVA'] + alpha)
figura.show()

# Beta por Matriz de Covariância/Variancia
matriz_covariancia = dataset_taxa_retorno[['MGLU', 'BOVA']].cov() * 252
cov_mglu_bova = matriz_covariancia.loc['MGLU', 'BOVA']

variancia_bova = dataset_taxa_retorno['BOVA'].var() * 252

beta_mglu = cov_mglu_bova/variancia_bova

# Calculo do CAPM
Rm = dataset_taxa_retorno['BOVA'].mean() * 252
Rf = 0.7

CAPM_mglu = Rf + (beta * (Rm - Rf))

# Calculo para todas as acoes
acoes = dataset_taxa_retorno.columns.to_list()
acoes.remove('BOVA')

betas = []
alphas = []
capms = []

for acao in acoes:
    beta, alpha = np.polyfit(x=dataset_taxa_retorno['BOVA'], y=dataset_taxa_retorno[acao], deg=1)
    betas.append(beta)
    alphas.append(alpha)

    capms.append(Rf + (beta * (Rm - Rf)))
