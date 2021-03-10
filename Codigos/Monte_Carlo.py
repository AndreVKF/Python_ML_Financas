# Preço de hoje = preço de ontem * e^r
# Brownian Move
#   -- Drift: Direção que as taxas de retorno tiveram no passado
#   -- Volatilidade: Variável aleatória

# Drift = Mi - 1/2*sigma^2
# Volatility = sigma * Z[Rand(0;1)]
# r = (Mi - 1/2*sigma^2) + sigma*Z[Rand(0;1)]

# St = St-1 * e ^ r

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy import  stats

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys=['Date'], inplace=True)

# Gráfico historico das acoes
figura = px.line(title = 'Histórico do preço das ações')
for i in dataset.columns:
    figura.add_scatter(x=dataset.index, y=dataset[i], name=i)
figura.show()

# Normalização dos precos
dataset_normalizado = dataset/dataset.iloc[0]

dataset_taxa_retorno = np.log(dataset/dataset.shift(1))
dataset_taxa_retorno.fillna(0, inplace=True)

########## Cálculo do Drift ##########
media = dataset_taxa_retorno['BOVA'].mean()
variancia = dataset_taxa_retorno['BOVA'].var()

drift = media - (0.5 * variancia)

########## Cálculo dos Retornos Diários com Base no Drift ##########
dias_frente = 50
simulacoes = 10

desvio_padrao = dataset_taxa_retorno['BOVA'].std()

Z = stats.norm.ppf(np.random.rand(dias_frente, simulacoes))
# Z.shape
# sns.histplot(Z)

# e^r = drift + sigma*Z[Rand(0;1)]
retornos_diarios = np.exp(drift + desvio_padrao * Z)
retornos_diarios.shape

########## Previsões de Preços Futuros ##########
previsoes = np.zeros_like(retornos_diarios)

# Inicialização com o ultimo valor
previsoes[0] = dataset['BOVA'].iloc[-1]

# Loop para gerar preços
for dia in range(1, dias_frente):
    previsoes[dia] = previsoes[dia-1] * retornos_diarios[dia]

# Previsão de preços de ações - simulações
figura = px.line(title = 'Previsões do preço das ações - simulações')
for i in range(len(previsoes.T)):
    figura.add_scatter(y = previsoes.T[i], name=i)
figura.show()

########## Previsões X Preços Reais ##########
