import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys=['Date'], inplace=True)
dataset.index = pd.to_datetime(dataset.index)

# BOVA Time Series
time_series = dataset['BOVA']

# plt.plot(time_series)
# figura = px.line(title='Histórico de Preços de Ações')
# figura.add_scatter(x=time_series.index, y=time_series)

######## Decomposição da Série Temporal ########
decomposicao = seasonal_decompose(time_series, period=761)

tendencia = decomposicao.trend
sazonal = decomposicao.seasonal
aleatorio = decomposicao.resid

# plt.plot(tendencia)
# plt.plot(sazonal)
# plt.plot(aleatorio)

######## Previsão com ARIMA ########
# Time series com metade do numero de dados
modelo = auto_arima(time_series
    ,suppress_warnings=True
    ,error_action='ignore')

# Parâmetros P, Q e D
# modelo.order
previsoes = modelo.predict(n_periods=90)

######## Visualização da Previsão ########
len(time_series)

# Breakdown do datafram em treinamento/teste
n_break = len(time_series) - 365
treinamento = time_series[:n_break]
treinamento.index.min(), treinamento.index.max()

teste = time_series[n_break:]
teste.index.min(), teste.index.max()

# Treino Modelo
modelo2 = auto_arima(treinamento
    ,suppress_warnings=True
    ,error_action='ignore')

# Predict
previsoes = pd.DataFrame(modelo2.predict(n_periods=365), index=teste.index)
previsoes.rename(columns={0: 'BOVA'}, inplace=True)

# Plot Chart
plt.figure(figsize=(16, 12))
plt.plot(treinamento, label='Treinamento')
plt.plot(teste, label='Teste')
plt.plot(previsoes, label='Previsões')
plt.legend();