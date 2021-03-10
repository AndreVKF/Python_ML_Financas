from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error
import pandas as pd

dataset = pd.read_csv('acoes.csv')
# dataset.set_index(keys=['Date'], inplace=True)

dataset = dataset[['Date', 'BOVA']].rename(columns={'Date': 'ds', 'BOVA': 'y'})

# Modelo
modelo = Prophet()
modelo.fit(dataset)

futuro = modelo.make_future_dataframe(periods=90)
previsoes = modelo.predict(futuro)

# Gráfico das previsões
modelo.plot(previsoes, xlabel='Data', ylabel='Preço');

modelo.plot_components(previsoes);

plot_plotly(modelo, previsoes)
plot_components_plotly(modelo, previsoes)

# Avaliação do modelo
pred = modelo.make_future_dataframe(periods=0)
previsoes = modelo.predict(pred)

previsoes = previsoes['yhat'].tail(365)
mean_absolute_error(teste, previsoes)