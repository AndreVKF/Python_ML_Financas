import pandas as pd
import numpy as np
import plotly.express as px

dataset = pd.read_csv('acoes.csv')
dataset.set_index(keys=['Date'], inplace=True)

# Taxa de Retorno Simples
dataset/dataset.iloc[0]-1

# Taxa de Retorno Diário
dataset/dataset.shift(1)-1
(dataset/dataset.shift(1)-1)['GOL'].plot();
(dataset/dataset.shift(1)-1).describe()

# Taxa de Retorno Log
np.log(dataset/dataset.iloc[0])

# Taxa de Retorno Log Diário
np.log(dataset/dataset.shift(1))
(np.log(dataset/dataset.iloc[0])).plot();