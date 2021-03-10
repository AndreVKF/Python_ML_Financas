import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from Classificacao_Empresas_Functions import *

dataset = pd.read_excel('BD Completo.xlsx')

######## Clear Data ########
# Clear NaN columns
sns.heatmap(dataset.isnull())

check_null = pd.DataFrame(data=dataset.isnull().sum(), columns=['Qtde'])
dataset.drop(labels=check_null.loc[check_null['Qtde']>200].index.to_list(), axis=1, inplace=True)

# Fill with mean
dataset.fillna(dataset.mean(), inplace=True)

# Drop na columns
dataset.dropna(inplace=True)

######## Análise Categórica/Visualização ########
sns.countplot(x=dataset['Situação'])
np.unique(dataset['Segmento'], return_counts=True)
dataset['Segmento'] = dataset['Segmento'].apply(corrige_segmento)

np.unique(dataset['Categoria'], return_counts=True)
dataset['Categoria'] = dataset['Categoria'].apply(corrige_categoria)
plt.figure(figsize=(15, 15))
sns.countplot(x=dataset['Categoria']);

dataset.describe()

figura = plt.figure(figsize=(15,20))
eixo = figura.gca()
dataset.hist(ax=eixo)

######## Correlação Entre Atributos ########
plt.figure(figsize=(60, 50))
sns.heatmap(dataset.corr(), annot=True, cbar=False);

dataset.corr()

# Drop similar attributes
dataset.drop(labels=['Rec. Liquida', 'Caixa'], axis=1, inplace=True)
dataset.drop(labels=['Divida bruta', 'LPA', 'Caixa.1'], axis=1, inplace=True)
dataset.drop(labels=['At. Circulante', 'Liq. Corrente'], axis=1, inplace=True)

######## Variáveis Dummy ########
y = dataset['Situação'].values
empresa = dataset['Empresa']

X_cat = dataset[['Segmento', 'Categoria']]
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()

X_cat = pd.DataFrame(X_cat)

empresas = dataset_original['Empresa']

# Substituição
dataset_original = dataset.copy()

dataset.drop(['Segmento', 'Categoria', 'Situação', 'Empresa', 'Majoritar.'], axis=1, inplace=True)
dataset.index = X_cat.index

dataset = pd.concat([dataset, X_cat], axis=1)

######## Normalização ########
scaler = MinMaxScaler()
dataset_normalizado = scaler.fit_transform(dataset)

x = dataset_normalizado.copy()

######## Agrupamentos ########

######## K-Means ########
from sklearn.cluster import KMeans

# Obtenção do Número de Clusters
wcss = [] # within cluster sum of squares {qnto menor melhor}
faixas = range(1, 20)

for i in faixas:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# plt.plot(wcss, 'bx--')
# plt.xlabel('Clusters')
# plt.ylabel('WCSS');

# Agrupamento com k-means
kmeans = KMeans(n_clusters=7)
kmeans.fit(x)

labels = kmeans.labels_
np.unique(labels, return_counts=True)

cluster_centers = kmeans.cluster_centers_

# Análise de Agrupamentos
cluster_centers_inverse = scaler.inverse_transform(kmeans.cluster_centers_)
centroides = pd.DataFrame(data = cluster_centers_inverse, columns = dataset.columns)

dataset_cluster = pd.concat([dataset_original, pd.DataFrame({'cluster': labels})], axis=1)

categoria_cluster = dataset_cluster.groupby(['Categoria', 'cluster'])['cluster'].count()
situacao_cluster = dataset_cluster.groupby(['Situação', 'cluster'])['cluster'].count()
segmento_cluster = dataset_cluster.groupby(['Segmento', 'cluster'])['cluster'].count()

# Visualização 
# Redução PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
componentes = pca.fit_transform(x)

pca_df = pd.DataFrame(data = componentes, columns=['pca1', 'pca2'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df, palette=['red', 'green', 'blue', 'yellow', 'black', 'gray', 'magenta'])