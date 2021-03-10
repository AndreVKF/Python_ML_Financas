import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import re

from Exploracao_Texto_Financeiros_Functions import *

########### Carregamento Base Dados ###########
base = pd.read_csv('C:/Users/André Viniciu/OneDrive/Pasta/Documentos/Python_ML_Financas/Bases de Dados/stock_data.csv')

np.unique(base['Sentiment'], return_counts=True)
sns.countplot(x=base['Sentiment'])

base.isnull().sum()

########### Pré Processamento ###########
# Processamento do texto
import spacy

pln = spacy.load("en_core_web_sm")
base['Text'] = base['Text'].apply(preprocessamento, pln=pln)

# Número de caracteres
base['Tamanho'] = base['Text'].apply(len)

# Breakdown to positivo/negativo
positivo = base.loc[base['Sentiment']==1]
negativo = base.loc[base['Sentiment']==-1]

########### Nuvem de Palavras ###########
textos_positivos = positivo['Text'].tolist()
textos_positivos_strings = ' '.join(textos_positivos)

from wordcloud import WordCloud

plt.figure(figsize=(20, 10))
plt.imshow(WordCloud().generate(textos_positivos_strings))

textos_negativos = negativo['Text'].tolist()
textos_negativos_strings = ' '.join(textos_negativos)

plt.figure(figsize=(20, 10))
plt.imshow(WordCloud().generate(textos_negativos_strings))

########### Extração de Entitdades Nomeadas ###########
documento = pln(textos_positivos_strings)

from spacy import displacy
displacy.render(documento, style='ent', jupyter=True)

empresas_positivas = []
for entidade in documento.ents:
    if entidade.label_ == 'ORG':
        print(entidade.text, entidade.label_)
        empresas_positivas.append(entidade.text)


########### Tratamento de Classes ###########
base.head()
base.drop(columns=['Tamanho'], inplace=True)

from sklearn.model_selection import train_test_split
base_treinamento, base_teste = train_test_split(base, test_size=0.3)

base_treinamento_final = []
for index, row in base.iterrows():
    if row['Sentiment']==1:
        dic = [{'POSITIVO': True, 'NEGATIVO': False}]
    elif row['Sentiment']==-1:
        dic = [{'POSITIVO': False, 'NEGATIVO': True}]

    base_treinamento_final.append([row['Text'], dic.copy()])

########### Criação do Classificador ###########
modelo = spacy.blank("en")
categorias = modelo.create_pipe("textcat")
categorias.add_label('POSITIVO')
categorias.add_label('NEGATIVO')
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(5):
    random.shuffle(base_treinamento_final)
    erros = {}
    for batch in spacy.util.minibatch(base_treinamento_final, 512):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses=erros)
        historico.append(erros)
    if epoca % 1 == 0:
        print(erros)

historico_erro = []
for i in historico:
    historico_erro.append(i.get('textcat'))
historico_erro = np.array(historico_erro)

plt.plot(historico_erro)
plt.title('Progressão do erro')
plt.xlabel('Batches')
plt.ylabel('Erro')

modelo.to_disk('modelo')