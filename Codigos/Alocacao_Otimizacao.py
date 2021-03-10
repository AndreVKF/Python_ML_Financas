import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Functions
def alocacao_ativos(dataset, dinheiro_total, seed=0, log_return=False):
    dataset = dataset.copy()
    acoes = dataset.columns

    # Valores randomicos
    if seed != 0:
        np.random.seed(seed)
    pesos = np.random.random(len(dataset.columns))
    # Normalizacao dos pesos
    pesos = pesos/pesos.sum()
    # print(pesos)
    # Normalizacao do dataset
    dataset = dataset/dataset.iloc[0]

    for i, acao in enumerate(dataset.columns):
        dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total

    dataset['Soma_Valor'] = dataset.sum(axis=1)
    # Flag to calc daily retrun
    if log_return:
        dataset['Taxa_Retorno_Diario'] = np.log(dataset['Soma_Valor']/dataset['Soma_Valor'].shift(1))
    else:
        dataset['Taxa_Retorno_Diario'] = dataset['Soma_Valor']/dataset['Soma_Valor'].shift(1)-1

    dataset['Taxa_Retorno_Acumulado'] = dataset['Soma_Valor']/dataset['Soma_Valor'].iloc[0]-1
    # print(dataset.columns, pesos * 100)
    acoes_pesos = pd.DataFrame(data={'Acoes': acoes, 'Pesos': pesos * 100})

    return dataset, dataset.index, acoes_pesos, dataset.iloc[-1]['Soma_Valor']

def repeticoes_portfolios_markowitz(dataset, dinheiro_total, taxa_sem_risco, n_repeticoes, return_dtlists=False):
    
    melhor_sharpe_ratio = -999999
    melhores_pesos = pd.DataFrame()
    melhor_portfolio = pd.DataFrame()

    ls_retorno = []
    ls_volatilidade = []
    ls_sharpe = []

    for i in range(n_repeticoes):
        dtset, datas, acoes_pesos, soma_valor = alocacao_ativos(dataset, dinheiro_total)

        # Retorno por acao
        retorno_acao = np.log(dtset[acoes_pesos['Acoes']]/dtset[acoes_pesos['Acoes']].shift(1)).dropna()
        matriz_cov = retorno_acao.cov()

        retorno_esperado = np.sum(dtset['Taxa_Retorno_Diario'].mean()*acoes_pesos['Pesos'].values)*252
        volatilidade_esperada = np.sqrt(np.dot(acoes_pesos['Pesos'], np.dot(matriz_cov * 252, acoes_pesos['Pesos'])))
        sharpe_ratio = (retorno_esperado - taxa_sem_risco)/volatilidade_esperada
        # volatilidade_esperada = dtset['Taxa_Retorno_Diario'].std() * np.sqrt(252)
        # sharpe_ratio = (dtset['Taxa_Retorno_Diario'].mean() - taxa_sem_risco) / dtset['Taxa_Retorno_Diario'].std() * np.sqrt(252)

        ls_retorno.append(retorno_esperado)
        ls_volatilidade.append(volatilidade_esperada)
        ls_sharpe.append(sharpe_ratio)

        if sharpe_ratio > melhor_sharpe_ratio:
            melhor_sharpe_ratio = sharpe_ratio
            melhores_pesos = acoes_pesos
            melhor_portfolio = dtset

            melhor_retorno = retorno_esperado
            melhor_volatilidade = volatilidade_esperada

    if return_dtlists:
        return melhor_sharpe_ratio, melhores_pesos, ls_retorno, ls_volatilidade, ls_sharpe, melhor_retorno, melhor_volatilidade
    else:
        return melhor_sharpe_ratio, melhores_pesos
    

def fitness_function(solucao, dinheiro_total=100, sem_risco=0.000378):
    acoes_pesos = create_dataset_pesos(solucao)
    dataset = create_dateset_precos()

    # Normalização
    dataset = dataset/dataset.iloc[0]

    # Atribuição do montante investido a cada acao
    for i, acao in enumerate(acoes_pesos['Acoes']):
        dataset[acao] = dataset[acao] * acoes_pesos.loc[(acoes_pesos['Acoes']==acao)]['Pesos'].iloc[0]/100 * dinheiro_total

    dataset['Soma_Valor'] = dataset.sum(axis=1)
    dataset['Taxa_Retorno_Diaria'] = (dataset['Soma_Valor']/dataset['Soma_Valor'].shift(1) - 1) * 100

    sharpe_ratio = (dataset['Taxa_Retorno_Diaria'].mean()  - sem_risco) / dataset['Taxa_Retorno_Diaria'].std() * np.sqrt(252)

    return sharpe_ratio

def visualiza_alocacao(acoes, pesos):
    alocacoes = pd.DataFrame(data={
        'Acoes': acoes
        ,'Pesos': pesos
    })

    return alocacoes

def create_dataset_pesos(pesos):
    dataset = pd.read_csv('acoes.csv')
    dataset.set_index(keys='Date', inplace=True)

    return pd.DataFrame(
        data={'Acoes': dataset.columns
        ,'Pesos': pesos}
    )

def create_dateset_precos():
    dataset = pd.read_csv('acoes.csv')
    dataset.set_index(keys='Date', inplace=True)

    return dataset

if __name__ == '__main__':
    dataset = pd.read_csv('acoes.csv')
    dataset.set_index(keys='Date', inplace=True)

    dtset, datas, acoes_pesos, soma_valor = alocacao_ativos(dataset, 5000)
    solucao = [0.2, 0.1, 0.1, 0.2, 0.1, 0.3]
    create_dataset_pesos(solucao)

    fitness_function(solucao)
    


    # Otimização de Portfolio
    # sharpe_ratio, pesos, ls_retornos, ls_volatilidade, ls_sharpe, melhor_retorno, melhor_volatilidade = repeticoes_portfolios_markowitz(dataset, 5000, 10, 1000, return_dtlists=True)
    # plt.figure(figsize=(16, 12))
    # plt.scatter(ls_volatilidade, ls_retornos, c=ls_sharpe)
    # plt.colorbar(label='Sharpe Ratio')
    # plt.xlabel('Volatilidade')
    # plt.ylabel('Retorno')
    # plt.scatter(melhor_volatilidade, melhor_retorno, c='red', s=100);

    # # Gráfico de retorno diário
    # figura = px.line(x=dtset.index, y=dtset['Taxa_Retorno_Diario'], title='Retorno diário do portfólio')
    # figura.show()

    # # Evolução por ação
    # figura = px.line(title = 'Evolução de Patrimônio Por Ação')
    # for i in dtset[acoes_pesos['Acoes'].values].columns:
    #     figura.add_scatter(x=dtset.index, y=dtset[i], name=i)
    # figura.show()

    # # Evolução total do patrimonio
    # figura = px.line(x = dtset.index, y = dtset['Soma_Valor'])
    # figura.show()