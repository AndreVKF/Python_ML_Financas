import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import six
sys.modules['sklearn.externals.six'] = six
import mlrose

import sys
sys.path.insert(1, 'C:/Users/André Viniciu/OneDrive/Pasta/Documentos/Python_ML_Financas/Codigos')
from Alocacao_Otimizacao import *

# Variaveis p/ Fitness
# fitness_function(dtset, acoes_pesos, sem_risco, dinheiro_total)
# fitness_function(dtset, acoes_pesos, 0.000378, 5000)
fitness = mlrose.CustomFitness(fitness_function)

# Maximizacao
problema_maximizacao = mlrose.ContinuousOpt(length=6
    ,fitness_fn=fitness
    ,maximize=True
    ,min_val=0
    ,max_val=1)

# Minimizacao
problema_minimizacao = mlrose.ContinuousOpt(length=6
    ,fitness_fn=fitness
    ,maximize=False
    ,min_val=0
    ,max_val=1)

############ Teste Hill Climb ############
melhor_solucao, melhor_custo = mlrose.hill_climb(problem=problema_maximizacao, random_state=1)
melhor_solucao = melhor_solucao/melhor_solucao.sum()

############ Teste Simulated Annealing ############
melhor_solucao, melhor_custo = mlrose.simulated_annealing(problem=problema_maximizacao, random_state=1)
melhor_solucao = melhor_solucao/melhor_solucao.sum()

############ Teste Genetic Algorithm ############
problema_maximizacao_ga = mlrose.ContinuousOpt(length=6
    ,fitness_fn=fitness
    ,maximize=True
    ,min_val=0.01
    ,max_val=1)

melhor_solucao, melhor_custo = mlrose.genetic_alg(problem=problema_maximizacao, random_state=1)
melhor_solucao = melhor_solucao/melhor_solucao.sum()