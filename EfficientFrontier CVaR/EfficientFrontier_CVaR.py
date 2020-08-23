import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import negative_cvar
from pypfopt.risk_models import CovarianceShrinkage

##Importar série de dados
df = pd.read_excel(r'C:\Users\Jhona\OneDrive\Área de Trabalho\PRBR11.xlsx', index_col='Data')

##Obter os retornos dos ativos
returns = pd.DataFrame()
for i in df:
    returns[i] = df[i].pct_change().dropna()

##Criar a matrix de covariância eficiente
covMatrix = CovarianceShrinkage(df)
e_cov = covMatrix.ledoit_wolf()

##Fronteira
ef = EfficientFrontier(None, e_cov)

##Encontrando o CVaR 95% minimizando 
optimal_weights = ef.custom_objective(negative_cvar, returns)
print(optimal_weights)