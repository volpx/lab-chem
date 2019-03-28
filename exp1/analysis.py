# %%
import pandas as pd
import numpy as np
from os import path
from np_fit_sine_poly import fit_sine_poly
from matplotlib import pyplot as plt, use

# use('Qt5Agg')

DATA = 'exp1/data'

'''
acqua distillata V = 80 ml
NaCl soluzione 0.25 M (corrispondente a 14.6 g/litro) 
CaCl2 soluzione 0.25 M (corrispondente a 27.75 g/litro) 
saccarosio soluzione 0.25M (corrispodente a 85.55 g/litro)
'''

acquaV = 80  # millilitri
M = 0.25    # M


# %%

Ktable = pd.read_csv(path.join(DATA, 'K.csv'))
CaCl2 = pd.read_csv(path.join(DATA, 'CaCl2.csv'))
NaCl = pd.read_csv(path.join(DATA, 'NaCl.csv'))
saccarosio = pd.read_csv(path.join(DATA, 'saccarosio.csv'))
acqua = pd.read_csv(path.join(DATA, 'acqua.csv'))

# %%
# analisi Cacl2, NaCl, saccarosio
K = Ktable.iloc[0, 0]
NaCl.loc[:, 'M[M]'] = NaCl.loc[:, 'V[ml]'] * M / (acquaV + NaCl.loc[:, 'V[ml]'])
NaCl.loc[:, 'C[S/cm]'] = NaCl.loc[:, 'C[S]'] * K
CaCl2.loc[:, 'M[M]'] = CaCl2.loc[:, 'V[ml]'] * M / (acquaV + CaCl2.loc[:, 'V[ml]'])
CaCl2.loc[:, 'C[S/cm]'] = CaCl2.loc[:, 'C[S]'] * K
saccarosio.loc[:, 'M[M]'] = saccarosio.loc[:, 'V[ml]'] * M / (acquaV + saccarosio.loc[:, 'V[ml]'])
saccarosio.loc[:, 'C[S/cm]'] = saccarosio.loc[:, 'C[S]'] * K

# regressione lineare
NaCl_ls = fit_sine_poly(t=NaCl.loc[:, 'M[M]'].values, x=NaCl.loc[:, 'C[S/cm]'].values)
CaCl2_ls = fit_sine_poly(t=CaCl2.loc[:, 'M[M]'].values, x=CaCl2.loc[:, 'C[S/cm]'].values)
saccarosio_ls = fit_sine_poly(t=saccarosio.loc[:, 'M[M]'].values, x=saccarosio.loc[:, 'C[S/cm]'].values)


# grafico relativo
fig, ax = plt.subplots()
ax.errorbar(y=NaCl.loc[:, 'C[S/cm]'].values, x=NaCl.loc[:, 'M[M]'].values,)
ax.errorbar(y=CaCl2.loc[:, 'C[S/cm]'].values, x=CaCl2.loc[:, 'M[M]'].values)
ax.errorbar(y=saccarosio.loc[:, 'C[S/cm]'].values, x=saccarosio.loc[:, 'M[M]'].values)
plt.show()
# %%
# analisi acqua


