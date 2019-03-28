# %%
import pandas as pd
import numpy as np
from os import path
from np_least_squares import wls_gen
from matplotlib import pyplot as plt, use

# use('Qt5Agg')

DATA = 'exp1/data'
SAVE = 'exp1/save'

# %%

''' 
acqua distillata V = 80 ml
NaCl soluzione 0.25 M (corrispondente a 14.6 g/litro) 
CaCl2 soluzione 0.25 M (corrispondente a 27.75 g/litro) 
saccarosio soluzione 0.25M (corrispodente a 85.55 g/litro)
'''

acquaV = 80  # millilitri
Mol = 0.25    # M

# TODO errori
# discuterne con Belluno e Vicenza

# indici dataframe
M = 'M[M]'
C = 'C[S/cm]'
V = 'V[ml]'
Ca = 'C[S]'

Cteo = 'Cteo[S/cm]'
Casper = 'Csper[S]'
Csper = 'Csper[S/cm]'


# aquisizione dataframe
Ktable = pd.read_csv(path.join(DATA, 'K.csv'))
CaCl2 = pd.read_csv(path.join(DATA, 'CaCl2.csv'))
NaCl = pd.read_csv(path.join(DATA, 'NaCl.csv'))
saccarosio = pd.read_csv(path.join(DATA, 'saccarosio.csv'))
acqua = pd.read_csv(path.join(DATA, 'acqua.csv'))

K = Ktable.iloc[0, 0]

# %%
# analisi Cacl2, NaCl, saccarosio
NaCl.loc[:, M] = NaCl.loc[:, V] * Mol / (acquaV + NaCl.loc[:, V])
NaCl.loc[:, C] = NaCl.loc[:, Ca] * K

CaCl2.loc[:, M] = CaCl2.loc[:, V] * Mol / (acquaV + CaCl2.loc[:, V])
CaCl2.loc[:, C] = CaCl2.loc[:, Ca] * K

saccarosio.loc[:, M] = saccarosio.loc[:, V] * Mol / (acquaV + saccarosio.loc[:, V])
saccarosio.loc[:, C] = saccarosio.loc[:, Ca] * K

# regressione lineare

NaCl_A, NaCl_dA, NaCl_ls = wls_gen(X=NaCl.loc[:, M].values, Y=NaCl.loc[:, C].values, dX=None, dY=None)
CaCl2_A, CaCl2_dA, CaCl2_ls = wls_gen(X=CaCl2.loc[:, M].values, Y=CaCl2.loc[:, C].values, dX=None, dY=None)
saccarosio_A, saccarosio_dA, saccarosio_ls = wls_gen(X=saccarosio.loc[:, M].values, Y=saccarosio.loc[:, C].values, dX=None, dY=None)

# grafico

figcm, axcm = plt.subplots()
axcm.errorbar(x=NaCl.loc[:, M].values, y=NaCl.loc[:, C].values, fmt='.', color='orange', label='NaCl')
axcm.plot(NaCl.loc[:, M].values, NaCl_ls['model'][0], color='orange', alpha=0.5)

axcm.errorbar(x=CaCl2.loc[:, M].values, y=CaCl2.loc[:, C].values, fmt='.', color='green', label='CaCl2')
axcm.plot(CaCl2.loc[:, M].values, CaCl2_ls['model'][0], color='green', alpha=0.5)

axcm.errorbar(x=saccarosio.loc[:, M].values, y=saccarosio.loc[:, C].values, fmt='.', color='purple', label='saccarosio')
axcm.plot(saccarosio.loc[:, M].values, saccarosio_ls['model'][0], color='purple', alpha=0.5)

axcm.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useLocale=False, useMathText=True)
axcm.set_xlabel(M)
axcm.set_ylabel(C)
axcm.legend(loc='best')

figcm.savefig(path.join(SAVE, 'figcm.pdf'), dpi=None, facecolor='w', edgecolor='w', orientation='landscape', papertype='a4', transparent=True)


# %%
# analisi acqua

acqua.loc[:, Csper] = acqua.loc[:, Casper] * K

ind = np.arange(3)  # the x locations for the groups
width = 0.35       # the width of the bars

figaq, axaq = plt.subplots()
rects1 = axaq.bar(ind, acqua.loc[:, Cteo].values, width, color='r', label='official')
rects2 = axaq.bar(ind + width, acqua.loc[:, Csper].values, width, color='y', label='measured')

axaq.set_xticks(ind + width / 2)
axaq.set_xticklabels(acqua.loc[:, 'Brand'].values)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axaq.text(rect.get_x()+rect.get_width()/2., 1.05*height, '{:.3}'.format(height), ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

axaq.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useLocale=False, useMathText=True)
axaq.set_ylabel(C)
axaq.legend(loc='best')

figaq.savefig(path.join(SAVE, 'figaq.pdf'), dpi=None, facecolor='w', edgecolor='w', orientation='landscape', papertype='a4', transparent=True)

# %%
plt.show()

