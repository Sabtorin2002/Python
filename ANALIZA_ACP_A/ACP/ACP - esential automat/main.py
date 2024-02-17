#ANALIZA COMPONENTELOR PRINCIPALE

#CERINTA: Variantele componentelor principale

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

dataCSV = pd.read_csv('dataIN/MiseNatPopTari.csv',index_col=0)
nume_instante = list(dataCSV['Country_Name'].values)
denumire_variabile=list(dataCSV.columns)[1:]
variabile_numerice = denumire_variabile[1:]

data_std = (dataCSV[variabile_numerice] - dataCSV[variabile_numerice].mean())/dataCSV[variabile_numerice].std()
model = PCA()
model.fit(data_std)
variance = model.explained_variance_ratio_
print("B.1.")
print(variance)

#CERINTA: Scorurile asociate instantelor
nr_componente = len(variance)
et_componente = ["C" + str(i+1) for i in range(nr_componente)]
print(et_componente)
scores=np.dot(data_std,model.components_.T)
t_scores = pd.DataFrame(scores,index=nume_instante,columns=et_componente)
t_scores.to_csv('dataOUT/scoruri.csv')

#CERINTA:graficul scorurilor in primele 2 axe principala

t_componente = pd.DataFrame(data=model.components_, index=et_componente, columns=variabile_numerice)
print(t_componente)

from matplotlib import pyplot as plt
def plot_componente(t,x,y,title="Componente"):
    fig = plt.figure(title,figsize=(9,9))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(x, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(y, fontdict={"fontsize":12, "color": "b"})
    ax.scatter(t.loc[x],t.loc[y],color='r')

    for var in variabile_numerice:
        ax.text(t.loc[x].loc[var], t.loc[y].loc[var],var)

plot_componente(t_componente,'C1','C2')
plt.show()
