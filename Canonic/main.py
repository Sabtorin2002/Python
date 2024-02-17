import pandas as pd
import numpy as np
import copy
import seaborn as sb
import sklearn.cross_decomposition as skl
from matplotlib import pyplot as plt
#analiza canonica

vot = pd.read_csv("dataIN/Vot.csv")
localitatile = pd.read_csv("dataIN/Coduri_localitati.csv")

coloane = list(vot)
categorii = list(coloane[3:])
print(categorii)

#ACC(analiza componentelor canonice)
tabel = pd.read_csv("dataIN/Vot.csv")
obs = tabel.index.values#numarul de linii din csv
#print(obs)
var = tabel.columns.values#etichetele coloanelor din csv
#print(var)

#extragere subseturi de date
#doar etichetele coloanelor
x_col = var[3:7]
y_col = var[7:]

#salvarea de date in 2 subseturi
#valori barbati
X = tabel[x_col].values
#valori femei
Y = tabel[y_col].values
print(X)

#CERINTA1.calculul scorurilor canonice pentru cele 2 seturi(Barbati si femei)

def standardizare(X):
    medii = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X-medii)/std
    return Xstd

#seturile de date standardizate
Xstd = standardizare(X)
Ystd = standardizare(Y)

n, p = np.shape(X)#n=numarul de linii din X; p=numarul de coloane din X
s,q = np.shape(Y)#q=numarul de coloane din Y
#se face minimul ca asa e corect
m=min(p,q)

modelACC = skl.CCA(n_components=m)
modelACC.fit(X=Xstd, Y=Ystd)

z,u=modelACC.transform(X=Xstd, Y=Ystd)

#cate coloane sunt, atatea z-uri sunt
z_df = pd.DataFrame(data=z, index=obs, columns=['z'+ str(j+1) for j in range(p)])
u_df = pd.DataFrame(data=u, index=obs, columns=['u'+ str(j+1) for j in range(q)])

#afisarea scorurilor canonice
z_df.to_csv('dataOUT/z_barbati.csv')
u_df.to_csv('dataOUT/u_femei.csv')


#CERINTA2.calculul corelatiilor canonice

#extrage incarcarile canonice din X
cor_can_x = modelACC.x_loadings_
cor_can_x_df =pd.DataFrame(data=cor_can_x, index=x_col, columns=['z'+ str(j+1) for j in range (p)])

#extrage incarcarile canonice din Y
cor_can_y = modelACC.y_loadings_
cor_can_y_df = pd.DataFrame(data=cor_can_y, index=y_col, columns=['u'+ str(j+1) for j in range(q)])

cor_can_x_df.to_csv("dataOUT/CorelatiiCanoniceBarbati.csv")
cor_can_y_df.to_csv("dataOUT/CorelatiiCanoniceFemei.csv")

#csv concatenat barbati+femei
cor_can_total_df = pd.concat([cor_can_x_df, cor_can_y_df], axis=1)
cor_can_total_df.to_csv("dataOUT/r.csv")

#CERINTA 3.Trasarea plotului de instante pentru primele 2 radacini canonice
#(x:z1, y=z2) si (cox:u1, y=u2) --> biplot si sa fie in acelasi grafic -->unire radacini inseamna biplot
def biplot(x,y,xlabel,ylabel,titlu,e1,e2):
    fg = plt.figure(figsize=(11,8))
    ax = fg.add_subplot(1,1,1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=14, color='b')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #x[:,0] = selectarea tuturor elementelor de pe prima coloana
    #c='r' = culoarea punctelor(rosu)
    #label = adaugi o eticheta
    ax.scatter(x=x[:,0], y=x[:,1], c='r', label='Set X')
    ax.scatter(x=y[:,0], y=y[:,1], c='blue', label='Set Y')
    if e1 is not None:
        for i in range(len(e1)):
            ax.text(x=x[i, 0], y=x[i, 1], s=e1[i])
    if e2 is not None:
        for i in range(len(e2)):
            ax.text(x=y[i, 0], y=y[i, 1], s=e2[i])
    ax.legend()
    plt.show()

# primul indice(:) se refera la toate randurile
# al doilea indice (:) se refera la primele 2 coloane de pe feicare rand
# noua ne cere primele 2, deci :2

#in e1 si e2 ar trebui clase de tip LISTA
#obs=class 'numpy.ndarray'
#list(obs)=class 'list'
biplot(z[:,:2], u[:,:2],xlabel='(z1,z2)', ylabel='(u1,u2)',
       titlu='Biplot var in spatiul radacinilor canonice (z1,z2) si (u1,u2)',
       e1=list(obs), e2=list(obs))

#Cerinta4.Trasare corelograma corelatii variabile observate - variabile canonice

def corelograma(x,min,max,titlu="Corelograma"):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax=sb.heatmap(data=x, vmin=min, vmax=max, cmap='bwr', annot=True, ax=ax)
    for i in range(len(x)):
        ax.set_xticklabels(labels=x.columns, ha="right", rotation=30)


corelograma(cor_can_total_df,-1,1,"cor_can_total_df")
plt.show()

#CERINTA5.Calculul corelatii radacinii canonice
#se calculeaza matricea de corelatia intre z si u folosing np.corrcoef
#rowbar = False -->variabilele sunt considerate coloane
#[:m,m:]-->primele m randuri si m coloane
cor_rad = np.diag(np.corrcoef(z,u,rowvar=False)[:m,m:])
cor_rad_df =pd.DataFrame(data=cor_rad)
cor_rad_df.to_csv("dataOUT/cor_rad_df.csv")

#CERINTA6.Graficul distributiei observatiilor in spatiul radacinilor canonice z1,z2,u1,u2
#z[:,0] ->prima coloana
#z[:,1] ->a doua coloana
def biplotObs(x,y,xLabel='X',yLabel='Y',titlu='BiplotObs',obs=None):
    fig = plt.figure(figsize=(11,8))
    ax=fig.add_subplot(1,1,1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=12, color='Green')
    ax.set_xlabel(xLabel, fontsize=12, color='Green')
    ax.set_ylabel(yLabel, fontsize=12, color='Green')
    ax.scatter(x=z[:,0], y=z[:,1], color='Red', label='Set X')
    ax.scatter(x=u[:,0], y=u[:,1], color='Red', label='Set Y')
    if obs is not None:
        for i in range(len(obs)):
            ax.text(x=z[i,0], y=z[i,1], s=obs[i])
            ax.text(x=u[i,0], y=u[i,1], s=obs[i])
    ax.legend()

biplotObs(z,u, obs=obs)
plt.show()



















