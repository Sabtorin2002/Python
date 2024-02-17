import pandas as pd
import numpy as np
import copy
import seaborn as sb
from matplotlib import pyplot as plt

vot = pd.read_csv('dateIN/Vot.csv')
coduri_localitati = pd.read_csv('dateIN/Coduri_localitati.csv')

#CERINTA1.Procentele participarii la vot pe alegatori(numar voturi * 100/ votanti_LP)
cerinta1 = pd.DataFrame()
cerinta1 = vot.copy()
#CAND AM DIV, axis=0, impartirea o facem cu div
cerinta1.iloc[:,3:]=(cerinta1.iloc[:,3:]*100).div(cerinta1.iloc[:,2],axis=0)
cerinta1 = cerinta1.drop('Votanti_LP',axis=1)
cerinta1.to_csv('dateOUT/cerinta1.csv',index=False)

#CERINTA2.Procentele participarii la vot pe categorii de alegatori la nivel de judet.Se va salva indicativul judetului
#si procentele pe fiecare categorie

cerinta2 = pd.DataFrame()
cerinta2 = vot.drop('Localitate',axis=1).copy()
cerinta2 = cerinta2.merge(coduri_localitati[['Judet','Siruta']],left_on='Siruta', right_on='Siruta').drop('Siruta', axis=1)
cerinta2 = cerinta2.groupby(by='Judet').sum()
cerinta2.iloc[:,1:] = (cerinta2.iloc[:,1:]*100).div(cerinta2.iloc[:,0], axis=0)
cerinta2 = cerinta2.drop('Votanti_LP', axis=1)
cerinta2.to_csv('dateOUT/cerinta2.csv')
print(cerinta2)

print('SUBIECT B')

#SUBIECT B
#CANONIC

#ACP -> from sklearn.decomposition import PCA
#Canonic -> import sklearn.cross_decomposition as skl

# coloane = list(vot)
# print(coloane)
#
# categorii = list(coloane)[3:]
# print(categorii)

#ACC -> ANALIZA COMPONENTELOR CANONICE

# tabel = pd.read_csv('dateIN/Vot.csv')
# obs = tabel.index#numarul de linii din Vot.csv
# var = tabel.columns[:]#etichetele coloanelor din Vot.csv
#
# #extragere subseturi de date
# #doar etichetele coloanelor
# x_col = var[3:7]
# #print(x_col)
#
# y_col = var[7:]
# print(y_col)
#
# #salvarea de date in 2 subseturi
# X=tabel[x_col].values
# #print(X)
#
# Y=tabel[y_col].values
# #print(Y)
#
# #CERINTA 1.SCORURILE CANONICE PENTRU CELE 2 SETURI DE DATE
# def standardizare(X):
#     medii = np.mean(X,axis=0)
#     std = np.std(X,axis=0)
#     Xstd = (X-medii)/std
#     return Xstd
#
# #seturile de date standardizate
# Xstd = standardizare(X)
# Ystd = standardizare(Y)
#
# n, p = np.shape(X) #n=numarul de linii din X, p=numarul de coloane din X
# s, q = np.shape(Y)
# m = min(p,q)
#
# modelACC = skl.CCA(n_components=m)
# modelACC.fit(X=Xstd, Y=Ystd)
#
# #obtinere scoruri canonice
# z, u = modelACC.transform(X=Xstd, Y=Ystd) # z pentru barbati, u pentru femei
#
# #transformare scoruri in df
# z_df = pd.DataFrame(data=z, index=obs, columns=['z' + str(i+1) for i in range(p)])
# u_df = pd.DataFrame(data=u, index=obs, columns=['u' + str(i+1) for i in range(q)])
#
# z_df.to_csv('dateOUT/z_csv',index=False)
# u_df.to_csv('dateOUT/u_csv',index=False)
#
# #CERINTA 2.CORELATIILE CANONICE
# cor_can_x = modelACC.x_loadings_
# cor_can_x_df = pd.DataFrame(data=cor_can_x, index=x_col, columns=['z'+ str(i+1) for i in range(p)])
#
# cor_can_y = modelACC.y_loadings_
# cor_can_y_df = pd.DataFrame(data=cor_can_y, index=y_col, columns=['u'+ str(i+1) for i in range (q)])
#
# cor_can_x_df.to_csv('dateOUT/CorelatiiCanoniceBarbati.csv')
# cor_can_y_df.to_csv('dateOUT/CorelatiiCanoniceFemie.csv')
#
# cor_can_total_df = pd.concat([cor_can_x_df, cor_can_y_df],axis=1)
# cor_can_total_df.to_csv('dateOUT/r.csv')
#
# #CERINTA 3.CORELATII RADACINI CANONICE
# #se calculeaza matricea de corelatie intre z si u folosing np.corrcoef
# cor_rad = np.diag(np.corrcoef(z,u)[:m,:m])
# cor_rad_df = pd.DataFrame(data=cor_rad)
# cor_rad_df.to_csv('dateOUT/CorelatiiRadaciniCanonice.csv',index=False)
#
# #CERINTA 4.Trasare corelograma corelatii variabile observate
# def corelograma(x,min,max,title):
#     fig=plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#     ax = sb.heatmap(data=x, vmin=min, vmax=max)
#
# corelograma(cor_can_total_df,-1,1,'Corelograma')
# plt.show()

print('inca o data')

# #ANALIZA CANONICA
# import sklearn.cross_decomposition as skl
#
# tabel = pd.read_csv('dateIN/Vot.csv')
# index = tabel.index
# coloane_numerice = tabel.columns[:] #iau etichetele coloanelor
# print(coloane_numerice)
#
# #extragere subseturi de date
#
# x_col = coloane_numerice[3:7] #iau etichetele coloanelor pentru barbati
# y_col = coloane_numerice[7:] #iau etichetele coloanelor pentru femei
#
# #salvarea datelor in 2 subseturi
#
# X = tabel[x_col].values #extragrea VALORILOR pentru barbati
# Y = tabel[y_col].values #extragerea VALORILOR pentru femei
#
# #CERINTA 1.SCORURILE CANONICE IN CELE 2 SETURI DE DATE
#
# #STANDIZAREA DE LA ACP
# def standardizare(X):
#     medii = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     Xstd = (X-medii)/std
#     return Xstd
#
# Xstd = standardizare(X)
# Ystd = standardizare(Y)
#
# n, p = np.shape(X)
# s, q = np.shape(Y)
#
# m = min(p,q)
#
# modelACC = skl.CCA(n_components=m)
# modelACC.fit(X=Xstd, Y=Ystd)
#
# #obtinerea scorurilor canonice => coloane = etichete
#
# z, u = modelACC.transform(X=Xstd, Y=Ystd)
#
# etichete_z = ['z'+ str(i+1) for i in range(p)]
# etichete_u = ['u'+ str(i+1) for i in range(q)]
#
# z_df = pd.DataFrame(data=z, index=index, columns=etichete_z)
# u_df = pd.DataFrame(data=u, index=index, columns=etichete_u)
#
# z_df.to_csv('dateOUT/ScoruriCanoniceBarbati.csv')
# u_df.to_csv('dateOUT/ScoruriCanoniceFemei.csv')
#
#
# #CERINTA 2.CORELATII CANONICE
# cor_can_x = modelACC.x_loadings_
# cor_can_x_df = pd.DataFrame(data=cor_can_x, index=x_col, columns=etichete_z)
#
# cor_can_y = modelACC.y_loadings_
# cor_can_y_df = pd.DataFrame(data=cor_can_y, index=y_col, columns=etichete_u)
#
# cor_can_total_df = pd.concat([cor_can_x_df, cor_can_y_df], axis=0)
# cor_can_total_df.to_csv('dateOUT/Total.csv')
#
# #CERINTA 3.CORELATII RADACINI CANONICE
# #se calculeaza matricea de corelatie intre z si u folosind np.corrcoeff
#
# cor_rad = np.diag(np.corrcoef(z,u)[:m,m:])
# cor_rad_df = pd.DataFrame(data=cor_rad)
# cor_rad_df.to_csv('dateOUT/RadaciniCanonice.csv')
#
# #CERINTA 4.TRASARE CORELOGRAMA CORELATII CANONICE
#
# def corelograma(date, min, max, title):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#
#     ax = sb.heatmap(data=date, vmin=min, vmax=max)
#
# corelograma(cor_can_total_df,-1,1,'Corelograma')
# plt.show()

print('inca o data')

#ANALIZA CANONICA

import sklearn.cross_decomposition as skl

tabel = pd.read_csv('dateIN/Vot.csv')
index = tabel.index
coloane_numerice = tabel.columns[:]

x_col = coloane_numerice[3:7]
y_col = coloane_numerice[7:]

X = tabel[x_col].values
Y = tabel[y_col].values

#CERINTA 1.SCORURILE CANONICE PENTRU CELE 2 SETURI DE DATE
def standardizare(x):
    medii = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    Xstd = (x-medii)/std
    return Xstd
Xstd = standardizare(X)
Ystd = standardizare(Y)
n,p = np.shape(X)
s,q = np.shape(Y)

m = min(p,q)

modelCCA = skl.CCA(n_components=m)
modelCCA.fit(X=Xstd, Y=Ystd)

z,u = modelCCA.transform(X=Xstd, Y=Ystd)

etichete_x = ['z' + str(i+1) for i in range(p)]
etichete_y = ['u' + str(i+1) for i in range(q)]

z_df = pd.DataFrame(data=z, index=index, columns=etichete_x)
u_df = pd.DataFrame(data=u, index=index, columns=etichete_y)

z_df.to_csv('dateOUT/z1.csv')
u_df.to_csv('dateOUT/u1.csv')

#CERINTA 2.CORELATII CANONICE
cor_can_x = modelCCA.x_loadings_
cor_can_x_df = pd.DataFrame(data=cor_can_x, index=x_col, columns=etichete_x)

cor_can_y = modelCCA.y_loadings_
cor_can_y_df = pd.DataFrame(data=cor_can_y, index=y_col, columns=etichete_y)

cor_can_total_df = pd.concat([cor_can_x_df, cor_can_y_df],axis=0)

cor_can_total_df.to_csv('dateOUT/Total1.csv')

#CERINTA 3.RADACINI CANONICE
#se calculeaza cu np.corrcoef

cor_rad = np.diag(np.corrcoef(z,u)[:m,m:])
cor_rad_df = pd.DataFrame(data=cor_rad)
cor_rad_df.to_csv('dateOUT/radacini.csv')

#CERINTA 4.CORELOGRAMA CORELATII

def corelograma(date, min, max, title):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax, plt.Axes)

    ax=sb.heatmap(data=date, vmin=min, vmax=max)

corelograma(cor_can_total_df,-1,1,'Corelograma')
plt.show()
