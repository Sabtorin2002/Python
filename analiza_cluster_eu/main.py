import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA

# alcool = pd.read_csv('dateIN/alcohol.csv',index_col=0)
# coduri = pd.read_csv('dateIN/CoduriTariExtins.csv',index_col=0)
#
# index = alcool.index
# print(len(index))
#
# coloane = alcool.columns[1:]
# print(coloane)
#
# date_numerice = alcool[coloane].values
# print(date_numerice)
#
# #ASTA NU SE INVATA
# def change_nan(x):
#     #true = daca valorile nu sunt numere, false = daca valorile sunt numere
#     is_nan = np.isnan(x)
#     #gasire indici unde isnan este TRUE si ii stocam in k_nan
#     k_nan=np.where(is_nan)
#     #se inlocuiesc valorile Nan din 'x' cu media valorilor din aceeasi coloana ignorand valorile NaN
#     #axis=0 -> pe coloane, contine doar valorile corespunzatoare coloanelor in care exista valori NaN
#     x[k_nan] = np.nanmean(x[:,k_nan[1]], axis=0)
#
# change_nan(date_numerice)
#
# #CERINTA 1. MATRICEA IERARHIE,numar_jonctiuni, distanta maxima intre clusteri, numar_clusteri
#
# #afisare matrice
# metoda = 'ward'
# h=hclust.linkage(date_numerice,metoda)
# print(h) #h = matricea
#
# #numarul de jonctiuni
# numar_jonctiuni = len(index)-1
# print(numar_jonctiuni)
#
# #distanta intre 2 clusteri
# distanta_maxima = np.argmax(h[1:,2] - h[:(numar_jonctiuni-1),2])
# print(distanta_maxima)
#
# #numar clusteri
# nr_clusteri = numar_jonctiuni - distanta_maxima
# print(nr_clusteri)
#
# #CERINTA 2. DENDOGRAMA PENTRU PARTITIA OPTIMALA
#
# def dendograma(h, prag, index):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     ax.set_title('Dendograma')
#     assert isinstance(ax, plt.Axes)
#
#     hclust.dendrogram(h, color_threshold=prag, labels=index, ax=ax)
#
# def partitie(h, nr_clust, nr_jonct, index):
#     distanta = nr_jonct - nr_clust
#     prag = (h[distanta,2] + h[distanta+1,2])/2
#
#     #!!!apelare dendograma
#     dendograma(h,prag,index)
#
#     lungime_noua =  numar_jonctiuni + 1
#     numere_intregi = np.arange(lungime_noua)#iti ia nr intregi de la 1 la 100
#
#     for i in range(lungime_noua - nr_clust):
#         k1 = h[i,0]
#         k2 = h[i,1]
#
#         numere_intregi[numere_intregi==k1] = lungime_noua+i
#         numere_intregi[numere_intregi==k2] = lungime_noua+i
#
#     coduri = pd.Categorical(numere_intregi).codes
#     return pd.array(['c' + str(cod+1) for cod in coduri])
#
#
# partitie_optimala = partitie(h, nr_clusteri, numar_jonctiuni, index)
# plt.show()
#
# #CERINTA 3. Componenta partitiei optimale
# # partitie_optimala_df = pd.DataFrame(data=partitie_optimala, index=index)
# #partitie_optimala_df.to_csv('dateOUT/partitie_optimala.csv')
# componenta_df = pd.DataFrame(data=partitie_optimala)
# componenta_df.to_csv('dateOUT/componenta_partitie.csv')

print('inca o data')

# alcool = pd.read_csv('dateIN/alcohol.csv',index_col=0)
# coduri = pd.read_csv('dateIN/CoduriTariExtins.csv',index_col=0)
#
# index = alcool.index
# coloane_numerice = alcool.columns[1:]
#
# date_numerice = alcool[coloane_numerice].values
#
#
# #CERINTA 1. MATRICEA DE IERARHIE, etc
# import scipy.cluster.hierarchy as hclust
#
# #ASTA NU SE INVATA
# def change_nan(x):
#     #true = daca valorile nu sunt numere, false = daca valorile sunt numere
#     is_nan = np.isnan(x)
#     #gasire indici unde isnan este TRUE si ii stocam in k_nan
#     k_nan=np.where(is_nan)
#     #se inlocuiesc valorile Nan din 'x' cu media valorilor din aceeasi coloana ignorand valorile NaN
#     #axis=0 -> pe coloane, contine doar valorile corespunzatoare coloanelor in care exista valori NaN
#     x[k_nan] = np.nanmean(x[:,k_nan[1]], axis=0)
#
# change_nan(date_numerice)
#
# #afisare matrice
# import scipy.cluster.hierarchy as hclust
# metoda = 'ward'
# h = hclust.linkage(date_numerice, metoda)
# print(h)
#
# #numarul de jonctiuni
# numar_jonctiuni = len(index)-1
# print(numar_jonctiuni)
#
# #distanta maxima
# distanta_maxima = np.argmax(h[1:,2] - h[:(numar_jonctiuni-1),2])
# print(distanta_maxima)
#
# #numar de clusteri
# numar_clusteri = numar_jonctiuni - distanta_maxima
# print(numar_clusteri)
#
#
# #CERINTA 2.DENDOGRAMA PENTRU PARTITIA OPTIMALA
# def dendograma(h, prag, index):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#     ax.set_title('Dendograma')
#
#     hclust.dendrogram(h, color_threshold=prag, labels=index, ax=ax)
#
# def partitie(h, nr_jonc, nr_clust, index):
#     distanta= nr_jonc - nr_clust
#     prag = (h[distanta,2]+h[distanta+1,2])/2
#
#     dendograma(h, prag, index)
#
#     lungime_noua = numar_jonctiuni + 1
#     numere_intregi = np.arange(lungime_noua)
#
#     for i in range(lungime_noua-nr_clust):
#         k1 = h[i,0]
#         k2 = h[i,1]
#
#         numere_intregi[numere_intregi==k1] = lungime_noua+i
#         numere_intregi[numere_intregi==k2] = lungime_noua+i
#
#     coduri = pd.Categorical(numere_intregi).codes
#     return pd.array(['c'+ str(cod+1) for cod in coduri])
#
# partie_optimala = partitie(h, numar_jonctiuni, numar_clusteri, index)
# plt.show()
#
# #CERINTA 3. COMPONETENELE PARTITIEI OPTIMALE
#
# componenta_df = pd.DataFrame(data=partie_optimala)
# componenta_df.to_csv('dateOUT/componenta_partitie2.csv')

print('inca o data')

alcool = pd.read_csv('dateIN/alcohol.csv',index_col=0)
coduri = pd.read_csv('dateIN/CoduriTariExtins.csv',index_col=0)

index = alcool.index
coloane_numerice = alcool.columns[1:]#trb sa extrag de aici valorile

date_numerice = alcool[coloane_numerice].values

#CERINTA 1. MATRICEA IERARHIE, etc
import scipy.cluster.hierarchy as hclust

#ASTA NU SE INVATA
def change_nan(x):
    #true = daca valorile nu sunt numere, false = daca valorile sunt numere
    is_nan = np.isnan(x)
    #gasire indici unde isnan este TRUE si ii stocam in k_nan
    k_nan=np.where(is_nan)
    #se inlocuiesc valorile Nan din 'x' cu media valorilor din aceeasi coloana ignorand valorile NaN
    #axis=0 -> pe coloane, contine doar valorile corespunzatoare coloanelor in care exista valori NaN
    x[k_nan] = np.nanmean(x[:,k_nan[1]], axis=0)

change_nan(date_numerice)

metoda = 'ward'
h=hclust.linkage(date_numerice,metoda)
print(h)

#numar jonctiuni
numar_jonctiuni = len(index)-1
print(numar_jonctiuni)

#distanta maxima
distanta_maxima = np.argmax(h[1:,2] - h[:(numar_jonctiuni-1),2])
print(distanta_maxima)

#numar clusteri
numar_clusteri = numar_jonctiuni - distanta_maxima
print(numar_clusteri)

#CERINTA 2. DENDOGRAMA PENTRU PARTITIA OPTIMALA

def dendograma(h,prag, index):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Dendograma')
    assert isinstance(ax, plt.Axes)

    hclust.dendrogram(h, color_threshold=prag, labels=index, ax=ax)

def partitie(h, nr_clust, nr_jonct, index):
    distanta = nr_jonct - nr_clust
    prag = (h[distanta,2]+h[distanta+1,2])/2#medie

    dendograma(h,prag,index)

    lungime_noua = nr_jonct+1#nrjonc ca e mai mare
    numere_intregi = np.arange(lungime_noua)#alocare ca am modifciat lungime noua

    for i in range(lungime_noua - nr_clust):
        k1=h[i,0]
        k2=h[i,1]

        numere_intregi[numere_intregi==k1] = lungime_noua+i
        numere_intregi[numere_intregi==k2] = lungime_noua+i

    coduri = pd.Categorical(numere_intregi).codes#facem numerele intregi coduri
    return pd.array(['c'+ str(cod+i) for cod in coduri])#returnam etichete

partitie_optimala = partitie(h, numar_clusteri, numar_jonctiuni, index)
plt.show()

#CERINTA 3.COMPONENTELE PARTITIEI OPTIMALE
componenta_df = pd.DataFrame(data=partitie_optimala)
componenta_df.to_csv('dateOUT/componenta_partitie3.csv')
