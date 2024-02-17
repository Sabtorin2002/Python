import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
import seaborn as sb
from sklearn.decomposition import PCA

#ANALIZA CLUSTER

alcool=pd.read_csv('dateIN/alcohol.csv', index_col=0)
coduri_tari_extins = pd.read_csv('dateIN/CoduriTariExtins.csv',index_col=0)

#stocare valori index(toate tarile, type=class pandas..)
tari = alcool.index
#variabilele pentru analiza (cele relevante - anii)
variabile = alcool.columns[1:] #sare peste index si o sa fie codul coloana 0 si anul 200 coloana 1

#CERINTA1.Matrice ierarhie privind jonctiunile facute, pentru fiecare jonctiune se va specifica
#clusterii intrati in jonctiune, distana dintre cei 2 clusteri si numarul de instante in clusterul nou format

x=alcool[variabile].values#valorile din coloanele bune, DOAR VALORILE
#print(x)

#suplimentar
def change_nan(x):
    #true = daca valorile nu sunt numere, false = daca valorile sunt numere
    is_nan = np.isnan(x)
    #gasire indici unde isnan este TRUE si ii stocam in k_nan
    k_nan=np.where(is_nan)
    #se inlocuiesc valorile Nan din 'x' cu media valorilor din aceeasi coloana ignorand valorile NaN
    #axis=0 -> pe coloane, contine doar valorile corespunzatoare coloanelor in care exista valori NaN
    x[k_nan] = np.nanmean(x[:,k_nan[1]], axis=0)

change_nan(x)

#afisare matrice
metoda='ward'
h=hclust.linkage(x, metoda)
print(h)

#NUMARUL DE JONCTIUNI
#numarul tuturor tarilor
nr=len(tari)-1
print(nr)

#DISTANTA MAXIMA intre 2 clusteri
#h[1:,2] - incepand de la linia 2 pana la final si doar coloana
#h[:(nr-1),2] - incepe de la primul rand pana linia nr -1 si doar coloana 3
distanta_maxima = np.argmax(h[1:,2] - h[:(nr-1),2])
print(distanta_maxima)

#NUMARUL DE CLUSTERI
nr_clusteri = nr - distanta_maxima
print(nr_clusteri)


#CERINTA2.Graficul dendograma pentru partitia optimala

def dendograma(h, prag, tari):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    hclust.dendrogram(h, color_threshold=prag, labels=tari, ax=ax)

def partitie(h, nr_clusteri, nr, tari):
    distanta_maxima = nr - nr_clusteri
    prag =(h[distanta_maxima,2] + h[distanta_maxima+1,2])/2

    #!!! apelarea dendogramei
    dendograma(h,prag,tari)

    n = nr+1
    c = np.arange(n) # id-uri, nr intregi de la 0 la 186(interval inchis)

    #parcugem numarul de tari(lungimea)+1 - nr_clusteri
    for i in range (n-nr_clusteri):
        k1=h[i,0] #stocam in k1 valorile de pe prima coloana din h parcurgand liniile
        k2=h[i,1] #stocam in k2 valorile de pe a doua coloana din h parcurgand liniile

        # calculezi pozitia lui c ca fiind valorile de tip 'k1' ce sunt egale cu una din valorile id-urilor tarilor [0,186]
        #stocam in c: lungimea +1+ numarul liniei aferente
        c[c == k1] = n+i
        # calculezi pozitia lui c ca fiind valorile de tip 'k1' ce sunt egale cu una din valorile id-urilor tarilor [0,186]
        # stocam in c: lungimea +1+ numarul liniei aferente
        c[c == k2] = n+i

    #array de coduri numerice care reprezinta clusterele
    coduri = pd.Categorical(c).codes
    return np.array(['c'+ str(cod+1) for cod in coduri])

#componenta partitiei optimale
partitie_optimala = partitie(h, nr_clusteri, nr, tari)
plt.show()

#CERINTA3.Componenta paritiei optimale.Se va determina clusterul de care aparatine.Se va salva in popt.csv

partitie_optimatila_df = pd.DataFrame(data={"Tari": tari,
                                  "Cluster": partitie_optimala})
partitie_optimatila_df.to_csv('dateOUT/popt.csv')

#CERINTA4.Plot partitie in axe principale - distributie

def plot_partitie(z, partitie_optimala):
    fig = plt.figure('Plot partitie optimala in axele principale',figsize=(11,8))
    ax = fig.add_subplot(1,1,1)

    sb.scatterplot(x=z[:, 0], y=z[:, 1], hue=partitie_optimala, hue_order=np.unique(partitie_optimala), ax=ax)

model_PCA = PCA(2)#acp - analiza componente principale
z=model_PCA.fit_transform(x)

plot_partitie(z,partitie_optimala)
plt.show()


# def plot_partitie(z, paritie_optimala):
#     fig = plt.figure('Plot partitie optimala in axele principale', figsize=(9, 9))
#     ax = fig.add_subplot(1, 1, 1)
#
#     sb.scatterplot(x=z[:, 0], y=z[:, 1], hue=paritie_optimala, hue_order=np.unique(partitie_optimala), ax=ax)
#
#
# model_pca = PCA(2)
# z = model_pca.fit_transform(x)
#
# plot_partitie(z, partitie_optimala)
# plt.show()

#CERINTA5.Histograma clusteri
# CERINTA 5: Histograma pentru partitia optima
partitie_optimala_df = pd.DataFrame({"Tari": tari, "Cluster": partitie_optimala})

plt.figure(figsize=(10, 6))
sb.histplot(partitie_optimala_df["Cluster"], bins=range(1, nr_clusteri + 2), discrete=True)
plt.title('Histograma Partitiei Optime')
plt.xlabel('Cluster')
plt.ylabel('Numar de tari')
plt.show()