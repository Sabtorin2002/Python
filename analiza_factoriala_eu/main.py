import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

print('prima data')

# voturi = pd.read_csv('dateIN/VotBUN.csv',index_col=0)
# coduri_localitati = pd.read_csv('dateIN/Coduri_Localitati.csv',index_col=0)
#
# #denumirile vaiabilelor sub forma de index
# variabile = voturi.columns[2:]
# sirute = voturi.index
# print(sirute)
#
# #SUBIECTUL B
# #ANALIZA FACTORIALA
#
# #CERINTA 1.TESTUL BARTLETT -> P-value
# #extragem valorile relevante
#
# date_numerice=voturi[variabile].values
# print(date_numerice)
#
# bartlett = fact.calculate_bartlett_sphericity(date_numerice)
#
# print('Testul Bartlett:',bartlett) #prima valoare = coeficient , a doua valoare = P-value
#
# #CERINTA 2. KMO
#
# kmo = fact.calculate_kmo(date_numerice)
# #print(kmo)
#
# nume_coloane = variabile.tolist()
# print(nume_coloane)
# #append adauaga elemente intr-una array
# tabel_kmo = pd.DataFrame(data={'Index KMO': np.append(kmo[0], kmo[1])},
#                          index=nume_coloane + ['Total'])
#
# tabel_kmo.to_csv('dateOUT/tabel_kmo.csv')
#
#
# #CERINTA 3. SCORURILE FACTORIALE
#
# from sklearn.decomposition import FactorAnalysis
#
# model_factorial = FactorAnalysis(n_components=len(variabile))
#
# #fit_transform iti da SCORURILE
# scoruri_factoriale = model_factorial.fit_transform(date_numerice)
#
# etichete_factoriale =['F'+ str(i+1) for i in range (len(variabile))]
#
# scoruri_factoriale_df = pd.DataFrame(data=scoruri_factoriale,index=sirute, columns=etichete_factoriale)
#
# scoruri_factoriale_df.to_csv('dateOUT/scoruri_factoriale.csv')
# print(scoruri_factoriale_df)
#
# #CERINTA 4. GRAFICUL SCORURILOR FACTORIALE PENTRU PRIMII 2 FACTORI:F1 SI F2, grafic de puncte
#
# def grafic_plt(date, x, y, title,sirute):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax,plt.Axes)
#
#     #setam etichetele
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#
#     #scatter -> puncte
#     ax.scatter(date[x], date[y], color='r')
#
#     #text pentru puncte
#     for i in range(len(date)):
#         ax.text(date[x].iloc[i], date[y].iloc[i], sirute[i])
#
# #VREA DATAFRAME PRIMUL PARAMETRU
# grafic_plt(scoruri_factoriale_df,'F1','F2','Graficul scorurilor',sirute)
# plt.show()


print('a doua oara')


# voturi = pd.read_csv('dateIN/VotBUN.csv')
# coduri_locatlitati = pd.read_csv('dateIN/Coduri_Localitati.csv')
#
# variabile = voturi.columns[2:]
# print(variabile)
# sirute = voturi.index
# nume_coloane = variabile.tolist()
#
# #SUBIECTUL B
# #ANALIZA FACTORIALA
#
# #CERINTA 1.TESTUL BARTLETT
#
# date_numerice = voturi[variabile].values
# print(date_numerice)
#
# bartlett = fact.calculate_bartlett_sphericity(date_numerice)
#
# print(bartlett) #a doua valoare este P-value
#
# #CERINTA 2.KMO
#
# kmo = fact.calculate_kmo(date_numerice)
# # #append adauaga elemente intr-un array
# tabel_kmo = pd.DataFrame(data=np.append(kmo[0],kmo[1]),
#                          index=nume_coloane + ['Total'])
#
# tabel_kmo.to_csv('dateOUT/tabel_kmo2.csv')
#
# #CERINTA 3. SCORURILE FACTORIALE
# from sklearn.decomposition import FactorAnalysis
#
# model_factorial = FactorAnalysis(n_components=len(variabile))
#
# scoruri_factoriale = model_factorial.fit_transform(date_numerice)
#
# etichete_scoruri = ['F' + str(i+1) for i in range (len(variabile))]
#
# scoruri_factoriale_df = pd.DataFrame(data=scoruri_factoriale, columns=etichete_scoruri)
#
# scoruri_factoriale_df.to_csv('dateOUT/scoruri_factoriale_2.csv')
#
# #CERINTA 4. GRAFICUL SCORURILOR FACTORIALE F1 SI F2
#
# def grafic_plt(date,x,y,title, sirute):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#
#     #etichetele pentru x si y
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#
#     #punctele pe grafic
#     ax.scatter(date[x], date[y], color='r')
#
#     #text puncte
#     for i in range(len(date)):
#         ax.text(date[x].iloc[i], date[y].iloc[i], sirute[i])
#
# grafic_plt(scoruri_factoriale_df, 'F1', 'F2','Grafic', sirute)
# plt.show()

print('inca o data')

#ANALIZA FACTORIALA
voturi = pd.read_csv('dateIN/VotBUN.csv')
coduri = pd.read_csv('dateIN/Coduri_Localitati.csv')

# index = voturi.index
# coloane_numerice = voturi.columns[2:]#numele coloanelor sub forma de index
# nume_coloane = coloane_numerice.tolist()#numele coloanelor
#
# date_numerice = voturi[coloane_numerice].values
# print(date_numerice)
#
# #CERINTA 1.TEST BARTLET
# import factor_analyzer as fact
# bartlett=fact.calculate_bartlett_sphericity(date_numerice)
# print(bartlett)#P-value este a doua valoare
#
# #CERINTA 2.KMO
# kmo = fact.calculate_kmo(date_numerice)
# tabel_kmo = pd.DataFrame(data=np.append(kmo[0],kmo[1]),
#                          index=nume_coloane + ['Total'])
# tabel_kmo.to_csv('dateOUT/tabel_kmo3.csv')
#
# #CERINTA 3.SCORURILE FACTORILE
# from sklearn.decomposition import FactorAnalysis
#
# model_factorial = FactorAnalysis(n_components=len(coloane_numerice))
# scoruri_factoriale = model_factorial.fit_transform(date_numerice)
#
# etichete_scoruri = ['F' + str(i+1)for i in range(len(coloane_numerice))]
#
# scoruri_factoriale_df = pd.DataFrame(data=scoruri_factoriale, columns=etichete_scoruri)
#
# scoruri_factoriale_df.to_csv('dateOUT/scoruri_factoriale3.csv')
#
# #CERINTA 4.GRAFICUL SCORURILOR FACTORIALE F1 SI F2
# def grafic_plt(date,x,y,index):
#     fig=plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     ax.set_title('Grafic')
#     assert isinstance(ax, plt.Axes)
#
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#
#     ax.scatter(date[x],date[y],color='r')
#
#     for i in range(len(date)):
#         ax.text(date[x].iloc[i], date[y].iloc[i], index[i])
#
# grafic_plt(scoruri_factoriale_df,'F1','F2',index)
# plt.show()

index = voturi.index
coloane_numerice = voturi.columns[2:]
nume_coloane = coloane_numerice.tolist()

date_numerice = voturi[coloane_numerice].values
print(date_numerice)

#CERINTA 1.TEST BARTLETT
import factor_analyzer as fact

bartlett = fact.calculate_bartlett_sphericity(date_numerice)
print(bartlett)

#CERINTA 2.KMO
kmo = fact.calculate_kmo(date_numerice)

tabel_kmo = pd.DataFrame(data=np.append(kmo[0],kmo[1]),
                         index=nume_coloane + ['Total'])

tabel_kmo.to_csv('dateOUT/tabel_kmo4.csv')

#CERINTA 3.SCORURI FACTORIALE
from sklearn.decomposition import FactorAnalysis

model_factorial = FactorAnalysis(n_components=len(coloane_numerice))
scoruri_factoriale = model_factorial.fit_transform(date_numerice)
etichete_factoriale = ['F'+ str(i+1) for i in range(len(coloane_numerice))]
scoruri_factoriale_df = pd.DataFrame(data=scoruri_factoriale, columns=etichete_factoriale)
scoruri_factoriale_df.to_csv('dateOUT/scoruri_factoriale4.csv')

#CERINTA 4.GRAFIC SCORURILOR FACTORIALE F1 SI F2

def grafic_plt(date,x,y,index):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Grafic')
    assert isinstance(ax, plt.Axes)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    ax.scatter(date[x], date[y], color='r')

    for i in range(len(date)):
        ax.text(date[x].iloc[i], date[y].iloc[i], index[i])

grafic_plt(scoruri_factoriale_df,'F1','F2',index)
plt.show()

