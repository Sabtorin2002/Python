import pandas as pd
import numpy as np



miscare_tari = pd.read_csv('dateIN/MiseNatPopTari.csv')
coduri_tari = pd.read_csv('dateIN/CoduriTariExtins.csv')



#SUBIECT A

#CERINTA1.Salvarea tarilor care au sporul natural negativ
#RS-rata sporului natural
cerinta1 = pd.DataFrame()
cerinta1 = miscare_tari.copy()
cerinta1 = cerinta1[cerinta1['RS'] < 0]
cerinta1.to_csv('dateOUT/cerinta1.csv')

#CERINTA2.Salvarea tarilor care au sporul natural mai mic ca rata fertilitatii
cerinta2 = pd.DataFrame()
cerinta2 =miscare_tari.copy()
cerinta2 = cerinta2[cerinta2['RS']<cerinta2['FR']]
print(cerinta2)


#CERINTA3.Salvarea valorilor medii pe indicatori pe continent
cerinta3 = pd.DataFrame()
cerinta3 = miscare_tari.drop('Country_Name',axis=1).copy()
cerinta3 = cerinta3.merge(coduri_tari[['Continent','Country_Number']],left_on='Country_Number',right_on='Country_Number')\
    .drop('Country_Number',axis=1)
cerinta3 = cerinta3.drop('Three_Letter_Country_Code', axis=1)
cerinta3 = cerinta3.groupby(by='Continent').mean()
cerinta3.to_csv('dateOUT/cerinta2.csv')
print(cerinta3)


# #SUBIECT B
#
# #ANALIZA COMPONENTELOR PRINCIPALE
#
# #CERINTA1. VARIANTELE COMPONENTELOR PRINCIPALE. VARIANTELE VOR FI AFISATE LA CONSOLA
#
# populatie = pd.read_csv('dateIN/MiseNatPopTari.csv')
# coduri = pd.read_csv('dateIN/CoduriTariExtins.csv')
#
# #Pasul 1: se extrag datele numerice(standardizare)
# nume_variabile = list(populatie.columns)[1:]
# print(nume_variabile)
#
# #lista cu variabile care contin codurile selectate ->fara Country_Number
# coloane_numerice = nume_variabile[1:]
#
# #mean() -> calculeaza media pentru fiecare coloana
# #std() -> calculeaza deviatia standard pentru fiecare coloana
#
# date_standardizate =(populatie[coloane_numerice] - populatie[coloane_numerice].mean()) / populatie[coloane_numerice].std()
#
# #Pasul 2: Crearea obiectului PCA si ajustarea la datele standardizate
# model = PCA()
# model.fit(date_standardizate)#se face fit-ul datelor standardizate
#
# #Pasul 3:Afisarea VARIANTELOR COMPONENTELOR
# variance = model.explained_variance_ratio_
#
# #VARIANTA CUMULATA(cum+sum de variance)
# varianta_cumulata=np.cumsum(variance)
#
# #PROCENTUL DE VARIANTA(variance * 100)
# procentul_de_varianta = variance*100
#
# #PROCENTUL CUMULAT(varianta_cumulata * 100)
# procentul_cumulat = varianta_cumulata * 100
#
# rezultate_pca = pd.DataFrame({
#     'Varianta Componentelor Prinicipale':variance,
#     'Varianta Cumulata':varianta_cumulata,
#     'Procentul de Varianta':procentul_de_varianta,
#     'Procentul Cumulat de Varianta Explicata':procentul_cumulat
# })
#
# print(rezultate_pca)
# rezultate_pca.to_csv('dateOUT/VarianteleComponentelorPrincipale.csv')


# #CERINTA2. SCORURILE ASOCIATE INSTANTELOR
# #Se vor afisa in csv.
# nr_componente_acp = len(variance)
#
# #Etichetele vor fi retinute in lista sub forma: C1, C2, C3, etc
#
# etichete_componente = ['C'+ str(i+1) for i in range(nr_componente_acp)]
# #print(etichete_componente)
#
# #Scores - matricea scorurilor calculate
# # .dot realizeaza inmultirea
#
# #prin aplicarea .T obtinem matricea transpusa a componentelor principale
# SCORURILE SUNT INMULTIREA DINTRE DATELE STANDARDIZATE SI MODELUL DE COMPONENTE TRASNPUS
# scores = np.dot(date_standardizate, model.components_.T)
#
# #index = nume_instanta specifica numele instatelor ce vor fi folosite ca index-uri ale DataFrame-ului
# #columns = etichete_componente specifica etichetele create anterior pentru componente vor fi folosite ca si nume de coloane
#
# nume_instante = list(populatie['Country_Name'].values)
# t_scores = pd.DataFrame(scores, index=nume_instante, columns=etichete_componente)
# t_scores.to_csv('dateOUT/Scoruri.csv')
# print(t_scores)
#
# from matplotlib import pyplot as plt
# #CERINTA 3. GRAFICUL SCORURILOR IN PRIMELE 2 AXE PRINCIPALE
#
# #crearea unui DataFrame cu componentele principale si afisarea acestuia
#
# # t_components = pd.DataFrame(data=model.components_, index=etichete_componente, columns=coloane_numerice)
# # print(t_components)
# #
# # #Grafic
#
# #
# # def grafic_plot(t,x,y,title='Componente'):
# #     #Crearea unei figuri subplot
# #     fig = plt.figure(figsize=(11,8))
# #     ax = fig.add_subplot(1,1,1)
##      assert isinstance(ax, plt.Axes)
# #
# #     #Setarea etichetelor pentru axa x si y
# #     ax.set_xlabel(x, fontsize=12, color='b')
# #     ax.set_ylabel(y, fontsize=12, color='b')
# #
# #     #colorare puncte
# #     ax.scatter(t.loc[x], t.loc[y], color='r')
# #
# #     #eticheta pentru un punct
# #     for i in coloane_numerice:
# #         print(i)
# #         ax.text(t.loc[x].loc[i], t.loc[y].loc[i],i)
# #
# # grafic_plot(t_components, 'C1', 'C2')
# # plt.show()
#
#
# t_components = pd.DataFrame(data=model.components_, index=etichete_componente, columns=coloane_numerice)
#
# #Grafic
# def grafic_plot(t,x,y,title):
#     #Crearea unui figuri subplot
#     fig = plt.figure(title, figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#
#     #Setarea etichetelor pentru axa x si y
#     ax.set_xlabel(x, fontsize=12, color='b')
#     ax.set_ylabel(y, fontsize=12, color='b')
#
#     #Realizarea unui scatter plot intre x si y, cu punctele colorate in rosu
#     ax.scatter(t.loc[x], t.loc[y], color='r')
#
#     #Eticheta pentru un punct
#     for i in coloane_numerice:
#         ax.text(t.loc[x].loc[i], t.loc[y].loc[i], i)
#
# grafic_plot(t_components,'C1','C2','Componente')
# plt.show()
#
# #CERINTA 4. Corelograma corelatiilor din variabilele observate si componentele principale
# corelatii = np.corrcoef(date_standardizate.T, scores.T)[:len(coloane_numerice), len(coloane_numerice):]
# corelatii_df = pd.DataFrame(data=corelatii, columns=etichete_componente)
#
# #PLASAREA CORELATIILOR INTR-UN HEATMAP
#
# plt.figure(figsize=(11,8))
# plt.imshow(corelatii_df)
# plt.colorbar()
# plt.xticks(range(nr_componente_acp), etichete_componente)
# plt.yticks(range(len(coloane_numerice)),coloane_numerice)
# plt.title('Corelograma corelatiilor')
# plt.show()

#SUBIECT B
#ANALIZA COMPONENTELOR PRINCIPALE

# populatie=pd.read_csv('dateIN/MiseNatPopTari.csv')
# coduri = pd.read_csv('dateIN/CoduriTariExtins.csv')

# #CERINTA 1. VARIANTELE COMPONENTELOR PRINCIPALE
#
# nume_variabile = list(populatie)[1:]
# coloane_numerice = nume_variabile[2:]
# print(coloane_numerice)
#
# #INVERSUL LUI COEFICIENT DE VARIATIE CARE COEF.DE VARIATIE = std/mean
# date_standardizate = (populatie[coloane_numerice] - populatie[coloane_numerice].mean())/populatie[coloane_numerice].std()
#
# model = PCA()
# model.fit(date_standardizate)
#
# #TABEL
# variance = model.explained_variance_ratio_
#
# varianta_cumulata = np.cumsum(variance)
#
# procentul_de_varianta = variance*100
#
# procentul_cumulat = varianta_cumulata * 100
#
# rezultate_pca = pd.DataFrame({
#     'Varianta Componentelor Principale':variance,
#     'Varianta Cumulata':varianta_cumulata,
#     'Procentul de Varianta':procentul_de_varianta,
#     'Procentul cumulat':procentul_cumulat
# })
# print(rezultate_pca)
# rezultate_pca.to_csv('dateOUT/VARIANTA2.csv',index=False)
#
#
# #CERINTA 2. SCORURILE ASOCIATE INSTANTELOR
#
# nr_componente = len(variance)
#
# #index: nume_instante: franta, italia
# #columns:'C1','C2'
# nume_instante = list(populatie['Country_Name'].values)
# etichete_componente =['C'+str(i+1) for i in range(nr_componente)]
#
# #SCORURILE SUNT INMULTIREA DINTRE DATELE STANDARDIZATE SI MODELUL DE COMPONENTE TRANSPUS
# scores = np.dot(date_standardizate, model.components_.T)
#
# t_scores = pd.DataFrame(scores, columns=etichete_componente)
# t_scores.to_csv('dateOUT/Scoruri2.csv', index=False)
#
# #CERINTA 3. REALIZAREA GRAFICULUI SCORURILOR IN PRIMELE 2 AXE PRINCIPALE
# #index = 'C1','C2'..
# #columns = 'RS' 'FR' 'LM'..
# t_components = pd.DataFrame(data=model.components_, index=etichete_componente, columns=coloane_numerice)
#
# #Grafic
# from matplotlib import pyplot as plt
#
# def grafic_plt(t,x,y,title='Componente'):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#
#     #Setarea etichetelor pentru axa x si y
#     ax.set_xlabel(x, fontsize=12, color='b')
#     ax.set_ylabel(y, fontsize=12, color='b')
#
#     #Realizarea unui scatter plot intre x si y, punctele de culoare rosie
#     ax.scatter(t.loc[x], t.loc[y], color='r')
#
#     #Eticheta pentru un punct
#     for i in coloane_numerice:
#         ax.text(t.loc[x].loc[i], t.loc[y].loc[i], i)
#
# grafic_plt(t_components, 'C1', 'C2')
# plt.show()
#
#
# #CERINTA 4. Corelograma corelatiilor dintre variabilele observate si componentele principale
# #corelatiile dintre datele standadizate transpuse si scorurile transpuse
# corelatii = np.corrcoef(date_standardizate.T, scores.T)[:len(coloane_numerice), len(coloane_numerice):]
#
# corelatii_df = pd.DataFrame(data=corelatii)
# plt.figure(figsize=(11,8))
# plt.imshow(corelatii_df)
# plt.colorbar()
# plt.xticks(range(nr_componente),etichete_componente)
# plt.yticks(range(len(coloane_numerice)),coloane_numerice)
# plt.title('Corelograma corelatiilor')
# plt.show()


print('inca o data')

# #CERINTA1. VARIANTA MODELELOR PRINCIPALE
#
# #datele relevante
# coloane_numerice = populatie.columns[3:]
# print(coloane_numerice)
#
# date_standardizate = (populatie[coloane_numerice] - populatie[coloane_numerice].mean()) / populatie[coloane_numerice].std()
#
# model = PCA()
# model.fit(date_standardizate)
#
# variance = model.explained_variance_ratio_
#
# varianta_cumulata = np.cumsum(variance)
#
# procent_varianta = variance*100
#
# procent_cumulat = varianta_cumulata*100
#
# rezultate_pca = pd.DataFrame({
#     'Varianta Componentelor Principale':variance,
#     'Variata cumulata':varianta_cumulata,
#     'Procentul de Varianta':procent_varianta,
#     'Procent cumulat':procent_cumulat
# })
#
# rezultate_pca.to_csv('dateOUT/VARIANTA3.csv',index=False)
#
# #CERINTA 2.SCORURILE ASOCIATE INSTANTELOR
#
# nr_componente = len(variance)
#
# etichete_componente = ['C'+ str(i+1) for i in range(nr_componente)]
#
# #SCORURILE REPREZINTA INMULTIREA DINTRE DATELE STANDARDIZATE SI MODEL.COMPONENTS TRANSPUS
#
# scores = np.dot(date_standardizate, model.components_.T)
#
# t_scores = pd.DataFrame(scores, columns=etichete_componente)
# t_scores.to_csv('dateOUT/Scoruri3.csv', index=False)
#
#
# #CERINTA 3. REALIZAREA GRAFICULUI SCORURILOR IN PRIMELE 2 AXE PRINCIPALE
#
# t_components = pd.DataFrame(data=model.components_, index=etichete_componente, columns=coloane_numerice)
# print(t_components)
#
# from matplotlib import pyplot as plt
#
# def grafic_plot(t,x,y,titlu='Componente'):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#
#     #punctele de pe grafic
#     ax.scatter(t.loc[x], t.loc[y], color='r')
#
#     #textul pentru un punct
#     for i in coloane_numerice:
#         ax.text(t.loc[x].loc[i], t.loc[y].loc[i], i)
#
# grafic_plot(t_components,'C1','C2')
# plt.show()
#
#
# #CERINTA 4. CORELOGRAMA CORELATIILOR DINTRE VARIABILE OBS SI COMPONENTELE PRINCIPALE
# #CORELATII = CORRCOEFF de DATE STANDARDIZATE TRANSPUS SI SCORURILE TRANSPUSE
# corelatii = np.corrcoef(date_standardizate.T, scores.T)[:len(coloane_numerice), len(coloane_numerice):]
# corelatii_df = pd.DataFrame(data=corelatii)
#
# def corelograma(corelatii):
#     plt.figure(figsize=(11,8))
#     plt.imshow(corelatii)
#     plt.xticks(range(nr_componente), etichete_componente)
#     plt.yticks(range(len(coloane_numerice)),coloane_numerice)
#     plt.title('Corelograma corelatiilor')
#
# corelograma(corelatii_df)
# plt.show()

print('inca o data')

from sklearn.decomposition import PCA
populatie = pd.read_csv('dateIN/MiseNatPopTari.csv')
coduri = pd.read_csv('dateIN/CoduriTariExtins.csv')

# #CERINTA 1.VARIANTA COMPONENTELOR PRINCIPALE
# coloane_numerice = populatie.columns[3:]
# print(coloane_numerice)
#
# date_standardizate = (populatie[coloane_numerice] - populatie[coloane_numerice].mean())/populatie[coloane_numerice].std()
#
# model = PCA()
# model.fit(date_standardizate)
#
# variance = model.explained_variance_ratio_
#
# varianta_cumulata = np.cumsum(variance)
#
# procent_varianta = variance*100
#
# procent_varianta_cumulata = varianta_cumulata*100
#
# rezultate_pca = pd.DataFrame({
#     'Varianta':variance,
#     'Varianta cumulata':varianta_cumulata,
#     'Procent varianta':procent_varianta,
#     'Procent varianta cumulata':procent_varianta_cumulata
# })
#
# rezultate_pca.to_csv('dateOUT/VARIANTA4.csv')
#
# #CERINTA 2.SCORURILE ASOCIATE INSTANTELOR
#
# nr_componente = len(variance)
#
# etichete_componente = ['C'+ str(i+1) for i in range(nr_componente)]
#
# #SCORURILE LA ACP REPREZINTA INMULTIREA DINTRE DATELE STANDARDIZATE SI MODEL.COMPONENTS TRANSPUS
#
# scores = np.dot(date_standardizate, model.components_.T)
#
# scores_df = pd.DataFrame(data=scores, columns=etichete_componente)
#
# scores_df.to_csv('dateOUT/Scoruri4.csv')
#
# #CERINTA 3. REALIZAREA GRAFICULUI SCORURILOR IN PRIMELE 2 AXE PRINCIPALE
#
# components_df = pd.DataFrame(data=model.components_, index=etichete_componente, columns=coloane_numerice)
#
# from matplotlib import pyplot as plt
#
# def graifc_plot(date,x,y,titlu):
#     fig = plt.figure(figsize=(11,8))
#     ax = fig.add_subplot(1,1,1)
#     assert isinstance(ax, plt.Axes)
#
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#
#     ax.scatter(date.loc[x], date.loc[y], color='r')
#     for i in coloane_numerice:
#         ax.text(date.loc[x].loc[i], date.loc[y].loc[i], i)
#
# graifc_plot(components_df,'C1','C2', 'Graficul scorurilor')
# plt.show()

#CERINTA 1.VARIANTA COMPONENTELOR PRINCIPALE
print('inca o data')
coloane_numerice = populatie.columns[3:]
print(coloane_numerice)

date_standardizate = (populatie[coloane_numerice] - populatie[coloane_numerice].mean())/populatie[coloane_numerice].std()

model = PCA()
model.fit(date_standardizate)

variance = model.explained_variance_ratio_

varianta_cumulata = np.cumsum(variance)

procent_varianta = variance*100

procent_cumulat = varianta_cumulata *100

rezultate_pca = pd.DataFrame({
    'Varianta':variance,
    'Varianta cumulata':varianta_cumulata,
    'Procent varinata':procent_varianta,
    'Procent cumulat':procent_cumulat
})


rezultate_pca.to_csv('dateOUT/VARIANTA5.csv')

#CERINTA 2.SCORURILE ASOCIATE INSTANTELOR

nr_componente = len(variance)
etichete_componente = ['C'+str(i+1) for i in range(nr_componente)]

scores = np.dot(date_standardizate, model.components_.T)

scores_df = pd.DataFrame(data=scores, columns=etichete_componente)

scores_df.to_csv('dateOUT/Scoruri5.csv')

#CERINTA 3.REALIZAREA GRAFICULUI PENTRU PRIMELE 2 AXE PRINCIPALE
from matplotlib import pyplot as plt
componente_df = pd.DataFrame(data= model.components_, index=etichete_componente, columns=coloane_numerice)
print(componente_df)

def grafic_plot(date, x, y, title):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    ax.scatter(date.loc[x], date.loc[y], color='r')

    for i in coloane_numerice:
        ax.text(date.loc[x].loc[i], date.loc[y].loc[i], i)

grafic_plot(componente_df, 'C1', 'C2', 'Grafic')
plt.show()