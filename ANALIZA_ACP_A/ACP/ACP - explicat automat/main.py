#Importare biblioteci necesare
import pandas as pd
import numpy as np
import copy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#Subiect 1

#Citirea datelor din fisiere CSV
populatie = pd.read_csv('dataIN/MiseNatPopTari.csv')
coduri = pd.read_csv('dataIN/CoduriTariExtins.csv')

#Cerinta 1
# Salvarea in fisierul cerinta1.csv a tarilor in care
# rata sporului natural este negativ (RS)

# Crearea unei copii a DataFrame-ului 'populatie'
copy_populatie = copy.copy(populatie)

# Selectarea coloanelor 'Country_Name' si 'RS' din DataFrame-ul copy_populatie
populatie_filtrata = copy_populatie[['Country_Name','RS']]

#Filtrarea randurilor pentru care 'RS' este mai mic decat 0 si
# salvarea rezultatului in cerinta1.csv
cerinta1 = populatie_filtrata[populatie_filtrata['RS']<0]

#aceasta functie este folosita pentru a elimina coloana RS nefiltrata
#si o inlocuieste cu cea filtrata in care RS < 0
cerinta1.reset_index(drop=True)

#Scriu in csv rezultatele
cerinta1.to_csv('dataOUT/Cerinta1.csv', index = False)

#Cerinta 2

#Salvarea in fişierul cerinta2.csv a valorilor medii pentru indicatorii de
# mai sus, la nivel de continent
# Continente -> Europa, Asia
# Indifcatori -> RS, FR, IM, MMR, LE, LEM, LEF

# Pasul 1 - unirea celor 2 DataFrame-uri
# on = Country_Number deoarece este coloana comuna dintre fisiere
data_merge = pd.merge(populatie,coduri, on='Country_Number')

# Pasul 2: Selectare coloane specifice + calcul medie pe continent
# Creez o noua matrice in care selectez doar coloanele de care am nevoie
indicatori_si_continente = data_merge[['RS','FR','LM','MMR','LE','LEM','LEF','Continent']]

indicatori = ['RS','FR','LM','MMR','LE','LEM','LEF']

# Pasul 3: dataframe nou creat in care fiecare rand este un continent si
# valorile corespunzatoare pe coloane reprezinta media fiecarui indicator in acel continent

cerinta2 = indicatori_si_continente[indicatori + ['Continent']].groupby(by='Continent').mean()

# Pasul 4: afisare in csv
cerinta2.to_csv('dataOUT/Cerinta2.csv')


#Subiect 2
#Sa se efectueze analiza in componente principale standardizată şi
# să se furnizeze următoarele rezultate:

# Cerinta 1
# Variantele componentelor principale.
# Variantele vor fi afisate la consola.

# Pasul 1: Standardizarea datelor numerice ( centrare si scalare )
nume_variabile = list(populatie.columns)[1:]

#lista cu variabile care contin codurile selectate -> fara Country_Number
coloane_numerice = nume_variabile[1:]

# mean() -> calculeaza media pentru fiecare coloana

# std ()-> calculeaza deviatia standard pentru fiecare coloana

# (populatie[numeric_var] - populatie[numeric_var].mean())
# -> calculeaza diferenta intre fiecare valoare si media corespunzatoare pe coloana respectiva
# -> adica din fiecare numar scad media din coloana corespunzatoare ei

# /(populatie[numeric_var].std())
# -> se imparte fiecare diferenta la deviatia standard corespunzatoare din variabila

date_standardizate = (populatie[coloane_numerice] - populatie[coloane_numerice].mean()) / populatie[coloane_numerice].std()


# Pasul 2: Crearea obiectului PCA si ajustarea la datele standardizate
model = PCA()
model.fit(date_standardizate)

# Pasul 3: Afisare VARIANTELE COMPONENTELOR
variance = model.explained_variance_ratio_
# print("Variantele componentelor principale: ")
# print(variance)

# ------------- CALCUL NOU PENTRU
# VARIANTA CUMULATA
varianta_cumulata = np.cumsum(variance)

# PROCENTUL DE VARIANTA
procentul_de_varianta = variance * 100

# PROCENTUL CUMULAT
procntul_cumulat  = varianta_cumulata * 100

# Crearea unui DataFrame pentru stocarea rezultatelor
rezultate_pca = pd.DataFrame({
    'Varianta Componentelor Principale': variance,
    'Varianta Cumulata': varianta_cumulata,
    'Procentul de Varianta Explicata': procentul_de_varianta,
    'Procentul Cumulat de Varianta Explicata': procntul_cumulat
})

# Salvarea în fișierul CSV
rezultate_pca.to_csv('dataOUT/cerinta1-versiunea2.csv', index=False)

# Cerinta 2
#Scorurile asociate instantelor.
#Scorurile vor fi salvate in fisierul scoruri.csv

nr_componente_acp = len(variance)

# Etichetele vor fi retinute in lista sub forma : C1, C2, C3 etc.
etichete_componente = ["C"+str(i+1) for i in range(nr_componente_acp)]
print(etichete_componente)

# scores - matricea scorurilor calculate
# .dot realizeaza inmultirea
scores = np.dot(date_standardizate,model.components_.T)

# index = nume_instante specifica ca numele instantelor vor fi folosite ca indexuri ale DataFrame-ului
# columns = etichete_componente specifica ca etichetele create anterior pentru componente vor fi folosite ca si nume de coloane

nume_instante = list(populatie['Country_Name'].values)
t_scores = pd.DataFrame(scores,index= nume_instante, columns=etichete_componente)
t_scores.to_csv('dataOUT/scoruri.csv',index=False)
print(t_scores) # nu se salveaza in fisier, dar apare rezultatul corect in consola

#Cerinta 3
#Realizarea graficului scorurilor în primele două axe principale

#Crearea unui DataFrame cu componentele principale si afisarea acestuia
t_components = pd.DataFrame(data=model.components_,index=etichete_componente,columns=coloane_numerice)
print(t_components)

#Grafic
from matplotlib import pyplot as plt

def plot_componente(t,x,y,title="Componente"):
    #Crearea unei figuri cu subplot
    figura = plt.figure(title,figsize=(9,9))
    ax = figura.add_subplot(1,1,1)

    #Setarea etichetelor pentru axa x si y
    ax.set_xlabel(x, fontdict={"fontsize": 12,"color": "b"})
    ax.set_ylabel(y,fontdict={"fontsize": 12, "color":"b"})

    #Afisarea dataframe-ului cu componente
    print(t)

    #Realizarea unui scatter plot intre x si y , cu punctele colorate in rosu
    ax.scatter(t.loc[x], t.loc[y], color='r')

    #Eticheta pentru un punct
    for i in coloane_numerice:
        print(i)
        ax.text(t.loc[x].loc[i],t.loc[y].loc[i],i)

plot_componente(t_components,'C1','C2')
plt.show()


# CERINTE ADAUGATE

# Cerinta 4
# Plotul de varianță cu marcarea grafică a criteriilor de selecție a componentelor semnificative

# Plotul varianței explicate pentru fiecare componentă principală
plt.figure(figsize=(10, 6))
plt.bar(range(1, nr_componente_acp + 1), procentul_de_varianta, alpha=0.7, align='center', label='Varianță explicată')
plt.step(range(1, nr_componente_acp + 1), varianta_cumulata, where='mid', label='Varianță cumulată')

# Linie pentru pragul de 90% varianță explicată
plt.axhline(y=90, color='r', linestyle='--', label='90% Varianță explicată')

plt.xlabel('Componenta Principală')
plt.ylabel('Procentul de Varianță Explicată / Cumulată')
plt.title('Varianța și Varianța Cumulată a Componentelor Principale')
plt.legend(loc='best')
plt.show()


# Cerinta 5
# Corelograma corelațiilor dintre variabilele observate și componentele principale

# Calculul corelațiilor dintre variabilele observate și componentele principale
corelatii = np.corrcoef(date_standardizate.T, scores.T)[:len(coloane_numerice), len(coloane_numerice):]

# Crearea unui DataFrame pentru stocarea corelațiilor
corelatii_df = pd.DataFrame(data=corelatii, index=coloane_numerice, columns=etichete_componente)

# Plasarea corelațiilor pe un heatmap
plt.figure(figsize=(12, 8))
plt.imshow(corelatii_df, cmap='coolwarm', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(nr_componente_acp), etichete_componente)
plt.yticks(range(len(coloane_numerice)), coloane_numerice)
plt.title('Corelograma corelațiilor dintre variabilele observate și componentele principale')
plt.show()