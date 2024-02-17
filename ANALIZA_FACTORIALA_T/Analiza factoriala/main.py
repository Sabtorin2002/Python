#Import biblioteci necesare

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import factor_analyzer as fact
from sklearn.preprocessing import StandardScaler
import factor_analyzer
import factor_analyzer.factor_analyzer

#Subiect A

#Cerinta 1
# Salvarea in fisierul cerinta1.csv a categoriei de alegatori pentru care
# s-a inregistrat cel mai mic procent de prezenta

# Se salveaza :
# - codul Siruta
# - numele localitatii
# - categoria de alegatori

# Pasul 1: citire dat din csv
voturi = pd.read_csv('dataIN/VotBUN.csv',index_col=0)
coduri_localitati = pd.read_csv('dataIN/Coduri_Localitati.csv',index_col=0)

# Pasul 2: Selectia coloanelor incepand de la pozitia 1 (fara codul Siruta ) din DataFrame-ul "voturi"
variabile = voturi.columns[1:]
sirute = voturi.index # start = 0, stop  = 11, step = 1

# Pasul 3: Definim o functie care calculeaza cel mai mic procent de prezenta

#Functia returneaza localitatea si categoria cu cel mai mic procent de vot intr-o linie
def vot_minim(linie):
    # 1.Extrag coloana localitate din linie
    localitate = linie['Localitate']

    # 2. Extrag valorile de voturi (fara coloana 'Localitate')
    valori = linie[1:]

    # 3.Gaseste indexul categoriei cu cel mai mic vot
    index_min = np.argmin(valori)

    # 4.Creeaza o serie cu datele necesare
    data = [localitate,valori.index[index_min]]

    return pd.Series(data=data,index=['Localitate','Categorie'])

# Pasul 4: Aplicarea functiei 'vot_minim' pe fiecare linie a DataFrame-ului voturi
df_cerinta_1 = voturi.apply(func=vot_minim,axis=1) #axis=1 -> linie

#Pasul 5: salvarea in csv
df_cerinta_1.to_csv('dataOUT/cerinta1.csv')
#print(df_cerinta_1)

#Cerinta 2

#Salvarea in fisierul cerinta2.csv a valorilor medii la nivel de judet
#Se va salva indicativul judetului si valorile medii pentru fiecare judet

data_merge = pd.merge(voturi,coduri_localitati,on='Siruta')

# Grupez datele dupa judet
data_groupedby_judet = data_merge.groupby(by='Judet').mean(numeric_only=True)

#Salvarea datelor in csv
data_groupedby_judet.to_csv('dataOUT/cerinta2.csv')
print(data_groupedby_judet)

#Subiectul B

#Cerinta 1
# TESTUL BARTLETT de relevanta -> rezultatul = p-value
# - se calculeaza pragul de semnificatie asociat respingerii/accepatrii testului (p-value)

# Pasul 1: Extrag doar variabilele relevante
# ( nr de voturi din fiecare categorie) din DataFrame-ul "voturi" si
# le convertesc in matrice NumPy

x = voturi[variabile].values
print(x)

# Pasul 2: Calculez sfericitatea utilizand testul BARTLETT
# Matricea de covarianta trebuie sa fie sferica -> analiza factoriala este potrivita

bartlett = fact.calculate_bartlett_sphericity(x)

# Pasul 3: Afisare p-value asociat testului BARTLETT
print("P-value", bartlett[1])

#CERINTA2.Scorurile factoriale si salvarea in csv
# Alternative using sklearn for factor analysis
from sklearn.decomposition import FactorAnalysis

# Pasul 1: Creez un obiect FactorAnalysis cu numarul de factori specificat
model_fa = FactorAnalysis(n_components=len(variabile))

# Pasul 2: Antrenez modelul FactorAnalysis pe datele extrase
scoruri_fa = model_fa.fit_transform(x)

# Pasul 3: Generez etichete pentru factori ( F1, F2, etc. )
etichete_factori_fa = ["F" + str(i + 1) for i in range(len(variabile))]

# Pasul 4: Construiesc un DataFrame cu scorurile factoriale si le salvez intr-un fisier
df_scoruri_fa = pd.DataFrame(data=scoruri_fa, index=sirute, columns=etichete_factori_fa)

df_scoruri_fa.to_csv('dataOUT/f.csv')
print(df_scoruri_fa)

#Cerinta 3

#Graficul scorurilor factoriale pentru PRIMII 2 FACTORI: F1 si F2

def plot_componente(x,var_x,var_y,sirute):
    # 1. Creez o figura cu un singur subplot
    figura = plt.figure('Graficul scorurilor factoriale pentru primii 2 factori',figsize=(9,9))
    ax = figura.add_subplot(1,1,1)

    # 2. Setez etichetele pentru axa X si axa Y
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)

    # 3. Creez un scatter plot ( DIGRAMA DE DISPERSIE ) folosind variabilele
    # specificate pentru axa X si axa Y
    ax.scatter(x[var_x],x[var_y],color='r')

    # 4. Adaug etichete pentru fiecare punct din scatter plot
    for i in range(len(x)):
        ax.text(x[var_x].iloc[i],x[var_y].iloc[i],sirute[i])

# Apelarea functiei plot_componente
plot_componente(df_scoruri_fa, 'F1','F2',sirute)

# Afisarea graficului
plt.show()


#CERINTA 4.KMO
kmo = fact.calculate_kmo(x)
print("KMO", kmo)
nume_coloane = variabile.tolist()

tabel_kmo = pd.DataFrame(data={"Index KMO": np.append(kmo[0], kmo[1])},
                        index=nume_coloane + ["Total"])
tabel_kmo.to_csv('dataOUT/KMO.csv')






