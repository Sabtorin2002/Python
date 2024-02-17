import pandas as pd
import numpy as np
import copy


#SUBIECTUL A
buget = pd.read_csv('dateIN/Buget.csv')
pop_localitati = pd.read_csv('dateIN/PopulatieLocalitati.csv')

#CERINTA1.In Cerinta1.csv sa se salveze total venituri si total chelutieli la nivel de localitate.
#Pentru fiecare se va salva Siruta, numele localitatii si cele 2 totaluri.
#[:,2:7]
# : = toate randurile , si acum pe coloane 2:7 de la coloana 2 pana la coloana 6

#axis=0 -> pe randuri
#axis=1 -> pe coloane

cerinta1_df = pd.DataFrame()
cerinta1_df['Siruta'] = buget['Siruta'].copy()
cerinta1_df['Localitate'] = buget['Localitate'].copy()
cerinta1_df['Venituri'] = buget.iloc[:,2:7].sum(axis=1)
cerinta1_df['Cheltuieli'] = buget.iloc[:,7:].sum(axis=1)

cerinta1_df.to_csv('dateOUT/Cerinta1.csv', index=False)

#CERINTA2

cerinta2 = buget.merge(pop_localitati[['Judet','Populatie','Siruta']], left_on='Siruta', right_on='Siruta')
venituri_judet=pd.DataFrame()
venituri_judet = cerinta2[['Judet','V1','V2','V3','V4','V5','Populatie']].copy()
#cand faci GROUP BY TREBUIE SA STOCHEZI REZULTATUL INTR-O NOUA VARIABILIA
#GROUP BY ITI FUTE JUDETUL IN INDEX!!! DECI grupare_judet.iloc[0:0] NU ITI INTOARCE tr!!!
grupare_judet = venituri_judet.groupby(by='Judet').sum()
grupare_judet.iloc[:,:] = grupare_judet.iloc[:,:].div(grupare_judet.iloc[:,-1], axis=0)
#DAI DROP SI SALVEZI INTR-O NOUA VARIABILIA
grupare_judet_final = grupare_judet.drop('Populatie', axis=1)
#ascending=False --> descrescator
#ascening=True --> crescator
grupare_judet_final_sortat=pd.DataFrame()
grupare_judet_final_sortat = grupare_judet_final.sort_values(by='V1', ascending=False)
print(grupare_judet_final_sortat)
grupare_judet_final_sortat.to_csv('dateOUT/Cerinta2.csv')






