import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import factor_analyzer as fact

vot = pd.read_csv('dateIN/Vot.csv', index_col=0)
coduri_localitati = pd.read_csv('dateIN/Coduri_localitati.csv',index_col=0)

#CERINTA1.Catergoria de alegatori pentru care s-a inregistrat cel mai mic procent.

# cerinta1 = pd.DataFrame()
# cerinta1['Siruta'] = voturi['Siruta'].copy()
# cerinta1['Localitate'] = voturi['Localitate'].copy()
# cerinta1['Categorie'] = voturi.iloc[:, 2:].idxmin(axis=1)
# cerinta1.to_csv('dateOUT/cerinta1.csv',index=False)
#
# #CERINTA2.Valorile medii la nivel de judet.Se va salva indicativul judetului
# cerinta2 = pd.DataFrame()
# cerinta2 = voturi.drop('Localitate', axis=1).copy()
# cerinta2 = cerinta2.merge(coduri_localitati[['Judet', 'Siruta']], left_on='Siruta', right_on='Siruta').drop('Siruta',axis=1)
# cerinta2 = cerinta2.groupby(by='Judet').mean()
# cerinta2.to_csv('dateOUT/cerinta2.csv')
# print(cerinta2)
#
#
# cerinta3 = pd.DataFrame()
# cerinta3 = voturi.drop('Siruta', axis=1).copy()
# cerinta3 = cerinta3.merge(coduri_localitati[['Judet','Localitate']], left_on='Localitate', right_on='Localitate').drop('Localitate',axis=1)
# cerinta3 = cerinta3.groupby(by='Judet').mean()
# cerinta3.to_csv('dateOUT/cerinta3.csv')
# print(cerinta3)

print('--------------------------')
#SUBIECTUL B
#ANALIZA FACTORIALA






