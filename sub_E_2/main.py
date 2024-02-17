import pandas as pd
import numpy as np

industrie = pd.read_csv('dateIN/Industrie.csv')
populatie_localitati = pd.read_csv('dateIN/PopulatieLocalitati.csv')

#CERINTA1.

cerinta1 = pd.DataFrame()
cerinta1 = industrie.copy()
cerinta1 = cerinta1.merge(populatie_localitati[['Populatie', 'Siruta']], left_on='Siruta', right_on='Siruta')
cerinta1.iloc[:,2:-1] = cerinta1.iloc[:,2:-1].div(cerinta1.iloc[:,-1],axis=0)
cerinta1 = cerinta1.drop('Populatie',axis=1)
cerinta1.to_csv('dateOUT/cerinta1.csv',index=False)
print(cerinta1)

#CERINTA2

cerinta2 = pd.DataFrame()
cerinta2 = industrie.drop('Localitate', axis=1).copy()
cerinta2 = cerinta2.merge(populatie_localitati[['Judet', 'Siruta']], left_on='Siruta', right_on='Siruta').drop('Siruta', axis=1)
#cerinta2 = cerinta2.merge(populatie_localitati[['Populatie', 'Siruta']], left_on='Siruta', right_on='Siruta').drop('Siruta', axis=1)
#JUDET DEVINE INDEX-UL, GROUPBY E FARA axis
cerinta2 = cerinta2.groupby(by='Judet').sum()
cerinta2['Cifra Afaceri'] = cerinta2.iloc[:,:].max(axis=1)
cerinta2['Activitate'] = cerinta2.iloc[:,:].idxmax(axis=1)
cerinta2 = cerinta2[['Activitate', 'Cifra Afaceri']]
cerinta2.to_csv('dateOUT/cerinta2.csv')
print(cerinta2)