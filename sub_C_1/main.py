import pandas as pd
import numpy as np

#CERINTA1.Sa se salveze in cerinta1.csv a mediei consumului pe cei 5 ani pentru fiecare tara.Se va salva pentru fiecare tara
#codul de tara si media

alcool = pd.read_csv('dateIN/alchol.csv')
coduri_tari_extins = pd.read_csv('dateIN/CoduriTariExtins.csv')

output1 = pd.DataFrame()
output1['Code'] = alcool['Code'].copy()

output1['Medie'] = alcool.iloc[:,2:].mean(axis=1)
print(output1)
output1.to_csv('dateOUT/cerinta1.csv', index=False)

#CERIINTA2
#GROUP BY = NU VREA STRING-URI
cerinta2 = pd.DataFrame()
cerinta2 = alcool.drop('Țara', axis=1).copy()
cerinta2 = cerinta2.merge(coduri_tari_extins[['Continent', 'Code']], left_on='Code', right_on='Code').drop('Code', axis=1)
#Continentul devine index
cerinta2 = cerinta2.groupby(by='Continent').mean()
cerinta2['Cifra maxima'] = cerinta2.max(axis=1)
cerinta2['Anul'] = cerinta2.idxmax(axis=1)
print(cerinta2)
cerinta2 = cerinta2[['Cifra maxima', 'Anul']]
cerinta2 = cerinta2.drop('Cifra maxima', axis=1)
cerinta2.to_csv('dateOUT/cerinta2.csv')


#CERINTA3.
cerinta3 = pd.DataFrame()
cerinta3 = alcool.copy()
cerinta3 = cerinta3.merge(coduri_tari_extins[['Continent','Code']],left_on='Code', right_on='Code').drop('Code', axis=1)
cerinta3.set_index('Țara',inplace=True)
cerinta3 = cerinta3.groupby(by='Continent').idxmax()
print(cerinta3)
cerinta3.to_csv('dateOUT/cerinta3.csv')
