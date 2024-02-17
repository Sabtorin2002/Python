import pandas as pd
import numpy as np

industrie = pd.read_csv('dateIN/Industrie.csv')
populatie_locatlitati = pd.read_csv('dateIN/PopulatieLocalitati.csv')

#CERINTA1. Cifra de afaceri pe locuitor pentru fiecare activitate, la nivel de localitate

cerinta1 = pd.DataFrame()
cerinta1 = industrie.copy()
cerinta1 = cerinta1.merge(populatie_locatlitati[['Populatie','Siruta']],left_on='Siruta', right_on='Siruta')
cerinta1.iloc[:,2:-1] = cerinta1.iloc[:,2:-1].div(cerinta1.iloc[:,-1],axis=0)
cerinta1_sortat = cerinta1.sort_values(by='Populatie', ascending=False)
cerinta1.to_csv('dateOUT/cerinta1.csv',index=False)
print(cerinta1_sortat)

#CERINTA2
cerinta2 = pd.DataFrame()
cerinta2 = industrie.drop('Localitate',axis=1).copy()
cerinta2 = cerinta2.merge(populatie_locatlitati[['Judet', 'Siruta']],left_on='Siruta', right_on='Siruta').drop('Siruta',axis=1)
cerinta2 = cerinta2.groupby(by='Judet').sum()
cerinta2['Cifra Afaceri'] = cerinta2.iloc[:,:].max(axis=1)
cerinta2['Activitate'] = cerinta2.iloc[:,:].idxmax(axis=1)
cerinta2 = cerinta2[['Activitate', 'Cifra Afaceri']]
cerinta2.to_csv('dateOUT/cerinta2.csv')
print(cerinta2)

#CERINTA3.Sa se afiseze suma tuturor industriilor la nivel de localitate
cerinta3 = pd.DataFrame()
cerinta3['Siruta'] = industrie['Siruta'].copy()
cerinta3['Localitate'] = industrie['Localitate'].copy()
cerinta3['ValoareAdaugate'] = industrie.iloc[:,2:].sum(axis=1)
cerinta3.to_csv('dateOUT/cerinta3.csv')

#CERINTA4.Coeficientul de variatie a indicatorilor la nivel de judet
cerinta4 = pd.DataFrame()
cerinta4 = industrie.copy()
cerinta4['CoeficientDeVariatie'] = (cerinta4.iloc[:,2:].std(axis=1)/cerinta4.iloc[:,2:].mean(axis=1))*100
cerinta4 = cerinta4[cerinta4['CoeficientDeVariatie']>0]
cerinta4 = cerinta4.merge(populatie_locatlitati[['Judet','Siruta']],left_on='Siruta', right_on='Siruta').drop('Siruta',axis=1)
cerinta4 = cerinta4.drop('Localitate',axis=1)
cerinta4 = cerinta4.groupby('Judet').sum()
print(cerinta4)

#CERINTA 5.La nivel de judet, se va afisa localitatea dominanta pentru fiecare industrie
cerinta5= pd.DataFrame()
cerinta5=industrie.copy()
cerinta5=cerinta5.merge(populatie_locatlitati[['Judet','Siruta']], left_on='Siruta', right_on='Siruta').drop('Siruta',axis=1)
print("-----------------------")
cerinta5.set_index('Localitate',inplace=True)
#cerinta5['MaximIndustrie'] = cerinta5.iloc[:,1:-1].max(axis=1)
#cerinta5['IndustrieMax'] = cerinta5.iloc[:,1:-2].idxmax(axis=1)
cerinta5 = cerinta5.groupby(by='Judet').idxmax()
print(cerinta5)
cerinta5.to_csv('dateOUT/cerinta5.csv')








