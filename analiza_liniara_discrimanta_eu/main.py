import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import train_test_split
#
# #predictie in setul de date
# tabel_invatare_testare = pd.read_csv('dateIN/Pacienti.csv')
# tabel_aplicare= pd.read_csv('dateIN/Pacienti_apply.csv')
#
# #predictie pt setul de TESTARE
# variabile = list(tabel_invatare_testare)
# predictori = variabile[1:8]
# tinta = variabile[-1]
#
# predictori_date_numerice = tabel_invatare_testare[predictori]
# predictori_tinta = tabel_invatare_testare[tinta]
#
# #CERINTA 1.Predictie in setul de testare
# x_train,x_test,y_train,y_test = train_test_split(predictori_date_numerice,predictori_tinta,test_size=0.4)
#
# #x_train = set de testare
# #y_train = tinta de testare
#
# #x_test = date concrete pt predictie
# modelADL = LinearDiscriminantAnalysis()
# modelADL.fit(x_train,y_train)
# predictie_test = modelADL.predict(x_test)
#
# predictie_test_Df = pd.DataFrame(data=predictie_test,index=x_test.index)
# predictie_test_Df.to_csv('dateOUT/predictie.csv')
#
# #predictia pentru setul de APLICARE cu apply, adica fara tinta
#
# date_apply = tabel_aplicare[predictori]
# predictie_test_apply = modelADL.predict(date_apply)
#
# predictie_test_apply_df = pd.DataFrame(data=predictie_test_apply)
# predictie_test_apply_df.to_csv('dateOUT/predictie_apply.csv')
#
#
# #CERINTA 2.MATRICEA DE ACURATETE + ACURATATEA GLOBALA + ACURATATEA MEDIE
# from sklearn.metrics import confusion_matrix #matricea de confuzie
# def calcul_matrici(y1,y2,clase):
#     #MATRICEA DE ACURATETE(CONFUZIE)
#     c = confusion_matrix(y1,y2)
#     print(c)
#
#     #ACURATETE GLOBALA
#     tabel_c = pd.DataFrame(c,clase,clase)
#     print(tabel_c)
#
#     #ACURATETE MEDIE
#     tabel_c['Medie'] = (np.diag(c)*100/np.sum(c,axis=1))
#     print(c)
#
# #clase = optiunile de decizie
#
# clase = modelADL.classes_
#
# matrice_ADL = calcul_matrici(y_test, predictie_test, clase)
# print('Matrice ADL')
# print(matrice_ADL)
#
#
# #CERINTA 3.SCORURI PENTRU SETUL DE TEST
#
# scoruri_test = modelADL.transform(x_test)
#
# #scoruri =>etichete
#
# etichete_scoruri =['z' + str(i+1) for i in range(len(clase)-1)]
#
# scoruri_test_df = pd.DataFrame(data=scoruri_test,index=x_test.index,columns=etichete_scoruri)
# scoruri_test_df.to_csv('dateOUT/scoruri.csv')
#
#
# #CREARE MODEL BAYES
#
# from sklearn.naive_bayes import GaussianNB #bayes
# modeL_bayes = GaussianNB()
# modeL_bayes.fit(x_train, y_train)
#
# #PREDICTIE model bayes in setul de date de APLICARE
# predictie_bayes_apply = modeL_bayes.predict(date_apply)
# predictie_bayes_apply_df = pd.DataFrame(data=predictie_bayes_apply, index=date_apply.index)
# predictie_bayes_apply_df.to_csv('dateOUT/predictie_Bayes')

print('inca o data')

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import train_test_split
#
# tabel_invatare_testare = pd.read_csv('dateIN/Pacienti.csv')
# tabel_aplicare = pd.read_csv('dateIN/Pacienti_apply.csv')
#
# #predictie pentru setul de TESTARE
#
# variabile = list(tabel_invatare_testare)
# predictori = variabile[1:8]
# tinta = variabile[-1]
#
# predictori_date_numerice = tabel_invatare_testare[predictori]
# predictori_tinta = tabel_invatare_testare[tinta]
#
# #CERINTA 1.PREDICTIA IN SETUL DE DATE DE TESTARE
#
# x_train, x_test, y_train, y_test = train_test_split(predictori_date_numerice,predictori_tinta, test_size=0.4)
#
# #x_train = set de antrenare
# #y_train = set de antrenare
#
# #x_test = set de testare, ESTE PENTRU PREDICTIE
# modelADL = LinearDiscriminantAnalysis()
# modelADL.fit(x_train, y_train)
# predictie_test = modelADL.predict(x_test)
#
# predictie_test_df = pd.DataFrame(data=predictie_test, index=x_test.index)
# predictie_test_df.to_csv('dateOUT/predictie1.csv')
#
#
# #predictia pentru setul de APLICARE
# date_apply = tabel_aplicare[predictori]
# predictie_test_apply = modelADL.predict(date_apply)
#
# predictie_test_apply_df = pd.DataFrame(data=date_apply)
# predictie_test_apply_df.to_csv('dateOUT/predictie_apply1.csv')
#
# #CERINTA 2. MATRICEA DE CONFUZIE, etc
# from sklearn.metrics import confusion_matrix
# def calcul_matrici(y1, y2, clase):
#     c = confusion_matrix(y1,y2)
#     print(c)
#
#     #ACURTATE GLOBALA
#     tabel_c = pd.DataFrame(c,clase,clase)
#     print(tabel_c)
#
#     #ACURATETE MEDIE
#     tabel_c['Medie'] = (np.diag(c)*100/np.sum(c,axis=1))
#     print(c)
#
# clase = modelADL.classes_
#
# matrice_ADL = calcul_matrici(y_test, predictie_test, clase)
# print(matrice_ADL)
#
# #CERINTA 3.SCORURI PENTRU SETUL DE TEST
#
# scoruri_test = modelADL.transform(x_test)
# etichete_scoruri = ['z'+ str(i+1)for i in range(len(clase)-1)]
#
# scoruri_test_df = pd.DataFrame(data=scoruri_test, index=x_test.index, columns=etichete_scoruri)
# scoruri_test_df.to_csv('dateOUT/scoruri1.csv')
#
# #CREARE MODEL BAYES
# from sklearn.naive_bayes import GaussianNB
# model_bayes = GaussianNB()
# model_bayes.fit(x_train, y_train)
#
# #PREDICTIE model bayes in setul de date de APLICARE
# predictie_bayes_apply = model_bayes.predict(date_apply)
# predictie_bayes_apply_df = pd.DataFrame(data=predictie_bayes_apply, index=date_apply.index)
# predictie_bayes_apply_df.to_csv('dateOUT/predictie_Bayes1.csv')

print('inca o data')
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

tabel_invatare_testare = pd.read_csv('dateIN/Pacienti.csv')
tabel_aplicare = pd.read_csv('dateIN/Pacienti_apply.csv')

#predictie pentru setul de testare

variabile = list(tabel_invatare_testare)
predictori = variabile[1:8]
print(predictori)
tinta = variabile[-1]

predictori_date_numerice = tabel_invatare_testare[predictori]
predictori_tinta = tabel_invatare_testare[tinta]

#CERINTA 1.PREDICTIA IN SETUL DE TESTARE

x_train, x_test, y_train, y_test = train_test_split(predictori_date_numerice, predictori_tinta, test_size=0.4)

modelADL = LinearDiscriminantAnalysis()
modelADL.fit(x_train, y_train)
predictie_test = modelADL.predict(x_test)

predictie_test_df = pd.DataFrame(data=predictie_test, index=x_test.index)
predictie_test_df.to_csv('dateOUT/predictie2.csv')

#PREDICTIA IN SETUL DE APLICARE

date_apply = tabel_aplicare[predictori]
predictie_apply = modelADL.predict(date_apply)

predictie_apply_df = pd.DataFrame(data=predictie_apply, index=date_apply.index)
predictie_apply_df.to_csv('dateOUT/predictie_apply2.csv')

#CERINTA 2.MATRICEA DE CONFUZIE
from sklearn.metrics import confusion_matrix
def calcul_matrici(y1,y2,clase):
    c = confusion_matrix(y1,y2)
    print('MATRICEA DE CONFUZIE')
    print(c)
    #ACURATETE GLOBALA
    tabel_c = pd.DataFrame(c,clase,clase)
    print('ACURATETE GLOBALA')
    print(tabel_c)

clase = modelADL.classes_
matriceADL = calcul_matrici(y_test, predictie_test, clase)
print(matriceADL)

#CERINTA 3.SCORURI PENTRU SETUL DE TEST

scoruri = modelADL.transform(x_test)

etichete_scoruri = ['z'+ str(i+1)for i in range(len(clase)-1)]

scoruri_df = pd.DataFrame(data=scoruri, index=x_test.index, columns=etichete_scoruri)

scoruri_df.to_csv('dateOUT/scoruri2.csv')