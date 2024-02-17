import pandas as pd
import numpy as np
import copy

#pt analiza liniara discriminanta

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
#pt grafic
import seaborn as sb
from sklearn.naive_bayes import GaussianNB

#ADL

#CERINTA PREDICTIE IN SETUL DE TESTARE

#setul de antrenare testare = pacitenti.csv(Din cerinta)
#setul de aplicare = pacienti_apply.csv(din cerinta)

tabel_invatare_testare = pd.read_csv('dataIN/Pacienti.csv')
tabel_aplicare = pd.read_csv('dataIN/Pacienti_apply.csv')

#cand ai apply = Predicția în setul de aplicare model liniar
#cand nu ai apply = predictia in setul de testare
#mai jos am facut, predictia pt setul de testare

#punem intr o lista coloanele de testare
variabile = list(tabel_invatare_testare)
print(variabile)
#excludem Id si Decision pt ca nu sunt relevanti
predictori = variabile[1:8]
#print(predictori)
#tinta il ia pe Decision
tinta = variabile[-1]
#print(tinta)

#test_size=0.4 = 40% din date vor fi rezervate pt testare
# restul de 60% pt antrenare
# x_train: Caracteristicile de antrenare.
# x_test: Caracteristicile de testare.
# y_train: Variabila țintă de antrenare.
# y_test: Variabila țintă de testare.
x_train, x_test, y_train, y_test = train_test_split(tabel_invatare_testare[predictori],
                                                    tabel_invatare_testare[tinta], test_size=0.4)

modelADL = LinearDiscriminantAnalysis()
#set de antrenare+ variabila tinta de antrenare
modelADL.fit(x_train,y_train)

#predictie
#dupa ce modelul s-a antrenat, ii dam sa prezica pt testul pt setul de testare
predictie_ADL_test = modelADL.predict(x_test)

#PREDICTIA PENTRU SETUL DE TEST(fara apply)
predictie_ADL_test_df = pd.DataFrame(data={"Predictii:": predictie_ADL_test},
                                index=x_test.index)
predictie_ADL_test_df.to_csv('dataOUT/predict.csv')

#PREDICTIA PENTRU SETUL DE APLICARE (Cu apply), adica fara tinta
x_apply = pd.read_csv('dataIN/Pacienti_apply.csv',index_col=0)
predictie_adl_apply = modelADL.predict(x_apply[predictori])
predictie_adl_apply_df = pd.DataFrame(data={"Predictii apply":predictie_adl_apply},
                                      index=x_apply.index)
predictie_adl_apply_df.to_csv('dataOUT/predict_apply.csv')

#CERINTA2: Evaluarea modelului liniar
#MATRICE DE ACURATETE(matricea de confuzie) + ACURATETEA GLOBALA + ACURATETIE MEDIE

def calcul_matrici(y1,y2,clase):
    #MATRICEA DE CONFUZIE(De acuratede):
    c = confusion_matrix(y1,y2)
    print(c)
    #ACURATETE GLOBALA
    tabel_c = pd.DataFrame(c, clase, clase)
    print(tabel_c)
    #ACURATETE MEDIE
    tabel_c['Acuratete'] = np.round(np.diag(c) *100/np.sum(c,axis=1),3)
    print(tabel_c)

#clase= optiunile de decizie(Ce decizie putem avea)
#Calcul axe discriminante
clase = modelADL.classes_

#y_test = rezultatul cu optiunile predictiei
matrice_ADL = calcul_matrici(y_test, predictie_ADL_test,clase)
print(matrice_ADL)

#GRAFICUL DISTRIBUTIEI IN AXELE DISCRIMINANTE

def plot_distributii(scoruri,y,i=0):
    fig = plt.figure(figsize=(11,8))
    #facem axele
    #(1,1,1) -> un singur subplot
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Distributie in axa discriminanta", fontsize=11, color='b')
    sb.kdeplot(x=scoruri[:,i],hue=y,fill=True, ax=ax)
    plt.show()

#SCORURI pentru setul de test
#x_test= datele pe care s a facut predictia
scoruri_test = modelADL.transform(x_test)
# 2
lungime = len(clase)
print(lungime)
# 1
#nr de functii discriminante
m = lungime - 1

#AFISARE SCORURI IN CSV
etichete_scoruri=['z' + str(i+1) for i in range(m)]
scoruri_test_df = pd.DataFrame(scoruri_test,x_test.index,etichete_scoruri)
scoruri_test_df.to_csv('dataOUT/z.csv')

#AFISARE GRAFIC DISTRIBUTIE
# distributia implica calcularea scorurilor

#range(m) = merge pana la m-1 adica pana la 0
for i in range(m):
    print(i)
    plot_distributii(scoruri_test,y_test,i)


#GRAFIC SCORURILOR DISCRIMINANTE in primele 2 axe
def plot_instanta(z,y,clase,k1=0,k2=1):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_xlabel("z" + str(k1+1))
    #daca schimb in str k2 cu k2+2(Se duce z-ul specificat(
    ax.set_ylabel("z" + str(k2+1))
    ax.set_title("Plot instanta in axele discriminante", fontsize=14, color='b')
    sb.scatterplot(x=z[:,k1],y=z[:,k2],hue = y,hue_order=clase,ax=ax)
    plt.show()

#primele 2 axe discriminante: Z1 SI Z2!!!!!
# punem 2 deoarece vrem doar pt primele 2 axe!!
# daca punem m, afiseaza toate combinatiile intre axe
# range(m) --> pana la m-1
# range(i+1, m) --> inclusiv m
# for i in range(2):#m-1
#     for j in range(i+1,2): #m
#         plot_instanta(scoruri_test,y_test,clase,i,j)

#CREARE MODEL BAYES
model_b= GaussianNB()
model_b.fit(x_train,y_train)

#PREDICTIE model bayes in setul de date de APLICARE
predictie_b_test=model_b.predict(x_apply)
print(predictie_b_test)





