import pandas as pd
import numpy as np 
import matplotlib as plt
import seaborn as sns

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")
datacore = data.copy()
data1 = data.copy()
data.head()

#datacore = datacore.drop( )
datacore.drop(labels=datacore[datacore["Glucose"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["BloodPressure"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["BMI"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["SkinThickness"] == 0 ].index, inplace=True)
datacore.drop(labels=datacore[datacore["Insulin"] == 0 ].index, inplace=True)


datacore["Outcome"].value_counts()

from scipy.stats.mstats import winsorize

datacore['Insulin_winsorized'] = winsorize(datacore['Insulin'], limits=[0.05, 0.05]) 

datacore.columns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Define tus variables independientes (X) y dependiente (y)
X = datacore.drop(columns=['Outcome'], axis=1)  # Reemplaza 'columna_objetivo' por el nombre real de la variable objetivo
y = datacore["Outcome"]

# Divide el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Imprimir tamaños resultantes
print('Tamaño set de entrenamiento: ', X_train.shape, y_train.shape)
print('Tamaño set de prueba: ', X_test.shape, y_test.shape)

# Distribución de categorías
print('Distribución de categorías dataset original: ', y.value_counts(normalize=True))
print('Distribución de categorías dataset entrenamiento: ', y_train.value_counts(normalize=True))
print('Distribución de categorías dataset prueba: ', y_test.value_counts(normalize=True))

# Muestra las primeras filas del conjunto de entrenamiento
print(datacore["Outcome"].value_counts())
print(X.columns)
 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)


import matplotlib.pyplot as plt
from sklearn import tree

fig, axis = plt.subplots(2, 2, figsize = (15, 15))

# We show the first 4 trees out of the 100 generated (default)
tree.plot_tree(model.estimators_[0], ax = axis[0, 0], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[1], ax = axis[0, 1], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[2], ax = axis[1, 0], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[3], ax = axis[1, 1], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)

plt.show()


y_pred = model.predict(X_test)
y_pred


from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

from sklearn import metrics

accuracy_score(y_test, y_pred)

confusion_matrix(y_test,y_pred)

precision = precision_score(y_test, y_pred)

#COMPARANDO RESULTADOS CON DESICION TREE

# Inicializar modelos
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

# Listas para almacenar los resultados
lista_rf, lista_dt = [], []

# Repetir 20 veces
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    
    pred_rf = rf.predict(X_test)
    pred_dt = dt.predict(X_test)
    
    acc_rf = accuracy_score(y_test, pred_rf)
    acc_dt = accuracy_score(y_test, pred_dt)
    
    lista_rf.append(acc_rf)
    lista_dt.append(acc_dt)

# Promedio de accuracy
print("Accuracy promedio Random Forest:", np.mean(lista_rf))
print("Accuracy promedio Decision Tree:", np.mean(lista_dt))


#CAMBIANDO HIPERPARAMETROS

depths = range(1, 21)  # Probar max_depth desde 1 hasta 20
accuracy_por_depth = []

for depth in depths:
    rf = RandomForestClassifier(max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy_por_depth.append(acc)
    print(f"max_depth = {depth}, accuracy = {acc:.4f}")

# Si quieres ver qué profundidad dio mejor resultado
mejor_depth = depths[np.argmax(accuracy_por_depth)]
print(f"\n✅ Mejor max_depth: {mejor_depth} con accuracy de {max(accuracy_por_depth):.4f}")

model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=mejor_depth)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

confusion_matrix(y_test,y_pred)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

from sklearn.metrics import f1_score

# Ya deberías tener y_pred
f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)

from pickle import dump

with open("Random_forest_classifier_default_42.sav", "wb") as f:
    dump(model, f)