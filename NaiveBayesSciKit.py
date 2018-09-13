import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Un frame vacio creado con pandas
data = pd.DataFrame()

# Crearemos un clasificador donde nuestra variable meta es predecir el sexo en base a otras variables independientes
data['sexo'] = ['hombre','hombre','hombre','hombre','mujer','mujer','mujer','mujer']

# Variables
data['altura'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['peso'] = [180,190,170,165,100,150,130,150]
data['tamanio_pies'] = [12,11,12,10,6,8,7,9]

# dataframe vacio
persona = pd.DataFrame()

# datos para predecir
persona['altura'] = [6]
persona['peso'] = [130]
persona['tamanio_pies'] = [8]

# Dividimos los datos en entrenamiento y test
X_train, X_test = train_test_split(data, test_size=0.3, random_state=int(time.time()))


# Instanciamos al Clasificador
gnb = GaussianNB()

# Variables que vamos a usar
used_features =[
    'altura',
    'peso',
    'tamanio_pies',
]

# Entrenamos al clasificador
gnb.fit(
    X_train[used_features].values,
    X_train["sexo"]
)

# Hacemos predicción
y_pred = gnb.predict(X_test[used_features])

# Imprimimos resultados
print("Gaussiano, fallos de un total de {} : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["sexo"] != y_pred).sum(),
          100*(1-(X_test["sexo"] != y_pred).sum()/X_test.shape[0])
))

# Imprimimos resutado a partir de persona creada
y_pred = gnb.predict(persona)
print("Gaussiano dice: " + str(y_pred))





#Bernoulli
bern = BernoulliNB()
bern.fit(
    X_train[used_features].values,
    X_train["sexo"]
)
y_pred = bern.predict(X_test[used_features])

print("Bernoulli, fallos de un total de {} : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["sexo"] != y_pred).sum(),
          100*(1-(X_test["sexo"] != y_pred).sum()/X_test.shape[0])
))

y_pred = bern.predict(persona)
print("Bernoulli dice: " + str(y_pred))


#Multinomial
multi = MultinomialNB()
multi.fit(
    X_train[used_features].values,
    X_train["sexo"]
)
y_pred = multi.predict(X_test[used_features])

print("Multinomial, fallos de un total de {} : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["sexo"] != y_pred).sum(),
          100*(1-(X_test["sexo"] != y_pred).sum()/X_test.shape[0])
))

y_pred = multi.predict(persona)
print("Multinomial dice: " + str(y_pred))
