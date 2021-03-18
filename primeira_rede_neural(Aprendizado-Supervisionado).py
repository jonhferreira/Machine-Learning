import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#base de dados com imagens de roupas
base_roupas = keras.datasets.fashion_mnist

(image_train, label_train), (image_test, label_test) = base_roupas.load_data()

nomes_roupas = ["Camiseta","Calca","Sueter","Vestido","Casaco","Sandalia","Camisa","Tenis","Bolsa","Bota"]

image_train = image_train/255.0
image_test = image_test/255.0

# construindo modelo de rede neural
# formatando array para mono-dimesional
# criando camada com 128 neuronios
# criando camada com 10
modelo = keral.Sequential([
    keras.layers.Flatten(input_shape(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#configura modelo
    #      loss: essa funcao mede a precisao do modelo
    #            durante o treinamento
    # optimizer: como o modelo se atualiza baseado
    #            na medicao de loss
    #   metrics: qual a metrica de desempenho e
    #            monitoramento

modelo.compile(optimizer='adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

# treinando o modelo

modelo.fit(image_train, label_train, epochs=10)

# prevendo a label

img = image_test[0]
img = (np.expand_dims(img,0))

       
previsao = modelo.predict(img)

print(np.argmax(previsao))
