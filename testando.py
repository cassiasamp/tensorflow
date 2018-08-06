# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print('a versão do tensorflow é', tf.__version__)

#rodando esse exemplo precisei instalar o matplotlib e o python3-tk
#vamos ter que ignorar os warnings com:
#pip3 install matplotlib
#sudo apt-get install python3-tk

#load da base de dados importando do tensorflow e treinando
fashion_mnist = keras.datasets.fashion_mnist
(imagens_treino, identificacao_treino), (imagens_teste, identificacao_teste) = fashion_mnist.load_data()

nomes_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
imagens_treino.shape
len(identificacao_treino)
identificacao_treino

plt.figure()
plt.imshow(imagens_treino[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()
'''
imagens_treino = imagens_treino / 255.0

imagens_teste = imagens_teste / 255.0

import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagens_treino[i], cmap=plt.cm.binary)
    plt.xlabel(nomes_classes[identificacao_treino[i]])
#plt.show()

modelo = keras.Sequential(
	[
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)

	])

modelo.compile(optimizer = tf.train.AdamOptimizer(),
				loss = 'sparse_categorical_crossentropy',
				metrics = ['accuracy'])

modelo.fit(imagens_treino, identificacao_treino, epochs=1)

perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacao_teste)

print('acuracidade:', acuracia_teste)

predicoes = modelo.predict(imagens_teste)
#print(predicoes[0])
print(np.argmax(predicoes[0]), identificacao_teste[0])

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagens_teste[i], cmap=plt.cm.binary)
    identificacao_prevista = np.argmax(predicoes[i])
    identificacao_verdadeira = identificacao_teste[i]
    if identificacao_prevista == identificacao_verdadeira:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(nomes_classes[identificacao_prevista], 
                                  nomes_classes[identificacao_verdadeira]),
                                  color=color)
#plt.show()

# Grab an image from the test dataset
img = imagens_teste[0]
img = (np.expand_dims(img,0))

print(img.shape)
predicoes = modelo.predict(img)

print(predicoes)
predicao = predicoes[0]

print(np.argmax(predicao))