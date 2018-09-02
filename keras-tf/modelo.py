import tensorflow as tf
from tensorflow import keras

def cria_modelo_sequencial_keras():
  dimensoes_imagem = 28, 28
  numero_de_nos = 128
  numero_de_nos_segunda_camada = 256
  taxa_de_abandono = 0.2
  total_de_categorias = 10

  modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(dimensoes_imagem)),
    keras.layers.Dense(numero_de_nos, activation=tf.nn.relu),
    keras.layers.Dense(numero_de_nos_segunda_camada, activation=tf.nn.relu),
    keras.layers.Dropout(taxa_de_abandono),
    keras.layers.Dense(total_de_categorias, activation=tf.nn.softmax)
  ])

  return modelo

def configura_parametros_modelo(modelo):
  modelo.compile(
    optimizer = tf.train.AdamOptimizer(),
		loss = 'sparse_categorical_crossentropy',
		metrics = ['accuracy']
  )