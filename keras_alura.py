import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def carrega_dataset_keras():
  fashion_mnist = keras.datasets.fashion_mnist
  dataset = fashion_mnist.load_data()
  return dataset


def divide_treino_e_teste(dataset):
  imagens_treino = dataset[0][0]
  identificacao_treino = dataset[0][1]
  imagens_teste = dataset[1][0]
  identificacao_teste = dataset[1][1]
  return imagens_treino, identificacao_treino, imagens_teste, identificacao_teste


def explora_dados(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste):
  print(
    'Configuração das imagens de treino:', imagens_treino.shape,
    '\nNúmero das imagens de treino:', len(identificacao_treino), 
    '\nCategorias das imagens de treino:', identificacao_treino, 
    '\n\nConfiguração das imagens de teste:', imagens_teste.shape, 
    '\nTamanho das imagens de teste:', len(identificacao_teste),
    '\n'
  )


def escala_imagem(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste):
  maior_valor_pixel = 255
  imagens_treino = imagens_treino / float(maior_valor_pixel)
  imagens_teste = imagens_teste / float(maior_valor_pixel)


def cria_modelo_sequencial_keras():
  dimensoes_imagem = 28, 28
  numero_de_nos = 128
  total_de_categorias = 10

  modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(dimensoes_imagem)),
    keras.layers.Dense(numero_de_nos, activation=tf.nn.relu),
    keras.layers.Dense(total_de_categorias, activation=tf.nn.softmax)
  ])
  return modelo


def configura_parametros_modelo(modelo):
  modelo.compile(
    optimizer = tf.train.AdamOptimizer(),
		loss = 'sparse_categorical_crossentropy',
		metrics = ['accuracy']
  )


def treina_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  numero_de_treinos = 5
  modelo.fit(imagens_treino, identificacao_treino, epochs=numero_de_treinos)


def avalia_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacao_teste)
  print('\nAcurácia:', acuracia_teste)



def testa_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  predicoes = modelo.predict(imagens_teste)
  print(
    '\nValores de confiança da primeira predicao:', predicoes[0], '\n',
    '\nMaior valor de confiança da primeira predicao:', np.argmax(predicoes[0]),
    '\nIdentificação da classe no teste:', identificacao_teste[0]
  )
  
  imagem_misteriosa = imagens_teste[0]
  imagem_misteriosa = (np.expand_dims(imagem_misteriosa,0)) 
  predicao_misteriosa = modelo.predict(imagem_misteriosa) 
  print('\nPredição da imagem misteriosa:', np.argmax(predicao_misteriosa))
  

dataset = carrega_dataset_keras()
dados = divide_treino_e_teste(dataset)
explora_dados(*dados)
escala_imagem(*dados)
modelo_sequencial = cria_modelo_sequencial_keras()
configura_parametros_modelo(modelo_sequencial)
treina_modelo(*dados, modelo_sequencial)
avalia_modelo(*dados, modelo_sequencial)
testa_modelo(*dados, modelo_sequencial)