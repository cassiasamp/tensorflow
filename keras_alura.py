import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from dados import dados
from modelo import cria_modelo_sequencial_keras, configura_parametros_modelo
from treino import treina_modelo
from teste import testa_modelo



def escala_imagem(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste):
  maior_valor_pixel = 255
  imagens_treino = imagens_treino / float(maior_valor_pixel)
  imagens_teste = imagens_teste / float(maior_valor_pixel)
 
  
def plota_grafico_acuracia(historico):
  plt.plot(historico.history['acc'])
  plt.plot(historico.history['val_acc'])
  plt.title('Acurácia do modelo')
  plt.ylabel('acurácia')
  plt.xlabel('épocas')
  plt.legend(['treino', 'avaliacao'], loc='upper left')
  plt.show()
  plt.close()


def plota_grafico_perda(historico):
  plt.plot(historico.history['loss'])
  plt.plot(historico.history['val_loss'])
  plt.title('Perda do modelo')
  plt.ylabel('perda')
  plt.xlabel('épocas')
  plt.legend(['treino', 'avaliacao'], loc='upper left')
  plt.show()
  plt.close()

dataset = carrega_dataset_keras()
dados = divide_treino_e_teste(dataset)
explora_dados(*dados)
escala_imagem(*dados)
modelo_sequencial = cria_modelo_sequencial_keras()
configura_parametros_modelo(modelo_sequencial)
historico = treina_modelo(*dados, modelo_sequencial)
plota_grafico_acuracia(historico)
plota_grafico_perda(historico)
metricas = avalia_modelo(*dados, modelo_sequencial)
exibe_metricas(*metricas)
testa_modelo(*dados, modelo_sequencial)

