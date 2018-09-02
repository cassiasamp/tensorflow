from tensorflow import keras
from exibicao import exibe_exploracao_dados

def carrega_dataset_keras():
  fashion_mnist = keras.datasets.fashion_mnist
  dataset = fashion_mnist.load_data()
  return dataset

def divide_treino_e_teste(dataset):
  imagens_treino = dataset[0][0]
  identificacao_treino = dataset[0][1]
  imagens_teste = dataset[1][0]
  identificacao_teste = dataset[1][1]
  exibe_exploracao_dados(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste)
  return imagens_treino, identificacao_treino, imagens_teste, identificacao_teste

def escala_imagem(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste):
  maior_valor_pixel = 255
  imagens_treino = imagens_treino / float(maior_valor_pixel)
  imagens_teste = imagens_teste / float(maior_valor_pixel)