#Importando TensorFlow e Keras (de dentro do TF)
import tensorflow as tf
from tensorflow import keras

#Importando as bibliotecas auxiliares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Testando a versão do TF para ver se está tudo ok
#print('A versão do tensorflow é:', tf.__version__)

#Rodando esse exemplo precisei instalar o matplotlib e o python3-tk
#Tive que ignorar os warnings do numpy e da nvidia com:
#pip3 install matplotlib
#sudo apt-get install python3-tk

#IMPORTACAO DO DATASET (CONJUNTO DE DADOS)
#Fazendo o load da base de dados do Tensorflow e retornando 4 arrays do numpy, 2 de treino e 2 de teste
fashion_mnist = keras.datasets.fashion_mnist
print(fashion_mnist)
#onde esta o dataset -> /home/cassia/tensorflow/venv/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets
#mostrar como fazer o load de um csv local, etc
(imagens_de_treino, identificacao_treino), (imagens_de_teste, identificacao_teste) = fashion_mnist.load_data()

nomes_das_classes = ['Camiseta', 'Calça', 'Pulover', 'Vestido', 'Casaco', 
                        'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

#EXPLORACAO DOS DADOS
#Explorando e vendo que tem 60.000 imagens de treino de 28x28 pixels
print('Configuração das imagens de treino:', imagens_de_treino.shape)
print('Tamanho das imagens de treino:', len(identificacao_treino))
#Vendo que cada nome/etiqueta/identificação é um inteiro entre 0 e 9
print('Nomes das imagens de treino:', identificacao_treino)

#Explorando e vendo que tem 10.000 imagens no conjunto de teste, cada com 28x28 pixels
print('Configuração das imagens de treino:', imagens_de_teste.shape)
print('Tamanho das imagens de teste:', len(identificacao_teste))

#Inspecionando a primeira imagem do conjunto de treino e vendo que os valores dos pixels vão de 0 a 255
def inspecionandoPrimeiraImagemDeTreino(imagens_de_treino):
  plt.figure('Primeira imagem de treino')
  plt.imshow(imagens_de_treino[0])
  plt.colorbar()
  plt.gca().grid(False)
  plt.show()
 
inspecionandoPrimeiraImagemDeTreino(imagens_de_treino)

#PRE PROCESSAMENTO DOS DADOS
#fazer uma variacao das imagens - testar para ver o que acontece ao dividir por um numero diferente de pixels

#Escalando os valores dos pixels para um intervalo entre 0 e 1 antes de alimentar a rede.
#Fazendo isso com o casting de int para float e dividindo pelo número máximo de pixels da imagem, 255.
imagens_de_treino = imagens_de_treino / 255.0
imagens_de_teste = imagens_de_teste / 255.0

def mostraImagemCategoriasDeTreino(imagens_de_treino, nomes_das_classes, identificacao_treino):
  altura_da_figura = 10
  largura_da_figura = 10
  plt.figure(figsize=(altura_da_figura,largura_da_figura))
  plt.suptitle('25 Imagens de Treino')
  numero_de_linhas = 5
  numero_de_colunas = 5
  total_de_figuras = 25
  #cria uma ou 25 figuras
  for figura in range(total_de_figuras):
      plt.subplot(numero_de_linhas,numero_de_colunas,figura+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(imagens_de_treino[figura], cmap=plt.cm.binary)
      plt.xlabel(nomes_das_classes[identificacao_treino[figura]])
  #plt.show()
  plt.close()

mostraImagemCategoriasDeTreino(imagens_de_treino, nomes_das_classes, identificacao_treino)

#Definindo três camadas do modelo sequencial do Keras para uma classificação categórica com 10 classes
#Usando a camada de "achatar" (flatten) para reformatar as imagems de um array 2d (de 28 por 28 pixels) para um array 1d de 28*28 = 784 pixels
#Usando duas camadas "densas" (dense), a primeira com 128 nós (ou neurônios)e a segunda aplicando softmax com 10 nós, o que retorna um array com 10 probabilidades que somam 1. 
#Cada nó contém um valor que indica a probabilidade da imagem pertencer a uma das 10 classes.
#def configuracaoDoModeloSequencial():
modelo_sequencial = keras.Sequential(
  	[
  		keras.layers.Flatten(input_shape=(28,28)),
      #apresentacao de powerpoint
      #pegar a documentacao, mostrar com outros algoritmos para ver que nao funciona 
  		keras.layers.Dense(128, activation=tf.nn.relu),
  		keras.layers.Dense(10, activation=tf.nn.softmax)
  	])
#    return modelo_sequencial

#configuracaoDoModeloSequencial()

#Configurando o processo de aprendizado dizendo como será a otimização, a perda e a métrica
modelo_sequencial.compile(optimizer = tf.train.AdamOptimizer(),
				loss = 'sparse_categorical_crossentropy',
				metrics = ['accuracy'])

modelo_sequencial.fit(imagens_de_treino, identificacao_treino, epochs=0)

perda_teste, acuracia_teste = modelo_sequencial.evaluate(imagens_de_teste, identificacao_teste)

print('acuracidade:', acuracia_teste)

predicoes = modelo_sequencial.predict(imagens_de_teste)
#print(predicoes[0])
print(np.argmax(predicoes[0]), identificacao_teste[0])

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
fig = plt.figure('Imagens que a classificação acertou e errou', figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagens_de_teste[i], cmap=plt.cm.binary)
    identificacao_prevista = np.argmax(predicoes[i])
    identificacao_verdadeira = identificacao_teste[i]
    if identificacao_prevista == identificacao_verdadeira:
      cor = 'green'
    else:
      cor = 'red'
    plt.xlabel("{} ({})".format(nomes_das_classes[identificacao_prevista], 
                                  nomes_das_classes[identificacao_verdadeira]),
                                  color=cor)
plt.show()
plt.close(fig)

# Grab an image from the test dataset
imagem_misteriosa = imagens_de_teste[0]
imagem_misteriosa = (np.expand_dims(imagem_misteriosa,0))

print(imagem_misteriosa.shape)
predicoes = modelo_sequencial.predict(imagem_misteriosa)

print(predicoes)
predicao = predicoes[0]

print(np.argmax(predicao))
