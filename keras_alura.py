import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import KFold


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
  numero_de_nos_segunda_camada = 256
  taxa_dropout = 0.2
  total_de_categorias = 10

  modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(dimensoes_imagem)),
    keras.layers.Dense(numero_de_nos, activation=tf.nn.relu),
    keras.layers.Dense(numero_de_nos_segunda_camada, activation=tf.nn.relu),
    keras.layers.Dropout(taxa_dropout),
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
  numero_de_treinos = 2
  percentual_dados_validacao = 0.2
  treinos_parada_prematura = 5
  filepath="melhoria-pesos-epoca:{epoch:02d}.hdf5"
  parada_prematura = keras.callbacks.EarlyStopping(monitor='val_loss', patience=treinos_parada_prematura)
  ponto_de_controle = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  historico = modelo.fit(imagens_treino, identificacao_treino, epochs=numero_de_treinos, validation_split=percentual_dados_validacao, callbacks=[parada_prematura, ponto_de_controle])

  return historico


def avalia_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacao_teste)
 
  print('\nAcurácia do teste:', acuracia_teste)
  print('\nPerda do teste:', perda_teste)


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
#plota_grafico_acuracia(historico)
#plota_grafico_perda(historico)
avalia_modelo(*dados, modelo_sequencial)
testa_modelo(*dados, modelo_sequencial)
#kfold(*dados, modelo_sequencial)

'''
seguem implementações tristes
def kfold(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  
  learning_rate = 0.01
  batch_size = 500

  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  pred = tf.nn.softmax(tf.matmul(x, W) + b)
  cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  init = tf.global_variables_initializer()

  #mnist = input_data.read_data_sets("data/fasion-mnist-tf", one_hot=True)
  fashion_mnist = keras.datasets.fashion_mnist
  mnist = fashion_mnist.load_data()
  train_x_all = mnist[0][0]
  train_y_all = mnist[0][1]
  test_x = mnist[1][0]
  test_y = mnist[1][1]

  def run_train(session, train_x, train_y): 
    print ("\nStart training")
    session.run(init)
    for epoch in range(10):
      total_batch = int(train_x.shape[0] / batch_size)
      for i in range(total_batch):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0:
          print ("Epoch #%d step=%d cost=%f" % (epoch, i, c))

  def cross_validate(session, split_size=5):
    results = []
    kf = KFold(n_splits=split_size)
    for train_idx, val_idx in kf.split(train_x_all, train_y_all):
      train_x = train_x_all[train_idx]
      train_y = train_y_all[train_idx]
      val_x = train_x_all[val_idx]
      val_y = train_y_all[val_idx]
      run_train(session, train_x, train_y)
      results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
    return results

  with tf.Session() as session:
    result = cross_validate(session)
    print ("Cross-validation result: %s" % result) 
    print ("Test accuracy: %f" % session.run(accuracy, feed_dict={x: test_x, y: test_y})) 
'''

'''
k_fold = KFold(n_splits=3)
  for indices_treino, inidices_teste in k_fold.split(dados):
    print('Treino: %s | teste: %s' % (indices_treino, inidices_teste))



def cross-val1(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
   n_folds = 10
    data, labels, header_info = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print "Running Fold", i+1, "/", n_folds
            model = None # Clearing the NN.
            model = create_model()    

    X, Y = load_model()
                kFold = StratifiedKFold(n_splits=10)
                scores = np.zeros(10)
                idx = 0
                for train, test in kFold.split(X, Y):
                  model = create_model()
                  scores[idx] = train_evaluate(model, X[train], Y[train], X[test], Y[test])
                  idx += 1
                print('AQUI',scores)
                print(' AQUI', scores.mean())


  def cria_classificador():
    classificador = Sequential()
    classificador.add(Dense(3, kernel_initializer = 'uniform', activation = 'relu', input_dim=5))
    classificador.add(Dense(3, kernel_initializer = 'uniform', activation = 'relu'))
    classificador.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classificador.compile(optimizer= 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
  
    return classificador

classificador = KerasClassifier(build_fn = cria_classificador,batch_size=10, nb_epoch=10)

def x(classificador, imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  acuracias_cross = cross_val_score(estimator = classificador,X = imagens_treino, y = identificacao_treino, cv = 10, n_jobs = -1)

  media = acuracias_cross.mean()

  print('MEDIA' , media)

  import keras
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
'''
