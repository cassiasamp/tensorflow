import matplotlib.pyplot as plt

def plota_grafico_por_epocas(historico_treino, historico_avaliacao, titulo, titulo_eixo_y):
  plt.plot(historico_treino)
  plt.plot(historico_avaliacao)
  plt.title(titulo)
  plt.xlabel('épocas')
  plt.ylabel(titulo_eixo_y)
  plt.legend(['treino', 'avaliacao'], loc='upper left')
  plt.show()
  plt.close()

def plota_grafico_acuracia(historico):
  plota_grafico_por_epocas(
    titulo = 'Acurácia do modelo',
    historico_treino = historico.history['acc'],
    historico_avaliacao = historico.history['val_acc'],
    titulo_eixo_y = 'acurácia'
  )

def plota_grafico_perda(historico):
  plota_grafico_por_epocas(
    titulo = 'Perda do modelo',
    historico_treino = historico.history['loss'],
    historico_avaliacao = historico.history['val_loss'],
    titulo_eixo_y = 'perda'
  )
