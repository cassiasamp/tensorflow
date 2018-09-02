from tensorflow import keras
from datetime import datetime
from graficos import plota_grafico_acuracia
from graficos import plota_grafico_perda

def treina_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  numero_de_treinos = 2
  percentual_dados_validacao = 0.2
  treinos_parada_prematura = 5
  hora_treino = datetime.now().strftime('%H:%M:%S.%f')[:-3]
  localizacao_arquivo='resultados/'+ '[' + hora_treino + ']' + 'melhoria-pesos-epoca-{epoch:d}.hdf5' 
  parada_prematura = keras.callbacks.EarlyStopping(monitor='val_loss', patience=treinos_parada_prematura)
  ponto_de_controle = keras.callbacks.ModelCheckpoint(localizacao_arquivo, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
  historico = modelo.fit(imagens_treino, identificacao_treino, epochs=numero_de_treinos, validation_split=percentual_dados_validacao, callbacks=[parada_prematura, ponto_de_controle])
  plota_grafico_acuracia(historico)
  plota_grafico_perda(historico)

