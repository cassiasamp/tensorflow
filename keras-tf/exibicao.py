import numpy as np

def exibe_exploracao_dados(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste):
  print(
    'Configuração das imagens de treino:', imagens_treino.shape,
    '\nNúmero das imagens de treino:', len(identificacao_treino), 
    '\nCategorias das imagens de treino:', identificacao_treino, 
    '\n\nConfiguração das imagens de teste:', imagens_teste.shape, 
    '\nTamanho das imagens de teste:', len(identificacao_teste),
    '\n'
  )

def exibe_metricas(perda_teste, acuracia_teste):
  print('\nAcurácia do teste:', acuracia_teste)
  print('\nPerda do teste:', perda_teste)

def exibe_resultado_teste(predicoes, identificacao_teste):
   print(
    '\nValores de confiança da primeira predicao:', predicoes[0], '\n',
    '\nMaior valor de confiança da primeira predicao:', np.argmax(predicoes[0]),
    '\nIdentificação da classe no teste:', identificacao_teste[0]
  )

def exibe_resultado_teste_misterioso(predicao_misteriosa):
   print('\nPredição da imagem misteriosa:', np.argmax(predicao_misteriosa))

