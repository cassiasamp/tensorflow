from exibicao import exibe_metricas

def avalia_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacao_teste)
  exibe_metricas(perda_teste, acuracia_teste)