import numpy as np
from exibicao import exibe_resultado_teste
from exibicao import exibe_resultado_teste_primeira_imagem

def testa_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  predicoes = modelo.predict(imagens_teste)
  exibe_resultado_teste(predicoes, identificacao_teste)
  prediz_primeira_imagem(imagens_teste, modelo)

def prediz_primeira_imagem(imagens_teste, modelo):
  primeira_imagem_teste = imagens_teste[0]
  lote_de_uma_imagem = (np.expand_dims(primeira_imagem_teste,0)) 
  predicao_primeira_imagem_teste = modelo.predict(lote_de_uma_imagem) 
  exibe_resultado_teste_primeira_imagem(predicao_primeira_imagem_teste)