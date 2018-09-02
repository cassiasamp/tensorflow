import numpy as np
from exibicao import exibe_resultado_teste
from exibicao import exibe_resultado_teste_misterioso

def testa_modelo(imagens_treino, identificacao_treino, imagens_teste, identificacao_teste, modelo):
  predicoes = modelo.predict(imagens_teste)
  exibe_resultado_teste(predicoes, identificacao_teste)

  imagem_misteriosa = imagens_teste[0]
  imagem_misteriosa = (np.expand_dims(imagem_misteriosa,0)) 
  predicao_misteriosa = modelo.predict(imagem_misteriosa) 
  exibe_resultado_teste_misterioso(predicao_misteriosa)