from dados import carrega_dataset_keras 
from dados import divide_treino_e_teste
from dados import escala_imagem
from modelo import cria_modelo_sequencial_keras
from modelo import configura_parametros_modelo
from treino import treina_modelo
from teste import testa_modelo
from avaliacao import avalia_modelo

dataset = carrega_dataset_keras()
dados = divide_treino_e_teste(dataset)
escala_imagem(*dados)
modelo_sequencial = cria_modelo_sequencial_keras()
configura_parametros_modelo(modelo_sequencial)
treina_modelo(*dados, modelo_sequencial)
avalia_modelo(*dados, modelo_sequencial)
testa_modelo(*dados, modelo_sequencial)

