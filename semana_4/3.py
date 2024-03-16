from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from os import system, name

if name == "nt":
    system("cls")
else:
    system("clear")

# Características:
# [idade, renda_mensal, historico_credito, emprego, valor_emprestimo, duracao_emprestimo, dividas_existentes, estado_civil, educacao, tipo_emprego]

# Dados de treinamento
treino_x = [
    [25, 5000, 1, 1, 10000, 12, 0, 1, 2, 1],
    [35, 7000, 1, 1, 20000, 24, 1000, 2, 3, 2],
    [40, 8000, 1, 1, 30000, 36, 2000, 2, 3, 2],
    [30, 6000, 0, 1, 15000, 18, 500, 1, 2, 1],
    [22, 3000, 0, 0, 5000, 6, 0, 1, 1, 0],
    [45, 10000, 1, 1, 40000, 48, 5000, 2, 4, 2],
    [28, 5500, 1, 1, 18000, 24, 1000, 1, 3, 1],
    # Mais exemplos de treinamento aqui...
]

# Rótulos correspondentes aos dados de treinamento (1 para empréstimo aprovado, 0 para empréstimo não aprovado)
treino_y = [
    1,
    1,
    1,
    1,
    0,
    1,
    1,
]  # Exemplo: empréstimo aprovado para todos, exceto exemplo 5

# Inicializar e treinar o modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

# Dados de teste
teste_x = [
    [33, 6500, 1, 1, 25000, 24, 1500, 1, 3, 1],
    [20, 2000, 0, 0, 4000, 12, 0, 1, 1, 0],
    [38, 8500, 1, 1, 35000, 36, 3000, 2, 4, 2],
    [27, 4000, 0, 1, 10000, 18, 2000, 1, 2, 1],
    [42, 9000, 1, 1, 45000, 48, 6000, 2, 4, 2],
    [23, 5500, 0, 1, 12000, 12, 1000, 1, 2, 1],
    [36, 7500, 1, 1, 28000, 24, 2000, 2, 3, 2],
]

# Rótulos correspondentes aos dados de teste (para fins de validação)
teste_y = [1, 0, 1, 1, 1, 1, 1]  # Rótulos para os exemplos de teste

# Previsões do modelo para os dados de teste
previsoes = modelo.predict(teste_x)

# Calcular a acurácia do modelo
acuracia = accuracy_score(teste_y, previsoes)
print("A acurácia do modelo LinearSVC é: {:.2f}%".format(acuracia * 100))

# Comparação com o classificador não guiado (DummyClassifier)
dummy = DummyClassifier(strategy="uniform")
dummy.fit(treino_x, treino_y)
predicao_dummy = dummy.predict(teste_x)
acuracia_dummy = accuracy_score(teste_y, predicao_dummy)
print("A acurácia do DummyClassifier é: {:.2f}%".format(acuracia_dummy * 100))
