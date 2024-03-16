from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from os import system, name

if name == "nt":
    system("cls")
else:
    system("clear")

# Dados de treinamento
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]

# Inicializar e treinar o modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

# Animal misterioso
animal_misterioso = [0, 1, 0]
result = modelo.predict([animal_misterioso])
print(
    "O animal misterioso é um porco."
    if result == 1
    else "O animal misterioso é um cachorro."
)

# Teste de múltiplos animais misteriosos
animais_misteriosos = [[1, 1, 1], [1, 1, 0], [0, 1, 1]]

for animal in animais_misteriosos:
    resultado = modelo.predict([animal])
    print(
        "O animal misterioso é um porco."
        if resultado == 1
        else "O animal misterioso é um cachorro."
    )

# Avaliação da acurácia do modelo
teste_y = [0, 1, 1]
previsoes = modelo.predict(animais_misteriosos)
acuracia = accuracy_score(teste_y, previsoes)
print("A acurácia do modelo de aprendizado de máquina é de: {}%".format(acuracia * 100))

# Desenvolvimento do classificador burro - não guiado
salva_acuracia_dummy = []
testes = 300

for i in range(testes):
    dummy = DummyClassifier(strategy="uniform")
    dummy.fit(treino_x, treino_y)
    predicao_dummy = dummy.predict(animais_misteriosos)
    acuracia_dummy = accuracy_score(teste_y, predicao_dummy)
    salva_acuracia_dummy.append(acuracia_dummy)

media = sum(salva_acuracia_dummy) / testes
print("A média da acurácia do algoritmo dummy é: {}%".format(media * 100))
