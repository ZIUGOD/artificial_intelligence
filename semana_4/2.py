from sklearn.svm import LinearSVC
from os import system, name

if name == "nt":
    system("cls")
else:
    system("clear")

# Características
# 1 - Fala
# 2 - Anda
# 3 - Atravessa parede
# 4 - Tem corpo físico

# Humanos
humano1 = [1, 0, 0, 1]
humano2 = [1, 1, 0, 1]
humano3 = [1, 1, 0, 1]
humano4 = [0, 1, 0, 1]

# Espíritos
espirito1 = [0, 1, 1, 0]
espirito2 = [1, 0, 1, 0]
espirito3 = [1, 1, 1, 0]
espirito4 = [0, 0, 0, 0]

# Conjunto de treinamento
treino_x = [
    humano1,
    humano2,
    humano3,
    humano4,
    espirito1,
    espirito2,
    espirito3,
    espirito4,
]
treino_y = [1, 1, 1, 1, 0, 0, 0, 0]

# Inicializar e treinar o modelo
modelo2 = LinearSVC()
modelo2.fit(treino_x, treino_y)

# Ser misterioso
ser_misterioso = [1, 1, 0, 1]
result = modelo2.predict([ser_misterioso])

if result == 1:
    print("Este ser é um humano.")
else:
    print("Este ser é um espírito.")
