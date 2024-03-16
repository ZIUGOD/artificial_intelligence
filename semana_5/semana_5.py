# colar no colab
import pandas as pd
import io

# importar o bd
from google.colab import files

uploaded = files.upload()

dados = pd.read_csv(io.BytesIO(uploaded["wine_dataset_henri.csv"]))
dados.head()

x = dados[
    [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
]
print(set(dados))
x.head()

y = dados[["style"]]
y.head()

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# estaciar o modelo
modelo = LinearSVC()

# treinar o modelo
modelo.fit(x, y)

previsoes = modelo.predict(x)

print(
    "A acurácia sem a divisão do dataset é de: {}%".format(
        round(accuracy_score(y, previsoes) * 100, 2)
    )
)

from sklearn.model_selection import train_test_split

modelo = LinearSVC()

# 75% será utilizado para treinamento do algoritmmo
# 25% será utilizado para validação do conjunto
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y)

# treinando o algoritmo utilizando os 75% do bd
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

print(
    "A acurácia após a divisão do dataset é de: {}%".format(
        round(accuracy_score(teste_y, previsoes) * 100, 2)
    )
)
