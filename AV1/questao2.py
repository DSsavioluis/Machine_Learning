
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.

df = pd.read_csv(r'C:\Users\Sávio\Downloads\flavors_of_cacao_ajustado.csv')

# Remova a coluna "Rating" original (opcional, dependendo da sua necessidade)
df.drop('Rating', axis=1, inplace=True)

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = df.drop(['Rating_Categories'], axis=1)
y = df['Rating_Categories']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Ajuste o modelo de KNN aos dados de treinamento.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Avalie o modelo nos dados de teste.
accuracy = knn.score(X_test, y_test)
print('Acurácia do modelo KNN: {:.2f}'.format(accuracy))


