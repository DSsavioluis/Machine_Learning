
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv(r'C:\Users\Sávio\Downloads\flavors_of_cacao_ajustado.csv')


# Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
df_log_normalized = np.log(df.drop(['Rating_Categories'], axis=1) + 1)

X_train_log, X_test_log, y_train, y_test = train_test_split(df_log_normalized, df['Rating_Categories'], test_size=0.3, random_state=42)

# Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.

scaler = StandardScaler()
df_standardized = scaler.fit_transform(df.drop(['Rating_Categories'], axis=1))
X_train_std, X_test_std, y_train, y_test = train_test_split(df_standardized, df['Rating_Categories'], test_size=0.3, random_state=42)


# Ajuste o modelo KNN aos dados de treinamento para ambas as versões dos dados
knn_log = KNeighborsClassifier()
knn_log.fit(X_train_log, y_train)

knn_std = KNeighborsClassifier()
knn_std.fit(X_train_std, y_train)

# Avalie o modelo nos dados de teste para ambas as versões dos dados
accuracy_log = knn_log.score(X_test_log, y_test)
accuracy_std = knn_std.score(X_test_std, y_test)

# Print as duas acuracias lado a lado para comparar.

print("Acuracia da normalização logaritmica:", accuracy_log)
print("Acurácia da normalização de media zero:", accuracy_std)




