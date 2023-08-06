#importe as bibliotecas necessárias
from sklearn.preprocessing import LabelEncoder
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv(r"C:\Users\Sávio\Downloads\flavors_of_cacao.csv")

# Rename columns that have a space in the name
df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# Pré-processamento

'''print(df.info())'''

#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

'''print(df.isnull().sum())'''

df = df.drop(['Bean\nType'], axis=1)

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

# Codificar as colunas categóricas

label_encoder = LabelEncoder()

df['Company \n(Maker-if_known)'] = label_encoder.fit_transform(df['Company \n(Maker-if_known)'])
df['Specific_Bean_Origin\nor_Bar_Name'] = label_encoder.fit_transform(df['Specific_Bean_Origin\nor_Bar_Name'])
df['Company\nLocation'] = label_encoder.fit_transform(df['Company\nLocation'])
df['Broad_Bean\nOrigin'] = label_encoder.fit_transform(df['Broad_Bean\nOrigin'])

#convertendo a coluna Cocoa Percent de string para float

df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('%', '').astype(float) / 100.0

# Transformando Rating me discretos
# Defina os intervalos e rótulos para categorização
intervals = [0, 3, 3.5, 5]
labels = [0, 1, 2]

# Categorize a coluna "Rating"
df['Rating_Categories'] = pd.cut(df['Rating'], bins=intervals, labels=labels, include_lowest=True)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

'''print(df['Rating'].value_counts())
print(df['Cocoa\nPercent'].value_counts())
print(df['Company \n(Maker-if_known)'].value_counts())
print(df['Specific_Bean_Origin\nor_Bar_Name'].value_counts())
print(df['Company\nLocation'].value_counts())
print(df['Broad_Bean\nOrigin'].value_counts())
print(df['REF'].value_counts())
print(df['Review\nDate'].value_counts())
'''

#Salve o dataset atualizado se houver modificações.

df.to_csv(r'C:\Users\Sávio\Downloads\flavors_of_cacao_ajustado.csv', index=False)
