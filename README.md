Claro, aqui está a reedição do seu script:

---

# Redução de Dimensionalidade

Em muitos problemas de IA que envolvem uma grande quantidade de dados, nos deparamos com o desafio do desbalanceamento de classes, conhecido como a maldição da dimensionalidade. Com o avanço da mineração de dados, podemos afirmar de forma satisfatória que há maneiras de resolver esse problema de desequilíbrio que afeta significativamente a precisão do modelo.

Ao reduzir a dimensão dos seus dados, o objetivo é tornar o treinamento mais rápido e ajudar a encontrar uma boa solução para o problema proposto. No entanto, após a execução da redução, perdemos algumas informações, o que implica que o modelo pode perder sua capacidade potencial sem uma razão forte. Portanto, é importante primeiro tentar treinar o modelo com os dados originais antes de considerar a redução da dimensionalidade.

Neste artigo, abordaremos de forma conceitual e prática três algoritmos de Redução de Dimensionalidade: t-SNE, PCA e Truncated SVD.

### PCA

O PCA é um procedimento numérico que tenta encontrar uma combinação linear das variáveis originais que captura melhor a variância dos dados. Em outras palavras, o PCA projeta os dados em um espaço de menor dimensão, mantendo o máximo de informação possível. Ele é baseado em álgebra linear e utiliza a decomposição de valores singulares (SVD) para calcular as componentes principais.

A análise de componentes principais (PCA) gera informações que permitem manter os componentes mais relevantes enquanto preserva os segmentos mais importantes do conjunto geral de dados. Além disso, há uma vantagem adicional, já que cada um dos novos destaques ou segmentos gerados após a aplicação do PCA são, em geral, independentes uns dos outros.

### t-SNE

O t-SNE é uma técnica de redução de dimensionalidade não linear que atua de forma indireta, sendo adequada para conjuntos de dados de alta dimensão. É um procedimento probabilístico que tenta preservar a estrutura de vizinhança dos dados originais em um espaço de menor dimensão. Em vez de focar na variância, o t-SNE tenta encontrar uma distribuição de probabilidade que reflita a semelhança entre pares de pontos nos dados originais. Utiliza a distribuição t-student para modelar a distribuição de probabilidade e uma abordagem iterativa para otimizar a projeção dos dados.

O t-SNE é amplamente utilizado em problemas de manipulação de imagens, PNL, informação genômica e preparação de discurso. Ele pode ser implementado mapeando as informações multidimensionais para um espaço de menor dimensão e procurando padrões que possam gerar informações. De uma forma mais simples, ele incorpora os pontos de uma dimensão superior para uma dimensão inferior, tentando preservar a vizinhança daquele ponto.

### Diferença entre PCA e t-SNE

Ambos são técnicas de redução de dimensionalidade, porém algumas diferenças podem ser notadas ao utilizá-los:

1. O t-SNE tem um tempo de execução maior quando aplicado a conjuntos de dados com milhões de observações, sendo também computacionalmente mais caro. Enquanto isso, o PCA finalizará a atividade em um período menor de tempo.

2. Os procedimentos diferem: o PCA é um procedimento numérico, enquanto o t-SNE é um procedimento probabilístico. O PCA se concentra na variância dos dados, enquanto o t-SNE se concentra na semelhança entre pares de pontos nos dados originais.

3. O PCA é sensível a outliers, enquanto o t-SNE lida melhor com esse problema.

4. O PCA tenta preservar a estrutura global dos dados, enquanto o t-SNE tenta preservar a estrutura local (cluster) dos dados.

### Truncated SVD

Esta também é uma técnica de redução de dimensionalidade, mais utilizada em dados com um alto número de valores ausentes ou dados esparsos. Por exemplo, em sistemas de recomendação de produtos, onde muitos clientes podem não ter comentado ou classificado um produto, gerando assim valores zero nos dados.

O Truncated SVD utiliza uma fatoração de matriz semelhante ao PCA, com a diferença de que a Análise de Componentes Principais utiliza a matriz de covariância.

A SVD truncada com matriz de dados fatorada pode ser explicada como tendo o número de colunas igual ao truncamento. Além disso, ela exclui os dígitos após a casa decimal para diminuir o valor dos dígitos flutuantes matematicamente. Por exemplo, 3,349 pode ser truncado para 3,5.

### Parte Prática

**Bibliotecas:**
```python
import numpy as np
import pandas as pd
from dfply import*
```

**Coleta de dados:**
```python
df_train = pd.read_csv('train.csv')
print(df_train.head(3))
```

**Encoder:**
```python
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

df_train['Home Ownership'] = label_encoder.fit_transform(df_train['Home Ownership'])
df_train['Purpose'] = label_encoder.fit_transform(df_train['Purpose'])
df_train['Term'] = label_encoder.fit_transform(df_train['Term'])
df_train['Years in current job'] = label_encoder.fit_transform(df_train['Years in current job'])
```

**Preenchendo valores NaN:**
```python
df_train.fillna(-99999, inplace=True)
```

**Separando as variáveis X e Y:**
```python
X = df_train.drop('Credit Default', axis=1)
y = df_train['Credit Default']
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y,  test_size = 0.7, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Implementação dos algoritmos:**
```python
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# Implementação do T-SNE
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# Implementação do PCA
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
t1 = time.time()
print("PCA took {:.2}
