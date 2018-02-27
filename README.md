# Desafio_Kaggle
Minhas tentativas em resolver o desafio do titanic do kaggle


## Primeiro passo: Importar as bibliotecas que vou utilizar:

import pandas as pd
#Permite abrir, visualizar e manipular data mais facilmente

import numpy as np
#Manipulação de vetores - Funções matemáticas

import matplotlib.pyplot as plt
#Usamos pra plots 2D

import seaborn as sns
#Plots e analise estatistica

%matplotlib inline
#Dessa forma os gráficos aparecem na tela

---------------------------------------------------------------------------------------------------------------------------------
## Segundo passo: Lendo os dados e entendendo eles um pouco melhor

data_treino = pd.read_csv("train.csv", index_col=_col=0)
#Esse comando permite que eu "leia" o arquivo csv. Como salvei o script, código, na mesma pasta que o arquivo csv está, n preciso especificar o caminho dele pra conseguir abrir ele.
#O argumento index_col=_col=o faz com que a primeira coluna seja a usada de referencia, ao invés de criar uma numeração usual, sendo a primeira linho o 0.

data_treino.head()
#Esse comando mostra as 5 primeiras linhas da matriz. Assim é possível ver quais são as colunas que temos, e ter uma noção dos dados.
#Podemos colocar um número dentro do (), especificando assim o número de linhas que queremos que a função mostre.

data_treino.tail()
#Comando similar a .head(), só que esse mostra as últimas 5 linhas da matriz.

data_treino.columns.values
#Esse comando nos retorna uma string com o "nome" de todas as colunas da matriz

#Quero agora um breve resumo dos dados que tenho - Temos algumas infos estatísticas aqui.
data_treino.describe()
#É possível perceber que temos 891 entradas. E como se pode notar, em idade temos alguns dados faltando.

#Quero agora mais informações sobre os dados dessa matriz.
data_treino.info()
#Esse comando nos informa os tipos de dados da matriz, e novamente, quais esão devidamente preenchidos.
#Sabemos agora qual o tipo de cada um deles (isso vai ser importante para implementar o algoritmo).
#Sabemos também o número de elementos não vazios em cada coluna - Vamos usar isso para "filtrar" os dados. 
data_treino.isnull().sum()
#Aqui vou contar quantas entradas nulas tenho nos dados. Isso também vai me ajudar a ter ideias de como lidar com os dados
#A fç isnull() é da bibl pandas e sum() tbm - E isso q eu fiz foi combinar as duas de uma vez só.
#N tenho ctza se isso funciona com quaisquer funções

#Quero agora ver a relação entre algumas das variáveis

# Plot de gráficos:

- Vou plotar vários gráficos tentando relacionar parâmetros que acho que podem ter influenciado na sobrevivência ou não dos passageiros

g = sns.FacetGrid(data_treino, row='Sex', col='Survived')
# Initialize a grid of plots with an Axes for each "option" of the variable survive
g.map(plt.hist, 'Age', bins=100)

g = sns.FacetGrid(data_treino, col='Survived')
g.map(plt.hist, 'Embarked' , bins=100)

g = sns.FacetGrid(data_treino, row='Sex', col='Survived')
g.map(plt.hist, 'Embarked' , bins=100)

g = sns.FacetGrid(data_treino, row ='Sex', col='Survived')
g.map(plt.hist, 'SibSp' , bins=50)

g = sns.FacetGrid(data_treino, row='Sex', col='Survived')
g.map(plt.hist, 'Parch' , bins=10)

g = sns.FacetGrid(data_treino, row='Sex', col='Survived')
g.map(plt.hist, 'Pclass' , bins=100)

ax = sns.countplot(x = 'Pclass', hue = 'Survived', data = data_treino)
ax.set(xlabel = 'Passenger Class', ylabel = 'Total')

#BoxPLot relacionando idade e sobrevivencia
#Temos aqui que 75% das pessoas que sobreviveram tinha menos de 40 anos, por exemplo
#Conseguimos ver também algun outliers, que poderiamos vir a retirar dos dados - Mas não vamos fazer isso
ax = sns.boxplot(x='Survived', y='Age', data=data_treino)

      ----------------------------------------------------------------------------------------------------------------------------

                            Relembrando um pouco as informações do problema:

'Survived' - Se o paciente sobreviveu ou não: 0 = Não sobreviveu e 1 = Sobreviveu
'Pclass' - Em qual classe o passageiro estava no navio {1, 2, 3}
'Age' - Idade do passageiro
'SibSp' - Número de irmãos/marido/mulher que o passageiro tem no navio
'Parch' - Número de mãe/pai/filhos que o passageiro tem no navio
'Ticket' - Número do ticket do passageiro
'Fare' - Valor que o passageiro pagou pelo seu ticket
'Cabin' - Número da cabine do passageiro
'Embarked' - Local em terra onde o passageiro embarcou: C = Cherbourg, Q = Queenstown, S = Southampton

      ----------------------------------------------------------------------------------------------------------------------------


## Relações observadas

Observando os gráficos acima conseguimos notar que: Sexo, Local de embarque, SibSp, Parch e local de embarque influenciam no fato do passageiro ter sobrevivido ou não.
- Fare pode ser relacionada a classe do passageiro, assim, vou usar a classe do passageiro ao invés do valor que ele pagou pelo seu ticket.
- Cabin pode ser usada para ver em qual nível do navio o passageiro se encontra. Mas por enquanto vou ignorar essa informação.
- Dessa forma, vou limpar os dados das seguintes "colunas" (não vou usar esses dados na construção do modelo): Nome, Ticket, Fare e Cabin.
- Entretanto, temos muitas idades (177) com valores vazios, logo precisamos pensar em uma forma de preencher elas. Por outro lado, temos só 2 entradas de local de embarque vazis. Por isso, a princípio, vamos eleminar essas dua entradas.

--------------------------------------------------------------------------------------------------------------------------------------

## Terceiro passo: Lidando com os dados baseado nas informações que conseguimos acima

##### Para preencher Nan values no python vamos utilizar a função .fillna() - Precisamos ver agora como vamos fazer isso

Temos dois locais de embarque sem valor:
   Opção 1: Eliminar eles - São poucos dados; me parece ok fazer isso.
   Opção 2: Tentar modelar e adivinhar esses dois valores
   Opção 3: Preencher essas duas entradas aleatoriamente com uma das 3 opções possíveis
   
Temos 177 idades sem valor:
    Opção 1: Eliminar essas entradas - Não é viável, pois perderiamos muitos dados, e já não temos uitos para o modelo...
    ##### Para preencher os valores NaN vamos utilizar a função .fillna()
    Opção 2: Preencher eles com a média das idades que temos (dos resultados anteriores, temos que a média das idades é 29.69, e      o desvio padrão 14.526497) - Muito genérico
    Opção 3: Preencher eles aleatoriamente com valores entre a média +/- desvio padrão
    Opção 4: Preencher eles com a moda das idades que temos - interessante
    ##### Não sei direito como fazer isso ainda... Mas acho que precisamos separa as entradas NaN das que tem valores, para então aplicarmos o modelo que definirmos - Assim, preciso olhar possíveis modelos preditivos para isso. Talvez regressão ou então até mesmo ML mesmo...
    Opção 5: Criar um modelo pra predizer essas idades vazias
    
Acho válido testar os modelos que formos utilizar com cada uma dessas opções. Entretanto vamos analisar só as opçoes de idade, as de local de embarque não acho que valem a pena. Assim, vamos começar limpando os dados, que essas alterações vão ser mantidas para todas as opções...

#Removendo as colunas que não vamos utilizar e os valores não preechidos de embarque.
data_treino_limpo = data_treino
#Creiei uma outra matrix, só para não mexer nos dados originais.
del data_treino_limpo['Name']
del data_treino_limpo['Ticket']
del data_treino_limpo['Fare']
del data_treino_limpo['Cabin']
#Com esse comandos eu removi as colunas que não vou querer usar. Imagino que não seja algo absolutamente necessário, pois deve ser possível controlar os parâmetros que vc passa par ao modelo. Mas, como ainda não vi isso, essa opção me pareceu a mais fácil de se fazer.

#Como vimos mais para cima, temos 2 linhas preechidas com NaN em 'Embarked' - Como são só duas, vou remover elas
data_treino_limpo = data_treino_limpo.dropna(subset=['Embarked'])

#Verificando os dados depois das midificações até aqui
data_treino_limpo.info()
print('---'*10) # Só para separar melhor esses dois prints.
data_treino_limpo.isnull().sum()

#OBS: NÃO CONSEGUI PREENCHER OS VALORES VAZIOS DA IDADE COM VALORES ALEATÓRIOS ENTRE A MEDIA+/-O DESVIO PADRÃO - SEI QUE É POSSÍVEL, E ALGO QUE ACHO INTERESSANTE DE SE TENTAR.
-> Portanto, vou preecher só com a média mesmo... Entretanto a mediana dada é um valor não inteiro, por isso vou arredondar o valor de 29.64 para 30.
Dos dados acima, a mediana, 50%, é 28. vamos substituir por ela agora também

data_treino_limpo_age_media = data_treino_limpo.fillna(30)
#preenchi os Nan com a media ajusatada para cima. Ngm fala que tem 10.67 anos...
data_treino_limpo_age_mediana = data_treino_limpo.fillna(28)
#preenchi os NaN com a mediana

      ----------------------------------------------------------------------------------------------------------------------------

                          Vamos ver agora as possíveis ferramentas de predição
                          
Por conta dos dados vou usar supervised learning; "How it works: This algorithm consist of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data"

Quais ferramentas vou olhar: Decision Tree, Random Forest, KNN (k- Nearest Neighbors), Linear Regression, será que k-means é interessante?
A princípio estes são suficientes
Regressão pode não ser muito bom: ela nos fornece uma resposta continua
Problema: Não temos muitos dados para treinar e testar os resultados. Por isso, o que vamos fazer é dividir o data set de treino em dois. Uma parte de treino e outra de avaliação já. Para então depois testar no grupo só de teste
Obs: TEMOS QUE VER COMO DEVEMOS DEIXAR OS DADOS, POIS TEMOS DADOS QUE NÃO SÃO NUMÉRICOS, TIPO O SEXO E LOCAL DE EMBARQUE. SERÁ QUE TEMOS QUE TRANSFORMAR ELES EM VALORES NUMÉRICOS? VOU TENTAR SEM. SE DER RUIM EU MUDO.

      ----------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------


