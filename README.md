**Biblioteca Inteligência Artificial**

**Essa é uma Biblioteca construida em pyton para facilitar e abstrair conceitos, fazendo com que as pessoas que queiram implementar algoritmos de Machine Learning não se preocupem tanto com a lógica e matemática, se preocupando principalmente com os resultados do algoritmo**

**Instalação:** 
com o git já instalado utilize

    git clone https://github.com/CaioSampaioADS/Biblioteca-Inteligencia-Artificial.git

**Importação**
O projeto e dividido da seguinte forma:

todos os métodos estão contidos dentro da pasta principal, mas cada método esta separado dentro de um arquivo .py diferente

Caso o arquivo .py esteja da pasta da biblioteca utilize from NomeDoMetodo import NomeDaClasse, segue alguns exemplos

    from KNN import KNN

    from regressao import RegressaoLinear

    from regressao import RegressaoLinearMultiplasVariaveis


**Métodos implementados**

**KNN**

Esse é um algoritmos de classificação onde busca pelos dados de treinamento mais proximos do dado de teste

Os dados para esse algoritmo devem ser passados da seguinte forma: 

uma variavel contendo listas de listas contendo os dados de treinamento:

    x = [[1,2,3], [1,2,3]]

onde cada sub lista é um determinado conjunto de dados
e para cada sub lista deve haver os valores de classe de forma numérica:

    y = [1,0]

o 1 nesse caso representa a primeira sub lista e o 0 a segunda sendo 1 e 0 Classe A e Classe B podendo conter 2 ou mais classes nos dados de treinamento

Para executar o algoritmo devemos instanciar a classe:

    knn = KNN(x, y)

passando X e Y como dados de treinamento

logo após para fazer uma classificação utilizamos:

    knn.Classificar([4, 4, 4])

e para x com 2 valores em cada sub listas é possivel criar o gráfico da amostragem dos dados:

    knn.VisualizarGrafico()

**Regressão linear Multipla**

Esse algoritmo busca traçar uma melhor "reta/plano" para um determinado conjunto de dados.
A forma na quais os dados devem ser passados é igual ao KNN sendo X listas de listas e Y listas

para executar a Regressao Linear Multipla devemos primeiro instanciar a classe

    reg = RegressaoLinearMultiplasVariaveis(x, y)

Sendo X e Y os dados de treinamento

Após esse passo treinar o algoritmo

    reg.DescidaGradiente()

E para fazer uma previsão:

    print(reg.Prever([15, 20]))

Por final você pode salvar e carregar o treinamento utilizando os métodos

    reg.SalvarTreinamento()
    reg.CarregarTreinamento()


**Regressão Linear Simples**

Esse método é o mesmo que a Regressao Linear Mulplipla a unica diferença é que ele só possui 2 dimensões sendo x para os dados
que buscamos fazer a previsão e y o valor da previsão:

    x = [1,2,3,4]

    y = [1,2,3,4]

Para executar a Regressao Linear Simples devemos primeiro instanciar a classe:

    reg = RegressaoLinear(x, y)

Sendo X e Y os dados de treinamento

Após esse passo treinar o algoritmo:

    reg.DescidaGradienteStep()

E para fazer uma previsão:

    reg.Prever(20)
    
E existe alguns métodos Opcionais como 
    
    reg.VisualizarHipotese(x, "Titulo do grafico opcional")
    reg.VisualizarGraficoErro()
    reg.SalvarTreinamento("Nome do arquivo")
    reg.CarregarTreinamento("NomeArquivo")
    
    
Caso queira entender melhor cada algoritmo: https://www.udemy.com/course/masterclass-algoritmos-de-machine-learning/

