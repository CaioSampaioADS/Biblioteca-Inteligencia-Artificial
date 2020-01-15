from math import sqrt
import matplotlib.pyplot as plt

class KNN():
    def __init__(self, treinamento, classe):
        self.treinamento = treinamento
        self.classe = classe

    def IdentificaClasse(self, qntKProximos):
        DistanciaEuclidianaOrdenada = sorted(self.DistanciaEuclidiana)
        PosicoesKProximos = []
        ClasseKProximos = []
        QuantidadeClassesProximas = []
        for i in range(0, qntKProximos):
            for j in range(0, len(self.DistanciaEuclidiana)):
                if DistanciaEuclidianaOrdenada[i] == self.DistanciaEuclidiana[j]:
                    PosicoesKProximos.append(j)
                    self.DistanciaEuclidiana[j] = -1

        for i in range(0, len(PosicoesKProximos)):
            ClasseKProximos.append(self.classe[PosicoesKProximos[i]])

        '''identificando quais as classes mais encontradas'''

        ClasseKProximosSemRepeticoes = sorted(set(ClasseKProximos))
        for i in range(0, len(ClasseKProximosSemRepeticoes)):
            somatoria = 0
            for j in range(0, len(ClasseKProximos)):
                if ClasseKProximosSemRepeticoes[i] == ClasseKProximos[j]:
                    somatoria += 1

            QuantidadeClassesProximas.append(somatoria)

        aux = sorted(QuantidadeClassesProximas, reverse=True)
        MaiorQuantidadeDeClassesEncontradas = aux[0]

        for i in range(0 ,len(QuantidadeClassesProximas)):
            if QuantidadeClassesProximas[i] == MaiorQuantidadeDeClassesEncontradas:
                posicaoClasse = i



        ClasseResultante = ClasseKProximosSemRepeticoes[posicaoClasse]

        print(ClasseResultante)




    def Classificar(self, prever, quantidadeKProximos = 3):
        self.prever = prever
        self.DistanciaEuclidiana = []
        for i in range(0, len(self.treinamento)):
            somatoria = 0
            for j in range(0, len(self.prever)):
                somatoria += (self.treinamento[i][j] - self.prever[j])**2
            somatoria = sqrt(somatoria)

            self.DistanciaEuclidiana.append(somatoria)

        self.IdentificaClasse(quantidadeKProximos)

    def VisualizarGrafico(self):
        for i in range(0, len(self.treinamento)):
            plt.scatter(self.treinamento[i][0], self.treinamento[i][1], color='red')

        plt.scatter(self.prever[0], self.prever[1])
        plt.show()


x = [[1, 2], [2, 1], [1, 1], [2, 2], [4, 4], [5, 4], [3, 5],[5, 6]]
y = [0,0,0,0,1,1,1,1]
knn = KNN(x, y)
knn.Classificar([0, 0])
knn.VisualizarGrafico()






