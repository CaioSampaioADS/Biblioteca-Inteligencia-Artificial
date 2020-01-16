from math import sqrt
import matplotlib.pyplot as plt

'''Algoritmo KNN aplicada a Multiplas classes e variaveis '''
class KNN():
    '''metodo construtor inicializa como global as variaveis de 'treinamento' '''
    def __init__(self, treinamento, classe):
        self.treinamento = treinamento
        self.classe = classe


    '''Dada a distancia Euclidiana identifica as mais proximas e quais classe ela pertence'''
    def IdentificaClasse(self, qntKProximos):
        DistanciaEuclidianaOrdenada = sorted(self.DistanciaEuclidiana)
        '''Posiçao na lista dos k elementos mais proximos'''
        PosicoesKProximos = []
        ClasseKProximos = []
        QuantidadeClassesProximas = []

        '''Dado a quantidade de k mais proximos separa de todas as distancias euclidiana as k mais proximas'''
        for i in range(0, qntKProximos):
            for j in range(0, len(self.DistanciaEuclidiana)):
                if DistanciaEuclidianaOrdenada[i] == self.DistanciaEuclidiana[j]:
                    PosicoesKProximos.append(j)
                    '''Após identificar um k proximo atribui -1 para caso haja distancias iguais em uma mesma base de dados'''
                    self.DistanciaEuclidiana[j] = -1

        '''De cada posiçao de k mais proximos identifica sua classe'''
        for i in range(0, len(PosicoesKProximos)):
            ClasseKProximos.append(self.classe[PosicoesKProximos[i]])

        '''Identificando quais as classes mais encontradas'''

        '''dado as classes de k mais proximos identifica a quantidade de cada classe que possui'''
        ClasseKProximosSemRepeticoes = sorted(set(ClasseKProximos))
        for i in range(0, len(ClasseKProximosSemRepeticoes)):
            somatoria = 0
            for j in range(0, len(ClasseKProximos)):
                if ClasseKProximosSemRepeticoes[i] == ClasseKProximos[j]:
                    somatoria += 1

            QuantidadeClassesProximas.append(somatoria)

        '''depois de identificar a quantidade de cada classe identifica a com maior quantidade'''
        aux = sorted(QuantidadeClassesProximas, reverse=True)
        MaiorQuantidadeDeClassesEncontradas = aux[0]
        posicaoClasse = 0
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

'''Formato no qual o dados devem ser passado'''
x = [[1, 2, 4], [2, 1, 4], [1, 1, 4], [2, 2, 4], [4, 4, 4], [5, 4, 4], [3, 5, 4], [5, 6, 4]]
y = [0,0,0,0,1,1,1,1]



knn = KNN(x, y)
knn.Classificar([4, 4, 4])







