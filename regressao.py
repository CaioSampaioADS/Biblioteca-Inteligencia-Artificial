import numpy as np
import matplotlib.pyplot as plt
from random import *
from math import sqrt
import time

'''Classe para regressao linear com apenas uma variavel de Classificação'''
class RegressaoLinear():
    def __init__(self, classificador, previsao):
        '''inicializa todas as variaveis necessarias ao instanciar a classe'''
        self.classificador = np.array(classificador)
        self.classificadorLista = classificador
        self.previsao = np.array(previsao)
        self.previsaoLista = previsao
        self.verificaSeHouveTreinamento = False
        self.w0 = 0.1
        self.w1 = 0.1

    def VisualizarHipotese(self, x, titulo="Grafico"):
        '''Cria um gráfico com a hipotese de w0 e w1 no instante que o método é chamado'''
        x = np.array(x)
        self.hipotese = self.w0 + self.w1*x

        plt.scatter(self.classificador, self.previsao)
        plt.plot(self.classificador, self.hipotese)
        plt.title(titulo)
        plt.show()

    def Hipotese(self, x):
        '''Calcula a hipotese baseada nos valores de w0 e w0'''
        return self.w0 + self.w1*x

    def MSE(self, x):
        '''Faz o calculo do Mean square error(a media do erro) soma todas as distancias dos valores
         corretos menos a hipotese dividido pela quantidade de valores corretos'''
        x = np.array(x)
        '''Calcula a hipotese'''
        self.y = self.w0 + self.w1 * x

        mse = 0

        '''Soma todos os erros'''
        for i in range(0, len(self.y)):
            mse += (self.previsao[i] - self.y[i])**2
        '''Calcula a média'''
        mse = mse / len(self.previsao)

        return mse

    def DescidaGradienteStep(self, alpha=0.01, epocas = 5000):
        '''Atualiza os valores de w0 e w1 afim de minimizar o MSE / taxa de aprendizagem e epocas ja vem com valores pré definidos
         de 0,01 e 5000 para alterar basta passar como parametro do método'''
        '''------------------------------------------------------'''

        '''Cria uma lista com a quantidade de iterações do algoritmo para plotar o gráfico de custo'''
        self.verificaSeHouveTreinamento = True
        self.eixoX = []
        for i in range(0, epocas):
            self.eixoX.append(i)



        tamanho = len(self.classificador)
        m = float(len(self.classificadorLista))
        self.custo = []
        '''executa o algoritmo a quantidade de épocas determinada'''
        for i in range(0, epocas):
            '''cria uma lista com o valor do MSE a cada iteração do gráfico para plotar o gráfico depois'''
            self.custo.append(self.MSE(self.classificadorLista))
            '''Reseta as váriaveis'''
            erro_w0 = 0
            erro_w1 = 0

            '''Executa a somatoria dos erros dos valores preditos pelos valores corretos'''
            for j in range(0, tamanho):
                erro_w0 += self.Hipotese(self.classificadorLista[j]) - self.previsaoLista[j]
                erro_w1 += (self.Hipotese(self.classificadorLista[j]) - self.previsaoLista[j]) * self.classificadorLista[j]

                '''Atualiza os valores de w0 e w1 baseado nos erros de w0 e w1'''
                self.w0 = self.w0 - alpha * (1/m) * erro_w0
                self.w1 = self.w1 - alpha * (1/m) * erro_w1

    def Prever(self, x):
        print(self.w0 + self.w1*x)
        return self.w0 + self.w1*x

    def VisualizarGraficoErro(self):
        try:
            plt.plot(self.eixoX, self.custo)
            plt.show()
        except AttributeError:
            print("Primeiro execute o método de descida do gradiente para depois executar esse método\nFirst run the gradient descent method and then perform this method")


    '''Salva os valores de w0 e w1 em um arquivo texto de nome informado pelo usuario'''
    def SalvarTreinamento(self, NomeArquivo):
        if self.verificaSeHouveTreinamento:
            arquivo = open(NomeArquivo, 'w')
            arquivo.write(f'{self.w0}\n{self.w1}')
            arquivo.close()
        else:
            print('primeiro execute o treinamento do metodo')

    '''inicializa w0 e w1 com os valores contidos nos arquivos textos já salvos'''
    def CarregarTreinamento(self, NomeArquivo):
        arquivo = open(NomeArquivo, 'r')
        teste = arquivo.read()
        teste = teste.split()
        teste[0] = float(teste[0])
        teste[1] = float(teste[1])
        self.w0 = teste[0]
        self.w1 = teste[1]

class RegressaoLinearMultiplasVariaveis():
    def __init__(self, classificador, previsao):
        self.classificador = classificador
        self.previsao = previsao
        self.w0 = 0.1
        self.w = []
        for i in range(0, len(self.classificador[0])):
            self.w.append(uniform(-1,1))
        self.w = np.array(self.w)


    def Prever(self, x):
        x = np.array(x)
        y = x @ self.w
        y += self.w0
        return y

    def MSE(self, x, classe):
        x = np.array(x)
        y = []
        somatoria = 0


        for i in range(0, len(x)):
            aux = self.w @ x[i]
            y.append(aux + self.w0)
        '''
        print(f'w {self.w}')
        print(f'y = {y}')
        print(f'classe = {classe}')
        '''
        for i in range(0, len(y)):
            somatoria += (y[i] - classe[i])**2
            #print(somatoria)

        mse = sqrt(somatoria)

        return mse

    def DescidaGradiente(self, alpha = 0.1, epocas = 25000):
        self.eixox = []
        self.custo = []
        for i in range(0, epocas):
            self.eixox.append(i)


        m = float(len(self.classificador[0]))
        for h in range(0, epocas):
            self.custo.append(self.MSE(self.classificador, self.previsao))
            for i in range(0, len(self.classificador)):
                erro = self.Prever(self.classificador[i]) - self.previsao[i]
                #print(f'erro = {erro}')
                for j in range(0, len(self.classificador[i])):
                    self.w[j] = self.w[j] - alpha * (1 / m) * erro * (self.classificador[i][j]/1000)
                    self.w0 = self.w0 - alpha * (1 / m) * erro * 1

    def VisualizaGrafico(self):
        plt.plot(self.eixox, self.custo)
        plt.show()

    def SalvarTreinamento(self, NomeArquivo="Ws.txt"):
        arquivo = open(NomeArquivo, "w")
        arquivo.write("{:.5f}".format(self.w0))
        for x in self.w:
            arquivo.write("\n{:.5f}".format(x))

    def CarregarTreinamento(self, NomeArquivo="Ws.txt"):
        arquivo = open(NomeArquivo, "r")
        WNs = arquivo.read()
        WNs = WNs.split()

        for i in range(0, len(WNs)):
            WNs[i] = float(WNs[i])

        self.w0 = WNs[0]
        aux = []
        for i in range(1, len(WNs)):
            aux.append(WNs[i])

        self.w = np.array(aux)

'''Formato no qual o dados devem ser passado'''

'''
Exemplo de como utilizar a regressao linear simples
x = [15,16,17,18,19,20]
y = [2, 3, 4, 5, 6, 7]
reg = RegressaoLinear(x, y)
reg.DescidaGradienteStep()
reg.Prever(20)
reg.VisualizarGraficoErro()
'''


'''Exemplo regressao linear multipla (Ainda com os bugs de passar valores como 15, 16, 17)'''
x = [[15, 20], [16, 20], [16, 30], [17, 20], [18, 30], [19, 40], [20, 40], [21, 50]]
y = [1, 1.5, 2, 2, 3, 4, 4.5, 5]

reg = RegressaoLinearMultiplasVariaveis(x, y)
reg.CarregarTreinamento()
print(reg.Prever([16,30]))









