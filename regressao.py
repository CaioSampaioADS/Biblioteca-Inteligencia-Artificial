import numpy as np
import matplotlib.pyplot as plt


'''Classe para regressao linear com apenas uma variavel de Classificação'''
class RegressaoLinear():
    def __init__(self, classificador, previsao):
        '''inicializa todas as variaveis necessarias ao instanciar a classe'''
        self.classificador = np.array(classificador)
        self.classificadorLista = classificador
        self.previsao = np.array(previsao)
        self.previsaoLista = previsao
        self.w0 = 0.1
        self.w1 = 0.1

    def VisualizarHipotese(self, x, titulo = "Grafico"):
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



'''Formato no qual o dados devem ser passado'''
x = [15,16,17,18,19,20]
y = [2, 3, 4, 5, 6, 7]

reg = RegressaoLinear(x, y)
reg.DescidaGradienteStep(epocas=5000)
reg.Prever(25)









