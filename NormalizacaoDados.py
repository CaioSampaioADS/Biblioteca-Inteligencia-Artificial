from math import trunc


class Normalizacao():
    def GrandezaNormalizacao(self, normalizar):
        escalaGrandeza = []

        for i in range(0, len(normalizar)):
            escalaGrandeza.append(len(str(trunc(normalizar[i]))))

        escalaGrandeza = sorted(escalaGrandeza, reverse=True)

        dividir = 1

        for i in range(0, escalaGrandeza[0]):
            dividir *= 10

        for i in range(0, len(normalizar)):
            normalizar[i] = normalizar[i] / dividir

        return normalizar