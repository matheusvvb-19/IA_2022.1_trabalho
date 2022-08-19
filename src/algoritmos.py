import typing, random
import pandas as pd
import numpy as np


def k_means(dataset: typing.TextIO, k: int, it: int):
    """ Algoritmo K-Médias
        args:
        dataset -- arquivo contento o conjunto de dados
        k -- número de clusters desejado
        it -- número de iterações desejado
    """

    # leitura dos dados:
    df = pd.read_csv(dataset, sep="\t")

    # captura do intervalo de valores de cada atributo (coluna de dados) do dataset:
    intervalosPorAtributo = []
    for col in df.columns[1:]:
        intervalo = df[col].agg(['min', 'max'])
        intervalosPorAtributo.append(intervalo)

    # escolha aleatória dos centróides iniciais:
    centroides = []
    for i in range(k):
        ponto = ()
        for intervalo in intervalosPorAtributo:
            ponto += (random.randint(intervalo[0], intervalo[1]), )
        
        centroides.append(ponto)

    


def single_link(dataset: typing.TextIO, k_min: int, k_max: int):
    """ Algoritmo Single-Link
        args:
        dataset -- arquivo contento o conjunto de dados
        k_min -- número inicial do intervalo do número de clusters
        k_max -- número final do intervalo do número de clusters
    """

    pass


def complete_link(dataset: typing.TextIO, k_min: int, k_max: int):
    """ Algoritmo Complete-Link
        args:
        dataset -- arquivo contento o conjunto de dados
        k_min -- número inicial do intervalo do número de clusters
        k_max -- número final do intervalo do número de clusters
    """

    pass

def euclidean_dist(point, data):
    """ Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))