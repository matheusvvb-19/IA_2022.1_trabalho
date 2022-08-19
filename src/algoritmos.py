import typing, random, math
import pandas as pd
import numpy as np


def k_means(dataset: typing.TextIO, k: int, it: int):
    """ Algoritmo K-Médias
        args:
        dataset -- arquivo contento o conjunto de dados
        k -- número de clusters desejado
        it -- número de iterações desejado
    """

    df = pd.read_csv(dataset, sep="\t", header=0)

    rangeValuesPerAttribute = []
    for col in df.columns[1:]:
        intervalo = df[col].agg(['min', 'max'])
        rangeValuesPerAttribute.append(intervalo)

    centroids = []
    for i in range(k):
        while True:
            ponto = ()
            for intervalo in rangeValuesPerAttribute:
                ponto += (random.randint(intervalo[0], intervalo[1]), )

            if ponto not in centroids:
                break
            
        centroids.append(ponto)

    # iterações do algoritmo:
    for i in range(it):
        # esqueleto da tabela de distância euclidiana entre cada objeto e os centróides/clusters:
        distanceTable = {
            'object': [],
        }

        partition = {
            'object': [],
            'cluster': []
        }

        # adicionando uma nova coluna à tabela de distância euclidiana para cada cluster existente:
        for j in range(k):
            distanceTable.update({'c%d' % j: []})

        # convertendo dataset em uma lista de listas, onde cada elemento contém todas as informações sobre determinado objeto:
        listOfObjects = df.to_numpy().tolist()

        # percorrendo a lista de objetos:
        for object in listOfObjects:
            distanceTable['object'].append(object[0])

            objectPoint = tuple(object[1:])

            # calcular a distância euclidiana entre o objeto e cada um dos centróides existentes:
            for index, c in enumerate(centroids):
                euclidianDistance = euclidean_dist(objectPoint, c)
                distanceTable['c%d' % index].append(euclidianDistance)

        dfDistances = pd.DataFrame(distanceTable)

        # reagrupar os objetos, associando eles aos clusters mais próximos:
        listOfDistances = dfDistances.to_numpy().tolist()

        # para cada linha da tabela de distâncias euclidianas, selecionar a menor distância e associar o objeto ao determinado cluster:
        for dist in listOfDistances:
            minorDistIndex = np.where(dist[1:] == np.amin(dist[1:]))[0]

            # se houver mais de uma distância com o menor valor, é feita uma escolha aletória entre elas:
            if len(minorDistIndex) > 1:
                minorDistIndex = random.choice(minorDistIndex)
            
            else:
                minorDistIndex = minorDistIndex[0]

            # preenchendo o dicionário de partição:
            partition['object'].append(dist[0])
            partition['cluster'].append(minorDistIndex)

        # criando dataframe a partir do docionário - será usado para gerar arquivo de saída final do algoritmo:
        dfPartition = pd.DataFrame(partition)

        # para cada um dos clusters, selecionar os objetos que estão associados a ele:
        for c in range(k):
            selectedObjNames = dfPartition.loc[dfPartition['cluster'] == c]
            selectedObjNames = selectedObjNames['object'].to_numpy().tolist()

            if len(selectedObjNames) > 0:
                selectedObjValues = df.loc[df['sample_label'].isin(selectedObjNames)]
                selectedObjValues = selectedObjValues.drop(columns=['sample_label'])
                selectedObjValues = selectedObjValues.to_numpy().tolist()
                for index, m in enumerate(selectedObjValues):
                    selectedObjValues[index] = tuple(m)

                centroids[c] = recalculate_centroid(selectedObjValues)

    dfPartition.to_csv('./output/kmeans.clu', sep='\t', index=False)



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

def euclidean_dist(point1, point2):
    """ Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    sum = 0
    for p, c in zip(point1, point2):
        sum += (p - c)**2

    return math.sqrt(sum)

def recalculate_centroid(objs: list):
    numberDimensions = len(objs[0])
    sum = []
    for i in range(numberDimensions):
        sum.append(0)

    for obj in objs:
        for i in range(numberDimensions):
            sum[i] += obj[i]

    for index, s in enumerate(sum):
        sum[index] = s/len(objs)

    return tuple(sum)