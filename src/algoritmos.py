import typing
import random
import math
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score


def k_means(dataset: str, k: int, it: int):
    """ Algoritmo K-Médias
        args:
        dataset -- arquivo contendo o conjunto de dados
        k -- número de clusters desejado
        it -- número de iterações desejado
    """

    # lendo objetos de entrada:
    df = pd.read_csv(dataset, sep="\t", header=0)

    # criando uma lista formada por tuplas, onde cada tupla apresenta o menor e o maior valor para a respectiva coluna de dados no dataset original:
    rangeValuesPerAttribute = []
    for col in df.columns[1:]:
        intervalo = df[col].agg(['min', 'max'])
        rangeValuesPerAttribute.append(intervalo)

    # calculando centróides aleatórios iniciais:
    centroids = []
    for i in range(k):
        while True:
            ponto = ()
            for intervalo in rangeValuesPerAttribute:
                ponto += (random.uniform(intervalo[0], intervalo[1]), )

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
                selectedObjValues = df.loc[df['sample_label'].isin(
                    selectedObjNames)]
                selectedObjValues = selectedObjValues.drop(
                    columns=['sample_label'])
                selectedObjValues = selectedObjValues.to_numpy().tolist()
                for index, m in enumerate(selectedObjValues):
                    selectedObjValues[index] = tuple(m)

                centroids[c] = recalculate_centroid(selectedObjValues)

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

    Path(f'../output/').mkdir(parents=True, exist_ok=True)
    dfPartition.to_csv(f'../output/kmeans_%s_%d.clu' % (dataset.split('/')[-1][0:-4], k), sep='\t', index=False)


def hierarquic_link(dataset: str, k_min: int, k_max: int, strategy: str):
    """ Algoritmo Single-Link
        args:
        dataset -- arquivo contendo o conjunto de dados
        k_min -- número inicial do intervalo do número de clusters
        k_max -- número final do intervalo do número de clusters
    """

    if strategy == 'single-link':
        targetFolder = '../output/single_link/'
        file = 'single'

    elif strategy == 'complete-link':
        targetFolder = '../output/complete_link/'
        file = 'complete'

    # lendo arquivo de dados:
    # MUDAR sep=',' QUANDO FOR TESTAR ENTRADA TEST.TXT
    df = pd.read_csv(dataset, sep="\t", header=0)

    # criando pasta de saída do algoritmo e já escrevendo a partição para primeiro corte do dendograma (K = n° de objetos):
    Path(targetFolder).mkdir(parents=True, exist_ok=True)

    # tabela de distâncias euclidianas entre os objetos inicias do conjunto de dados:
    dfDistances, listOfObjects = initial_distances_between_objects(df)

    # somente se o parâmetro k_max for igual à quantidade de objetos do dataset é que se deverá criar o arquivo para esta partição inicial:
    # nesta partição inicial, cada cluster é composto por apenas um objeto
    if k_max == len(df.index):
        partition = {
            'object': [],
            'cluster': []
        }

        i = 0
        # percorrendo as colunas da tabela inicial de distâncias, atrelando um número de cluster diferente a cada objeto:
        columns = dfDistances.columns.values.tolist()
        for col in columns:
            partition['object'].append(col)
            partition['cluster'].append(i)
            i += 1

        # criando dataframe a partir do docionário - será usado para gerar arquivo de saída final do algoritmo:
        dfPartition = pd.DataFrame(partition)
        dfPartition.to_csv(targetFolder + file + '_%s_%d.clu' %
                           (dataset.split('/')[-1][0:-4], len(df.index)), sep='\t', index=False)

    # iteração principal do algoritmo:
    numObjs = len(df.index)
    for it in range(numObjs, k_min-1, -1):
        if it != 1:
            # variável dicionário que armazena a menor distância da tabela de distâncias euclidianas, assim como os nomes dos objetos que compõem essa menor distância:
            columns = dfDistances.columns.tolist()

            minOfEveryColumnIdx = dfDistances[dfDistances > 0].idxmin()
            minOfEveryColumnValues = dfDistances[dfDistances > 0].min()

            minValue = {
                'distance': -1,
                'object1': '',
                'object2': ''
            }

            # encontrando a menor distância na tabela de distâncias:
            for i in columns:
                value = minOfEveryColumnValues[i]
                if minValue['distance'] == -1 or value < minValue['distance']:
                    minValue['distance'] = value
                    minValue['object1'] = i
                    minValue['object2'] = minOfEveryColumnIdx[i]

            # dicionário que armazenará dados da nova coluna e linha a ser adicionada à tabela de distâncias:
            newCluster = {
                'name': '',
                'distances': [0],
            }

            # nome do novo objeto:
            newCluster['name'] = minValue['object1'] + \
                '_' + minValue['object2']

            # percorrendo a tabela de distâncias e selecionando as menores distâncias para os objetos que foram unidos:
            columns = dfDistances.columns.values.tolist()
            for col in columns:
                # pular colunas com nomes object1 e object2, ou seja, dos objetos que foram unidos:
                if col != minValue['object1'] and col != minValue['object2']:
                    if dfDistances[col][minValue['object1']] != 0:
                        if strategy == 'single-link':
                            aux = min(
                                dfDistances[col][minValue['object1']], dfDistances[col][minValue['object2']])

                        elif strategy == 'complete-link':
                            aux = max(
                                dfDistances[col][minValue['object1']], dfDistances[col][minValue['object2']])

                        # registrando a menor distância no dicionário criado:
                        newCluster['distances'].append(aux)

            # dropando colunas e linhas dos objetos que foram unidos:
            dfDistances.drop(
                columns=[minValue['object1'], minValue['object2']], inplace=True)
            dfDistances.drop(
                [minValue['object1'], minValue['object2']], axis=0, inplace=True)

            # inserindo nova coluna e linha de distâncias para o novo cluster (objetos unidos):
            dfDistances.insert(
                0, newCluster['name'], newCluster['distances'][1:], True)
            data = []
            data.insert(0, pd.Series(
                newCluster['distances'], index=dfDistances.columns))
            dfDistances = pd.concat(
                [pd.DataFrame(data, index=[newCluster['name']]), dfDistances])

            # atualizando a lista de objetos após a união dos objetos mais próximos:
            listOfObjects = [x for x in listOfObjects if x[0] not in [
                minValue['object1'], minValue['object2']]]    # removendo objetos unidos
            # inserindo novo objeto resltado da união
            listOfObjects.insert(0, [newCluster['name']])

            # caso a esteja no intervalo desejado - corte no dendograma:
            if it >= k_min + 1 and it <= k_max + 1:
                write_link_partition(dataset, dfDistances,
                                     it, targetFolder, file)

        else:
            partition = {
                'object': [],
                'cluster': []
            }

            # percorrendo as colunas da tabela inicial de distâncias, atrelando um número de cluster diferente a cada objeto:
            columns = dfDistances.columns.values.tolist()
            for col in columns:
                objs = col.split('_')
                for obj in objs:
                    partition['object'].append(obj)
                    partition['cluster'].append(0)

            # criando dataframe a partir do docionário - será usado para gerar arquivo de saída final do algoritmo:
            dfPartition = pd.DataFrame(partition)
            dfPartition.to_csv(targetFolder + file + '_%s_%d.clu' %
                               (dataset.split('/')[-1][0:-4], 1), sep='\t', index=False)


def write_link_partition(dataset: str, dfDistances: pd.DataFrame, it: int, targetFolder: str, file: str):
    """ Escreve a partição para um determinado corte do dendograma gerado pelos algoritmos hierárquicos.
        args:
        dataset -- nome do conjunto de dados, utilizadopara salvamento do arquivo de saída.
        dfDistances -- tabela de distância euclidiana entre os objetos/clusters da particão atual.
        it -- número de clusters do corte atual.

    """
    partition = {
        'object': [],
        'cluster': []
    }

    i = 0
    # percorrendo as colunas da tabela de distâncias:
    columns = dfDistances.columns.values.tolist()
    for col in columns:
        # extraindo nomes de objetos individuais, caso haja uniões:
        objs = col.split('_')
        for obj in objs:
            partition['object'].append(obj)

            # se o corte do dendograma for 1, significa que só há 1 cluster e todos os objetos pertencem a ele, por tanto, o número do cluster será zero para todos:
            if it == 1:
                partition['cluster'].append(0)

            # se não, escreve-se um novo cluster no arquivo de saída:
            else:
                partition['cluster'].append(i)

        i += 1

    # criando dataframe a partir do docionário - será usado para gerar arquivo de saída final do algoritmo:
    dfPartition = pd.DataFrame(partition)
    dfPartition.to_csv(targetFolder + file + '_%s_%d.clu' %
                       (dataset.split('/')[-1][0:-4], it-1), sep='\t', index=False)


def initial_distances_between_objects(df: pd.DataFrame):
    """ Calcula a distância euclidiana entre todos os objetos (individualmente) do conjunto de dados
        args:
        df -- tabela de conjuntos de dados, lida a partir de arquivo

    """
    # lista de nomes dos objetos:
    objectsNames = df['sample_label'].tolist()

    # dicionário que será transformado na tabela de distâncias euclidianas:
    distanceTable = {}
    for j in objectsNames:
        distanceTable.update({j: []})

    # iterando os objetos do dataset e preenchendo a tabela de distâncias (NxN):
    listOfObjects = df.to_numpy().tolist()
    for index_i, i in enumerate(listOfObjects):
        for index_j, j in enumerate(listOfObjects):
            # diagonal principal:
            if index_i == index_j:
                distance = 0

            # demais céulas da tabela:
            else:
                distance = euclidean_dist(tuple(i[1:]), tuple(j[1:]))

            distanceTable[i[0]].append(distance)

    return pd.DataFrame(distanceTable, index=objectsNames), listOfObjects


def euclidean_dist(point1: tuple, point2: tuple):
    """ Calcula a distância euclidiana entre duas tuplas de N dimensões.
        args:
        point1 -- tupla origem
        point2 -- tupla destino
    """

    sum = 0
    # itera sobre os elementos de ambas as tuplas, calculando o quadrado da diferença entre eles:
    for p, c in zip(point1, point2):
        sum += (p - c)**2

    return math.sqrt(sum)


def recalculate_centroid(objs: list):
    """ Recalcula o centróide de um cluster a partir dos objetos que o compõem.
        args:
        objs -- lista de objetos que compõem o cluster. O centróide será a média de cada um de seus atributos.

    """

    # número de dimensões (coordenadas, atributos) dos objetos na lista:
    numberDimensions = len(objs[0])

    # criando novo centróide preenchido por zeros:
    sum = []
    for i in range(numberDimensions):
        sum.append(0)

    # somando os atributos de cada objeto presente na lista:
    for obj in objs:
        for i in range(numberDimensions):
            sum[i] += obj[i]

    # calculando a média para cada atributo, após todas as somas:
    for index, s in enumerate(sum):
        sum[index] = s/len(objs)

    return tuple(sum)   # retornando resultado na forma de tupla


def plot_partition(partition: str, dataset: str):
    """ Plota os dados divididos em clusters. Pode receber a partição real ou a partição obtida pelos algoritmos implementados.
        args:
        partition -- nome do arquivo com a partição
        dataset -- nome do arquivo com os dados (atributos) dos objetos
    """

    # lendo partição (divisão dos objetos em cluster):
    # se a partição passada for a real, é preciso renomear suas colunas:
    if 'Real' in partition:
        partitionDf = pd.read_csv(partition, sep="\t", header=None)
        partitionDf.set_axis(['object', 'cluster'], axis=1, inplace=True)

    # se não, basta ler o arquivo e os nomes das colunas já estarão nele:
    else:
        partitionDf = pd.read_csv(partition, sep="\t", header=0)

    # lendo objetos de entrada:
    df = pd.read_csv(dataset, sep="\t", header=0)

    # plotando gráfico:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # eixos e clusters:
    try:
        x = df['d1'].to_numpy().tolist()
        plt.xlabel('d1')

    except KeyError:
        x = df['D1'].to_numpy().tolist()
        plt.xlabel('D1')

    try:
        y = df['d2'].to_numpy().tolist()
        plt.ylabel('d2')

    except:
        y = df['D2'].to_numpy().tolist()
        plt.ylabel('D2')

    clusters = partitionDf['cluster'].to_numpy().tolist()
    scatter = ax.scatter(x, y, c=clusters, s=50)

    # definindo o título do gráfico:
    if 'Real' in partition:
        titleName = 'Base de dados: {}\nClusters: {}'.format(
            partition.split('/')[2].split('Real')[0], len(set(clusters)))

    else:
        """formatos: 
                '../output/single_link/single_<entrada>_<numClusters>.clu'
                '../output/complete_link_complete_<entrada>_<numClusters>.clu'
        """
        if 'kmeans' in partition:
            algoritmo = 'K-Médias'
            dados = partition.split('_')[1][0:-4]
            numClusters = len(set(clusters))

        else:
            if 'single' in partition:
                algoritmo = 'Single-Link'

            else:
                algoritmo = 'Complete-Link'

            dados = partition.split('/')[3].split('_')[1]
            numClusters = partition.split('/')[3].split('_')[2][0:-4]

        titleName = '{}\n{} - {} clusters'.format(
            algoritmo, dados, numClusters)

    plt.title(titleName)
    plt.show()


def adjusted_rand_index(realPartitionPath: str, testPartitionPath: str):
    """ Retorna o Índice de Rand Ajustado para determinada partição obtida, em detrimento da partição real já conhecida.
        args:
        realPartitionPath: caminho para o arquivo que contém a partição real do conjunto de dados, está na pasta datasets;
        testPartitionPath: caminho para o arquivo que contém a partição gerada pelos nossos algoritmos, está na pasta output.
    
    """

    # lendo o conjunto de dados de entrada:
    dfReal = pd.read_csv(realPartitionPath, sep="\t")

    # Armazena a divisão dos objetos nos cluster da partição real
    realPartition = []
    for val in dfReal.to_numpy().tolist():
        realPartition.append(val[1])

    # Armazena a divisão dos objetos nos cluster da partição que será calculado o IR
    dfTest = pd.read_csv(testPartitionPath, sep="\t", header=0)
    testPartition = []
    for val in dfTest.to_numpy().tolist()[1:]:
        testPartition.append(val[1])

    # Função que calcula o AR
    return adjusted_rand_score(realPartition, testPartition)

def ar_bar_plot(dataset: str):
    """ Plota um gráfico de barras com os AR's decada algortimo aplicado ao conjunto de dados desejado.
        args:
        dataset -- nome do arquivo de entrada que se deseja analisar os AR's.
    """
    
    # caminho para o arquivo que contém a partição real daquele dataset:
    realPartitionPath = glob.glob('../datasets/*Real*.clu')
    realPartitionPath = [x for x in realPartitionPath if dataset in x]
    realPartitionPath = realPartitionPath[0]

    # lista de caminhos para arquivos com as partições geradas para aquele dataset:
    testPartitionsPath = glob.glob('../output/**/*.clu', recursive=True)
    testPartitionsPath.sort(key=len)
    testPartitionsPath = [x for x in testPartitionsPath if dataset in x]

    # lista de caminhos específicos, para cada algoritmo:
    testKMeansPartitionsPath = [x for x in testPartitionsPath if 'kmeans' in x]
    testSinglePartitionsPath = [x for x in testPartitionsPath if 'single' in x]
    testCompletePartitionsPath = [x for x in testPartitionsPath if 'complete' in x]
    
    # valores de K (quantidade de partições obtidas):
    labels = []
    for p in testPartitionsPath:
        label = p.split('.clu')[0]
        label = label.split('_')[-1]
        label = int(label)

        if label not in labels:
            labels.append(label)
    
    labels = sorted(labels)

    # calculando e armazenando os AR's para algoritmo e cada partição:
    kmeansARs = []
    singleLinkARs = []
    completeLinkARs = []
    for p in testKMeansPartitionsPath:
        kmeansARs.append(round(adjusted_rand_index(realPartitionPath, p), 5))

    for p in testSinglePartitionsPath:
        singleLinkARs.append(round(adjusted_rand_index(realPartitionPath, p), 5))

    for p in testCompletePartitionsPath:
        completeLinkARs.append(round(adjusted_rand_index(realPartitionPath, p), 5))

    # configurações do gráfico:
    if dataset == 'monkey':
        plt.rcParams["figure.figsize"] = (15,8)

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, kmeansARs, width, label='K-Médias')
    rects2 = ax.bar(x, singleLinkARs, width, label='Single-Link')
    rects3 = ax.bar(x + width, completeLinkARs, width, label='Complete-Link')

    ax.set_ylabel('AR')
    ax.set_xlabel('K')
    ax.set_title('AR - %s' % dataset)
    ax.set_xticks(x, labels)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()