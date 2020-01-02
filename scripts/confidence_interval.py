import sys

import pandas as pd
import math
from scipy import stats


def main():
    df = pd.read_csv(sys.argv[1], sep=',')

    columns = []
    uppers = []
    lowers = []
    for column, items in df.iteritems():
        columns.append(column)
        n = len(items)
        """
        alpha: 何パーセント信頼区間か
        df: t分布の自由度
        loc: 平均 X bar
        scale: 標準偏差 s
        """
        interval = stats.t.interval(alpha=0.95,
                                    df=n - 1,
                                    loc=items.mean(),
                                    scale=math.sqrt(items.var() / n))
        uppers.append(interval[0])
        lowers.append(interval[1])

    df_interval = pd.DataFrame.from_dict({'upper': uppers, 'lower': lowers},
                                         orient='index',
                                         columns=columns)
    print(df_interval)


def sample():
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    setosa = iris[iris['class'] == 'Iris-setosa']
    virginica = iris[iris['class'] == 'Iris-virginica']
    versicolor = iris[iris['class'] == 'Iris-versicolor']

    print(setosa['sepal_width'].mean())

    print(setosa['sepal_width'].var())

    n = 300
    """
    alpha: 何パーセント信頼区間か
    df: t分布の自由度
    loc: 平均 X bar
    scale: 標準偏差 s
    """
    interval = stats.t.interval(alpha=0.95,
                                df=n - 1,
                                loc=setosa.iloc[:n]['sepal_width'].mean(),
                                scale=math.sqrt(setosa.iloc[:n]['sepal_width'].var() / n))
    print(interval)


if __name__ == '__main__':
    main()
