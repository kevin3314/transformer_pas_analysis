from pathlib import Path
import sys

import pandas as pd
import mlflow
import math
from scipy import stats


def main():
    # Suppose that path name is result/experiment_name/aggregates..
    # result/BaselineBiBartModel-wn-8e-bibart-cz-vpa-npa-cr/aggregates/eval_test/kwdlc_pred.csv
    res_path = Path(sys.argv[1])
    config_name = res_path.parts[1]
    target = res_path.parts[-1].rstrip(".csv")

    df = pd.read_csv(sys.argv[1], sep=',')

    if len(df) == 1:
        print(df)
        return

    columns = []
    uppers = []
    deltas_plus = []
    middles = []
    deltas_minus = []
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
        lower, upper = stats.t.interval(alpha=0.95,
                                        df=n - 1,
                                        loc=items.mean(),
                                        scale=math.sqrt(items.var() / n))
        middle = (upper + lower) / 2
        uppers.append(upper)
        deltas_plus.append(upper - middle)
        middles.append(middle)
        deltas_minus.append(lower - middle)
        lowers.append(lower)

    df_interval = pd.DataFrame.from_dict({'upper': uppers,
                                          'delta+': deltas_plus,
                                          'middle': middles,
                                          'delta-': deltas_minus,
                                          'lower': lowers},
                                         orient='index',
                                         columns=columns)
    print(df_interval)

    mlflow.set_experiment(target)
    with mlflow.start_run(run_name=config_name):
        for upper, delta_p, middle, delta_m, lower, column in zip(
            uppers, deltas_plus, middles, deltas_minus, lowers, columns
        ):
            mlflow.log_metric(f"{column}.upper", upper)
            mlflow.log_metric(f"{column}.delta_plus", delta_p)
            mlflow.log_metric(f"{column}.middle", middle)
            mlflow.log_metric(f"{column}.delta_minus", delta_m)
            mlflow.log_metric(f"{column}.lowers", lower)


if __name__ == '__main__':
    main()
