from pathlib import Path
import sys

import pandas as pd
import mlflow
import math
import numpy as np
from scipy import stats


def _log_metric(key, value):
    if value is None or np.isnan(value):
        print(f"{key} is {str(value)}. Skip it.")
        return
    mlflow.log_metric(key, value)


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

    config_name = config_name.replace("_", "-")
    # BaselineBiBartModel-wn-8e-bibart-cz-vpa-npa-cr-yomiuri-ja
    config_components = config_name.split("-")

    backbone_name = config_components[3]

    # Get additional name
    # TODO: This setting ignores refine/conditional/interative model
    attr_names = ["debug", "cr", "bar", "npa", "vpa"]
    idx = None
    for attr in attr_names:
        if attr not in config_components:
            continue
        idx = config_components.index(attr)
        break
    exp_settings = config_components[4:idx]
    additional_names = config_components[idx+1:]
    full_target = target + "-" + "-".join(exp_settings)

    run_name = ""
    run_name += backbone_name + "-"
    run_name += "-".join(additional_names)
    run_name = run_name.rstrip("-")
    run_name += "-" + "-".join(config_components[:3])

    full_target = full_target.replace("_", "-").rstrip("-")
    run_name = run_name.replace("_", "-").rstrip("-")

    print(full_target, run_name)
    mlflow.set_experiment(full_target)
    with mlflow.start_run(run_name=run_name):
        for upper, delta_p, middle, delta_m, lower, column in zip(
            uppers, deltas_plus, middles, deltas_minus, lowers, columns
        ):
            _log_metric(f"{column}.upper", upper)
            _log_metric(f"{column}.delta_plus", delta_p)
            _log_metric(f"{column}.middle", middle)
            _log_metric(f"{column}.delta_minus", delta_m)
            _log_metric(f"{column}.lowers", lower)

    simple_target = target + "-simple-" + "-".join(exp_settings)
    simple_target = simple_target.replace("_", "-").rstrip("-")
    mlflow.set_experiment(simple_target)
    print(simple_target, run_name)
    with mlflow.start_run(run_name=run_name):
        for middle, column in zip(
            middles, columns
        ):
            _log_metric(f"{column}.middle", middle)


if __name__ == '__main__':
    main()
