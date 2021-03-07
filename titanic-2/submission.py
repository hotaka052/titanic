import numpy as np
import pandas as pd
import xgboost as xgb
from train import train
from data_preprocessing import test_dataset

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for xgboost"
)

parser.add_argument("--seed", default = 71, type = int,
                    help = "乱数を固定するシード値")
parser.add_argument("--eta", default = 0.05, type = float,
                    help = "モデルの学習率")
parser.add_argument("--max_depth", default = 3, type = int,
                    help = "決定木の最大深さ")
parser.add_argument("--min_child_weight", default = 4,  type = int,
                    help = "min_child_weight")
parser.add_argument("--early_stopping", default = 4, type = int,
                    help = "何回、損失が更新されなかった時学習を止めるか")

args = parser.parse_args()

def submission():
    sub = pd.read_csv('../data/gender_submission.csv')

    x_test = test_dataset()
    dtest = xgb.DMatrix(x_test)

    model = train(
        args.seed,
        args.eta,
        args.max_depth,
        args.min_child_weight,
        args.early_stopping
    )

    y_pred = model.predict(dtest)

    y_pred =np.where(y_pred > 0.5,1,0)

    sub['Survived'] = y_pred

    sub.to_csv('submission.csv', index = False)

if __name__ == "__main__":
    submission()