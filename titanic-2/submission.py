import numpy as np
import pandas as pd
import xgboost as xgb
from train import train
from data_preprocessing import test_dataset

import warnings
warnings.filterwarnings('ignore')

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
parser.add_argument("--data_folder", default = "/kaggle/input/titanic", type = str,
                    help = "データのあるフォルダーへのパス")
parser.add_argument("--output_folder", default = "/kaggle/working", type = str,
                    help = "提出用ファイルを出力したいフォルダー")

args = parser.parse_args()

def submission():
    # モデルの学習
    model = train(
        args.seed,
        args.eta,
        args.max_depth,
        args.min_child_weight,
        args.early_stopping,
        args.data_folder
    )

    x_test = test_dataset(args.data_folder)
    dtest = xgb.DMatrix(x_test)

    y_pred = model.predict(dtest)

    y_pred =np.where(y_pred > 0.5,1,0)

    sub = pd.read_csv(args.data_folder + '/gender_submission.csv')

    sub['Survived'] = y_pred

    sub.to_csv(args.output_folder + 'submission.csv', index = False)

if __name__ == "__main__":
    submission()