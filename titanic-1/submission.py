import numpy as np
import pandas as pd
from data_preprocessing import titanic_dataset, test_dataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(
    description = "titanic competition of kaggle"
)

parser.add_argument("--seed", default = 71, type = int,
                    help = "乱数を固定するシード値")
parser.add_argument("--lr", default = 0.05, type = float,
                    help = "モデルの学習率")
parser.add_argument("--n_estimators", default = 100, type = int,
                    help = "作成される決定木の数")
parser.add_argument("--max_depth", default = 3, type = int,
                    help = "探索したいmax_depthのリスト")
parser.add_argument("--min_child_weight", default = 4, type = int,
                    help = "探索したいmin_child_weightのリスト")
parser.add_argument("--data_folder", default = "/kaggle/input/titanic", type = str,
                    help = "データのあるフォルダーへのパス")
parser.add_argument("--output_folder", default = "/kaggle/working", type = str,
                    help = "提出用ファイルを出力したいフォルダー")

args = parser.parse_args()

def train():
    x_train, x_val, y_train, y_val = titanic_dataset()

    model = XGBClassifier(
        booster = 'gbtree', 
        random_state = args.seed, 
        learning_rate = args.lr,
        objective = 'binary:logistic',
        n_estimators = args.n_estimators,
        max_depth = args.max_depth,
        min_child_weight = args.min_child_weight
    )

    model.fit(x_train, y_train)

    #モデルの評価
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print('max_depth：', args.max_depth, 'min_child_weight：', args.min_child_weight)
    print('Train Score：', train_acc, 'Val Score：', val_acc)

    return model

def submission():
    sub = pd.read_csv(args.data_folder + '/gender_submission.csv')
    x_test = test_dataset(args.data_folder)
    model = train()

    y_pred = model.predict(x_test)
    sub['Survived'] = y_pred
    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    submission()