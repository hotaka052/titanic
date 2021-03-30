import numpy as np
from data_preprocessing import titanic_dataset
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
parser.add_argument("--max_depth_list", default = [2, 3, 4, 5, 6], nargs = '+', type = int,
                    help = "探索したいmax_depthのリスト")
parser.add_argument("--min_child_weight_list", default = [1, 2, 3, 4, 5, 6, 7], nargs = '+', type = int,
                    help = "探索したいmin_child_weightのリスト")
parser.add_argument('--data_folder', default = '/kaggle/input/titanic', type = str,
                    help = "データのあるフォルダへのパス")

args = parser.parse_args()

def train_grid():
    x_train, x_val, y_train, y_val = titanic_dataset(args.data_folder)

    np.random.seed(args.seed)

    for max_depth in args.max_depth_list:
        for min_child_weight in args.min_child_weight_list:

            model = XGBClassifier(
                booster = 'gbtree', 
                random_state = args.seed, 
                learning_rate = args.lr,
                objective = 'binary:logistic',
                n_estimators = args.n_estimators,
                max_depth = max_depth,
                min_child_weight = min_child_weight
            )

            model.fit(x_train, y_train)

            #モデルの評価
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)

            print('max_depth：', max_depth, 'min_child_weight：', min_child_weight)
            print('Train Score：', train_acc, 'Val Score：', val_acc)

if __name__ == '__main__':
    train_grid()