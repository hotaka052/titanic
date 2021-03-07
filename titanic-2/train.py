import numpy as np
import xgboost as xgb
from data_preprocessing import titanic_dataset
from sklearn.metrics import accuracy_score

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

def train(seed, eta, max_depth, min_child_weight, early_stopping):
    x_train, x_val, y_train, y_val = titanic_dataset()

    dtrain = xgb.DMatrix(x_train, label = y_train)
    dvalid = xgb.DMatrix(x_val, label = y_val)

    params={
        'booster':'gbtree',
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'eta':eta,
        'max_depth':max_depth,
        'min_child_weight':min_child_weight,
        'random_state':seed
    }

    evals=[(dtrain,'train'),(dvalid,'valid')]
    evals_result={}

    model=xgb.train(
        params,
        dtrain,
        num_boost_round = 1000,
        early_stopping_rounds = early_stopping,
        evals = evals,
        evals_result = evals_result
    )

    #================================================
    #モデルの評価
    #================================================
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dvalid)

    train_pred = np.where(train_pred > 0.5, 1,0)
    val_pred = np.where(val_pred > 0.5, 1,0)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print()
    print("==================モデルの評価==================")
    print('Train score:', train_acc)
    print('Valid score:', val_acc)

    return model
