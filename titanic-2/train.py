import numpy as np
import xgboost as xgb
from data_preprocessing import titanic_dataset
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

"""
train：モデルを学習させる関数
"""

def train(seed, eta, max_depth, min_child_weight, early_stopping, data_folder):
    x_train, x_val, y_train, y_val = titanic_dataset(data_folder)

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
