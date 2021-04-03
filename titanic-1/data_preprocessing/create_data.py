import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from .base_preprocessing import *

"""
モデルに合うようにデータを加工する関数
    read_data：データを読み込む関数
    preprocess：データを加工する関数をまとめた関数
    split：学習用データを分割する関数
    titanic_dataset：学習用データを作成する関数
    test_datasset：テスト用データセットを作成する関数
"""

#================================================
# read_data
#================================================
def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/test.csv')

    df_all = pd.concat([train, test])

    return train, test, df_all

#========================================
# preprocess
#========================================
def preprocess(data_folder):
    train, test, df_all = read_data(data_folder)
    train, test, df_all = missing_value(train, test, df_all)
    train, test, df_all = drop_data(train, test, df_all)
    train, test, df_all = categorize(train, test, df_all)

    return train, test

#========================================
# split
#========================================
def split(train):
    X = train.drop("Survived", axis = 1).values
    y = train['Survived'].values

    return train_test_split(X, y, test_size = 0.3, random_state = 71)

#=========================================
# titanic_dataset
#=========================================
def titanic_dataset(data_folder):
    train, _ = preprocess(data_folder)
    return split(train)

#=========================================
# test_dataset
#=========================================
def test_dataset(data_folder):
    _, test = preprocess(data_folder)
    return test.values