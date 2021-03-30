import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing.base_preprocessing import *
from data_preprocessing.add_column import *

"""
データを学習できるように作り変える
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

    return df_all

#================================================
# preprocess
#================================================
def preprocess(data_folder):
    df_all = read_data(data_folder)
    df_all = name_title(df_all)
    df_all = fam_size(df_all)
    df_all = alone(df_all)
    df_all = fill_age(df_all)
    df_all = missing_value(df_all)
    df_all = drop_data(df_all)
    df_all = categorize(df_all)

    train = df_all[:891]
    test = df_all[891:]

    test.drop('Survived', axis = 1, inplace = True)

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
    train, test = preprocess(data_folder)
    return split(train)

#=========================================
# test_dataset
#=========================================
def test_dataset(data_folder):
    train, test = preprocess(data_folder)
    return test.values