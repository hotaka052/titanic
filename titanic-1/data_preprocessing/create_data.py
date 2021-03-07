import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import pathlib

def read_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    df_all = pd.concat([train, test])

    return train, test, df_all

#================================================
#欠損地の処理
#Embarkedは一番多いSで埋める
#Fare, ageは平均で埋める
#Cabinは欠損値が多すぎるので削除
#=================================================
def missing_value(train, test, df_all):

    #Embarked
    train['Embarked'].fillna('S', inplace = True)

    #Fare
    fare_avg = df_all['Fare'].mean()
    test['Fare'].fillna(fare_avg, inplace=True) 

    #age
    age_avg = df_all['Age'].mean()
    train['Age'].fillna(age_avg, inplace = True)
    test['Age'].fillna(age_avg, inplace = True)

    #Cabin
    train.drop('Cabin',axis=1,inplace=True)
    test.drop('Cabin',axis=1,inplace=True)

    return train, test, df_all

#=================================================
#データの選別
#=================================================

def drop_data(train, test, df_all):
    #PassengerId
    #単なる客番号であり、生存の有無には関係ないので削除
    train.drop('PassengerId',axis=1,inplace=True)
    test.drop('PassengerId',axis=1,inplace=True)

    #Name
    #数値化が難しいので削除
    train.drop('Name',axis=1,inplace=True)
    test.drop('Name',axis=1,inplace=True)

    #Ticket
    #数値化が難しいので削除
    train.drop('Ticket',axis=1,inplace=True)
    test.drop('Ticket',axis=1,inplace=True)

    return train, test, df_all

#===================================================
#カテゴリ変数の数値化
#===================================================

def categorize(train, test, df_all):
    #Sex
    sex = df_all['Sex']
    sex_dummies = pd.get_dummies(sex)

    sex_dummies_train = sex_dummies[:891]
    sex_dummies_test = sex_dummies[891:]

    train = pd.concat([train,sex_dummies_train], axis=1)
    test= pd.concat([test,sex_dummies_test],axis=1)

    train.drop('Sex',axis=1,inplace=True)
    test.drop('Sex',axis=1,inplace=True)

    #Embarked
    embarked = df_all['Embarked']
    embarked_dummies = pd.get_dummies(embarked)

    embarked_dummies_train = embarked_dummies[:891]
    embarked_dummies_test = embarked_dummies[891:]

    train = pd.concat([train,embarked_dummies_train],axis=1)
    test = pd.concat([test,embarked_dummies_test],axis=1)

    train.drop('Embarked',axis=1,inplace=True)
    test.drop('Embarked',axis=1,inplace=True)

    return train, test, df_all

def preprocess():
    train, test, df_all = read_data()
    train, test, df_all = missing_value(train, test, df_all)
    train, test, df_all = drop_data(train, test, df_all)
    train, test, df_all = categorize(train, test, df_all)

    return train, test

def split(train):
    X = train.drop("Survived", axis = 1).values
    y = train['Survived'].values

    return train_test_split(X, y, test_size = 0.3, random_state = 71)

#=========================================
#学習用のデータ
#=========================================
def titanic_dataset():
    train, test = preprocess()
    return split(train)

#=========================================
#テスト用データ
#=========================================
def test_dataset():
    train, test = preprocess()
    return test.values