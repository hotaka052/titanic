import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import pathlib

def read_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    df_all = pd.concat([train, test])

    return df_all

#================================================
#新しい特徴量の追加
#title
#================================================
def name_title(df_all):
    #Nameをfamily name, title, last nameに分割
    df_fname = df_all['Name'].str.split(',', expand=True)
    df_lname = df_fname[1].str.split('.', expand=True)
    df_fname.drop(1, axis = 1, inplace = True)
    df_lname.drop(2, axis= 1, inplace = True)
    df_fname = df_fname.rename(columns = {0:'first-name'})
    df_lname = df_lname.rename(columns = {0:'title',1:'last-name'})
    
    #分割したものを合体
    df_name = pd.concat([df_fname,df_lname],axis = 1)

    #df_allと合流
    df_all = pd.concat([df_all,df_name],axis = 1)

    #必要のない特徴量を削除
    df_all.drop('Name', axis = 1, inplace = True)
    df_all.drop('first-name', axis = 1, inplace = True)
    df_all.drop('last-name', axis = 1, inplace = True)

    return df_all

#================================================
#新しい特徴量の追加
#family size
#================================================
def fam_size(df_all):
    df_all['famsize'] = df_all['SibSp'] + df_all['Parch']

    #famsizeに統合した特徴量を削除
    df_all.drop('SibSp',axis = 1,inplace = True)
    df_all.drop('Parch',axis = 1,inplace = True)

    return df_all

#================================================
#新しい特徴量の追加
#alone
#１人で乗っている場合は０それ以外は１
#================================================
def alone(df_all):
    df_all['alone'] = 0
    df_all.loc[df_all['famsize'] >= 1,'alone'] = 1

    return df_all

#================================================
#欠損地の処理
#Nameから抽出したtitle（敬称）を参照して埋めていく
#================================================
def fill_age(df_all):
    #年齢の欠損値に関しては敬称を参照して埋めていく
    age_ave = df_all['Age'].mean()
    Dr_age = df_all.groupby('title')['Age'].mean()[' Dr']
    Master_age = df_all.groupby('title')['Age'].mean()[' Master']
    Miss_age =df_all.groupby('title')['Age'].mean()[' Miss']
    Mr_age = df_all.groupby('title')['Age'].mean()[' Mr']
    Mrs_age = df_all.groupby('title')['Age'].mean()[' Mrs']

    #Dr
    Dr_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Dr')])
    for i in range(len(Dr_id)):
        df_all.loc[df_all['PassengerId'] == Dr_id[i],'Age'] = Dr_age

    #Master
    Master_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Master')])
    for i in range(len(Master_id)):
        df_all.loc[df_all['PassengerId'] == Master_id[i],'Age'] = Master_age

    #Miss
    Miss_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Miss')])
    for i in range(len(Miss_id)):
        df_all.loc[df_all['PassengerId'] == Miss_id[i],'Age'] = Miss_age

    #Mr
    Mr_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Mr')])
    for i in range(len(Mr_id)):
        df_all.loc[df_all['PassengerId'] == Mr_id[i],'Age'] = Mr_age

    #Mrs
    Mrs_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Mrs')])
    for i in range(len(Mrs_id)):
        df_all.loc[df_all['PassengerId'] == Mrs_id[i],'Age'] = Mrs_age

    Ms_id = list(df_all['PassengerId'][(df_all['Age'].isnull() == True) & (df_all['title'] == ' Ms')])
    for i in range(len(Ms_id)):
        df_all.loc[df_all['PassengerId'] == Ms_id[i],'Age'] = age_ave

    #年齢を埋めるのに使ったので削除
    df_all.drop('title', axis = 1, inplace = True)

    return df_all

#================================================
#欠損地の処理（age以外）
#Embarkedは一番多いSで埋める
#Fareは平均で埋める
#Cabinは欠損値が多すぎるので削除
#=================================================
def missing_value(df_all):

    #Embarked
    df_all['Embarked'].fillna('S', inplace = True)

    #Fare
    fare_avg = df_all['Fare'].mean()
    df_all['Fare'].fillna(fare_avg, inplace=True) 

    #Cabin
    df_all.drop('Cabin', axis=1, inplace=True)

    return df_all

#=================================================
#データの選別
#=================================================

def drop_data(df_all):
    #PassengerId
    #単なる客番号であり、生存の有無には関係ないので削除
    df_all.drop('PassengerId', axis=1, inplace=True)

    #Ticket
    #数値化が難しいので削除
    df_all.drop('Ticket',axis=1,inplace=True)

    return df_all

#===================================================
#カテゴリ変数の数値化
#===================================================

def categorize(df_all):
    #Sex
    sex = df_all['Sex']
    sex_dummies = pd.get_dummies(sex)

    df_all = pd.concat([df_all, sex_dummies], axis = 1)

    df_all.drop('Sex', axis =1, inplace = True)

    #Embarked
    embarked = df_all['Embarked']
    embarked_dummies = pd.get_dummies(embarked)

    df_all = pd.concat([df_all,embarked_dummies],axis = 1)

    df_all.drop('Embarked',axis = 1,inplace = True)

    return df_all

def preprocess():
    df_all = read_data()
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