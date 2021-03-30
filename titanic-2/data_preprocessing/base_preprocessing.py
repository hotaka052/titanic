import pandas as pd

"""
データに基本的な加工を施す
    missing_value：欠損値を加工する関数
    drop_data：必要のないカラムを削除する関数
    categorize：カテゴリ変数を数値化する関数
"""

#================================================
# missing_value
# Embarkedは一番多いSで埋める
# Fareは平均で埋める
# Cabinは欠損値が多すぎるので削除
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
# drop_data
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
# categorize
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