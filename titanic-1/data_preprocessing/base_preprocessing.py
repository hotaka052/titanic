import pandas as pd

"""
基本的なデータを行う関数
    missing_value：欠損値を加工する関数
    drop_data：必要のないカラムを削除する関数
    categorize：カテゴリ変数を数値化する関数
"""

#================================================
# missing_value
# Embarkedは一番多いSで埋める
# Fare, ageは平均で埋める
# Cabinは欠損値が多すぎるので削除
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
# drop_data
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
# categorize
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