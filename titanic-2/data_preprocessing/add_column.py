import pandas as pd

"""
新しいカラムを追加する関数
    name_title：titleを追加する関数
    fam_size：family_size(famsize)を追加する関数
    alone：１人で乗っている場合は０それ以外は１というaloneを追加する関数
    fill_age：Nameから抽出したtitle（敬称）を参照して埋めていく関数
"""
#================================================
# name_title
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
# fam_size
#================================================
def fam_size(df_all):
    df_all['famsize'] = df_all['SibSp'] + df_all['Parch']

    #famsizeに統合した特徴量を削除
    df_all.drop('SibSp',axis = 1,inplace = True)
    df_all.drop('Parch',axis = 1,inplace = True)

    return df_all

#================================================
# alone
#================================================
def alone(df_all):
    df_all['alone'] = 0
    df_all.loc[df_all['famsize'] >= 1,'alone'] = 1

    return df_all

#================================================
# fill_age
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
