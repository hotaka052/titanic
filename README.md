# 概要
kaggle competitionの代表的なコンペであるtitanicコンペのノートです  
参加期間：2020年4月中旬～5月中旬  
</br>  

# フォルダについて
titanic-1：scikit-learn インターフェースのXGBoostを使用したコード  
titanic-2：ネイティブなXGBoostを使用したコード  
</br>

# 必要なライブラリ
・python 3系  
・numpy  
・pandas  
・scikit-learn  
・xgboost  
</br>

# 使いかた
１．kaggleにファイルをデータセットとしてアップロード

２．ノートにtitanicのデータとアップロードしたデータセットを追加

３．作業ディレクトリ(titanic-1 or titanic-2)に移動  
```py  
import os  
path = "../input/データセット名/tianic/作業ディレクトリ"
os.chdir(path)" 
```

４．ファイルの実行  
```py
!python 実行したいファイル.py
```