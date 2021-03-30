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
※githubから直接使う場合  
１．kaggleを開いてノートにtitanicのデータを追加

２．githubからコードをノートに追加  
!git clone https://github.com/haru-mingshi052/titanic.git   

３．作業ディレクトリに移動  
import os  
path = "作業ディレクトリ" &nbsp; 例）path = './titanic/titanic-1'  
os.chdir(path)

４．ファイルの実行  
!python 実行したいファイル.py &nbsp; 例）!python submission.py

※一度手元に置いてから使う場合  
１．kaggleにファイルをデータとしてアップロード

２．ノートにtitanicのデータとアップロードしたファイルを追加

３．作業ディレクトリに移動  
import os  
path = "../input/データ名/tianic/作業ディレクトリ" &nbsp; 例）"../input/データ名/titanic/titanic-1"  
os.chdir(path)

４．ファイルの実行  
!python 実行したいファイル.py &nbsp; 例）!python submission.py