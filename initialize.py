#==================================
# ライブラリ
#==================================
# 必要ライブラリ
import pickle
# 関数一覧
from functions import _PREPARE_DATA, _SETUP_RAG


#==================================
# 初期設定
#==================================
# 追加参照データ
url = "https://ja.m.wikipedia.org/wiki/ONE_PIECE"
filename = 'textfile.txt'

# データのベクトル化等の準備
filename = _PREPARE_DATA(url, filename)
retriever = _SETUP_RAG(filename)

# 初期化済みのretrieverを保存する
with open('retriever.pkl', 'wb') as f:
    pickle.dump(retriever, f)
