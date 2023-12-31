#==================================
# ELYZAによるRAG検証スクリプト
#
# 今回はワンピースのWikiを読み込ませ、その内容から回答を行う。
# 参考：https://note.com/alexweberk/n/n3cffc010e9e9
#==================================

#==================================
# ライブラリ
#==================================
# 必要ライブラリ
from flask import Flask, request, Response
import json
import pickle
# 関数一覧
from functions import _PREPARE_DATA, _SETUP_RAG, _SETUP_MODEL, _SETUP_PROMPT, _SETUP_QA

#==================================
# API
#==================================
app = Flask(__name__)

# SSE処理
def event_stream(qa, context, question):
    try:
        response = qa.run(context=context, question=question)
        yield 'data: {}\n\n'.format(json.dumps(response))
    except Exception as e:
        print(f"Error occurred: {e}")
        yield 'data: {"error": "An error occurred while processing the request."}\n\n'

# ask
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    context = ""  # ここに適切なコンテキストを設定します

    return Response(event_stream(qa, context, question), mimetype="text/event-stream")

if __name__ == "__main__":
    # 初期設定
    url = "https://ja.m.wikipedia.org/wiki/ONE_PIECE"
    model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    filename = 'textfile.txt'

    # データのベクトル化等の準備
    filename = _PREPARE_DATA(url, filename)
    retriever = _SETUP_RAG(filename)

    # retrieverをロードする
    with open('retriever.pkl', 'rb') as f:
        pickle.dump(retriever, f)

    # モデルとトークナイザを設定する
    model, tokenizer = _SETUP_MODEL(model_id)

    # プロンプトを設定する
    pipe, PROMPT = _SETUP_PROMPT(tokenizer, model)

    # QAシステムを設定する
    qa = _SETUP_QA(retriever, model, tokenizer, pipe, PROMPT)

    # Flaskサーバーを起動
    app.run(host='0.0.0.0', port=5000)
