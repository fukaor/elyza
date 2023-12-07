# Pythonの公式イメージをベースにする
FROM python:3.8

# 必要なパッケージをインストールする
RUN pip install transformers langchain accelerate bitsandbytes pypdf tiktoken sentence_transformers faiss-gpu trafilatura --quiet

# LLMのコードをコンテナにコピーする
COPY . /app

# コンテナ内でコマンドを実行するディレクトリを設定する
WORKDIR /app

# コンテナが起動したときに実行するコマンドを設定する
CMD ["python", "elyza_script.py"]
