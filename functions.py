#==================================
# ライブラリ
#==================================
# 必要ライブラリ
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate
# ウェブページのテキストを抽出してくれるライブラリ
from trafilatura import fetch_url, extract


#==================================
# 関数一覧
#==================================
# 指定されたURLからデータを取得し、テキストファイルに保存する
def _PREPARE_DATA(url, filename):
    document = fetch_url(url)
    text = extract(document)
    print(text[:1000])
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    return filename

# ファイルからドキュメントを読み込み、それらを分割してベクトルデータベースを作成する
def _SETUP_RAG(filename):
    loader = TextLoader(filename, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size=300,
        chunk_overlap=20,
    )
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# 指定されたモデルIDを使用して、トークナイザとモデルを設定する
def _SETUP_MODEL(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    ).eval()
    return model, tokenizer

# トークナイザを使用してプロンプトテンプレートを設定する
def _SETUP_PROMPT(tokenizer, model):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。"
    text = "{context}\nユーザからの質問は次のとおりです。{question}"
    template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        template_format="f-string"
    )
    return pipe, PROMPT

# リトリーバ、モデル、トークナイザ、パイプライン、プロンプトを使用して、質疑応答(QA)システムを設定する
def _SETUP_QA(retriever, model, tokenizer, pipe, PROMPT):
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(
            pipeline=pipe,
        ),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    return qa
