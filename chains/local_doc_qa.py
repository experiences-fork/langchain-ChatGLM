from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from chains.modules.analyticdb import AnalyticDB
from models.chatglm_llm import ChatGLM
from vectorstores import MyFAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from configs.model_config import *
from textsplitter import ChineseTextSplitter
from typing import List
from utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from models.base import (BaseAnswer,
                         AnswerResult)
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import models.shared as shared
from agent import bing_search
from langchain.docstore.document import Document
from functools import lru_cache
from textsplitter.zh_title_enhance import zh_title_enhance

CONNECTION_STRING = AnalyticDB.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD", "postgres"),
)

# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash


# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return MyFAISS.load_local(vs_path, embeddings)


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    if len(related_docs) > 0:
        context = "\n".join([doc.page_content for doc in related_docs])
    else:
        context = "\n".join("无")
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

class LocalDocQA:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: BaseAnswer = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="加载文件"):
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            print("文件加载完毕，正在生成向量库")
            AnalyticDB.from_documents(documents=docs, embedding=self.embeddings,
                                      connection_string=CONNECTION_STRING, collection_name=vs_path)
            torch_gc()
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def get_knowledge_based_answer(self,
                                   query,
                                   vs_path,
                                   chat_history=[],
                                   streaming: bool = STREAMING):
        print(vs_path)
        vector_store = AnalyticDB.from_documents(documents=[], embedding=self.embeddings,
                                                 connection_string=CONNECTION_STRING, collection_name=vs_path)
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        torch_gc()
        prompt = generate_prompt(related_docs, query)
        print(prompt)

        # if streaming:
        #     for result, history in self.llm._stream_call(prompt=prompt,
        #                                                  history=chat_history):
        #         history[-1][0] = query
        #         response = {"query": query,
        #                     "result": result,
        #                     "source_documents": related_docs}
        #         yield response, history
        # else:
        for result, history in self.llm._call(prompt=prompt,
                                              history=chat_history,
                                              streaming=streaming):
            torch_gc()
            history[-1][0] = query
            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs}
            yield response, history
            torch_gc()

    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history

    # query      查询内容
    # vs_path    知识库路径
    # chunk_conent   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度
    def get_knowledge_based_conent_test(self, query, vs_path, chunk_conent,
                                        score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
                                        vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_size=CHUNK_SIZE):
        vector_store = load_vector_store(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_conent = chunk_conent
        vector_store.score_threshold = score_threshold
        vector_store.chunk_size = chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k)
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])
        response = {"query": query,
                    "source_documents": related_docs_with_score}
        return response, prompt

    def get_search_result_based_answer(self, query, chat_history=[], streaming: bool = STREAMING):
        results = bing_search(query)
        result_docs = search_result2docs(results)
        prompt = generate_prompt(result_docs, query)

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": result_docs}
            yield response, history

    def delete_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      vs_path):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.delete_doc(filepath)
        return status

    def update_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      vs_path,
                                      docs: List[Document],):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.update_doc(filepath, docs)
        return status

    def list_file_from_vector_store(self,
                                    vs_path,
                                    fullpath=False):
        vector_store = load_vector_store(vs_path, self.embeddings)
        docs = vector_store.list_docs()
        if fullpath:
            return docs
        else:
            return [os.path.split(doc)[-1] for doc in docs]

    def get_collections(self):
        return AnalyticDB(connection_string=CONNECTION_STRING, embedding_function=self.embeddings).get_collections()


if __name__ == "__main__":
    # 初始化消息
    args = None
    args = parser.parse_args(args=['--model-dir', '/media/checkpoint/', '--model', 'chatglm-6b', '--no-remote-model'])

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins)
    query = "本项目使用的embedding模型是什么，消耗多少显存"
    vs_path = "/media/gpt4-pdf-chatbot-langchain/dev-langchain-ChatGLM/vector_store/test"
    last_print_len = 0
    # for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
    #                                                              vs_path=vs_path,
    #                                                              chat_history=[],
    #                                                              streaming=True):
    for resp, history in local_doc_qa.get_search_result_based_answer(query=query,
                                                                     chat_history=[],
                                                                     streaming=True):
        print(resp["result"][last_print_len:], end="", flush=True)
        last_print_len = len(resp["result"])
    source_text = [f"""出处 [{inum + 1}] {doc.metadata['source'] if doc.metadata['source'].startswith("http")
    else os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                   # f"""相关度：{doc.metadata['score']}\n\n"""
                   for inum, doc in
                   enumerate(resp["source_documents"])]
    logger.info("\n\n" + "\n\n".join(source_text))
    pass
