import base64
import glob
import os
import uuid
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.schema.document import Document
from langchain_core.stores import BaseStore
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from configs.app_configs import AppConfigs
from libraries.data_processing.ImagesRetrievalFromPdf import summarize_image, ImagesRetrievalFromPdf
from libraries.data_processing.TablesRetrievalFromPdf import TablesRetrievalFromPdf
from libraries.data_processing.TextRetrievalFromPdf import TextRetrievalFromPdf

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

_CHROMA_PERSIST_DIRECTORY = os.path.join(*app_configs.configs.MultiChromaConfigs.path)
_LOCAL_FILE_STORE_PATH = os.path.join(*app_configs.configs.LocalFileStoreConfigs.path)
_FIGURES_STORAGE_PATH = os.path.join(*app_configs.configs.Documents.Images.path)
_IMAGE_SUMMARIES_PATH = os.path.join(*app_configs.configs.Documents.ImageSummaries.path, app_configs.configs.Documents.ImageSummaries.file_name)
_COLLECTION_NAME = app_configs.configs.MultiChromaConfigs.default_collection_name
_ID_KEY = app_configs.configs.MultiChromaConfigs.default_docstore_key


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def decode_image(base64_image: str) -> bytes:
    return base64.b64decode(base64_image)


def save_summaries(base64_images: List[str], summaries: List[str]) -> bool:
    import json

    try:
        json_doc = [{
            'image_base64': i,
            'summary': s,
        } for i, s in list(zip(base64_images, summaries))]

        with open(os.path.join(_IMAGE_SUMMARIES_PATH), 'w') as fp:
            json.dump(json_doc, fp)
            return True

    except Exception:
        return False


def load_summaries() -> (List[str], List[str]):
    import json
    with open(_IMAGE_SUMMARIES_PATH) as json_file:
        image_summaries = json.load(json_file)
    summaries, base64_images = [], []
    for summary in image_summaries:
        base64_images.append(summary['image_base64'])
        summaries.append(summary['summary'])
    return base64_images, summaries


class MultiModalRetrieval:

    id_key: str = _ID_KEY
    collection_name: str = _COLLECTION_NAME
    chroma_persist_directory: str = _CHROMA_PERSIST_DIRECTORY
    embeddings: Embeddings = OpenCLIPEmbeddings(model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k")
    docstore: BaseStore = LocalFileStore(root_path=_LOCAL_FILE_STORE_PATH)  # persistent ByteStore
    # docstore: BaseStore = InMemoryStore()  # non-persistent ByteStore
    source_filename: str
    vectorstore: Chroma
    retriever: MultiVectorRetriever

    def __init__(self, from_persistent: bool = True, source_filename: str = "Manuale operativo iliad FTTH (1).pdf"):
        self.source_filename = source_filename

        self.vectorstore = Chroma(
            persist_directory=self.chroma_persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        if not from_persistent:
            self.__add_documents()

    def __add_documents(self):
        self.__add_texts()
        self.__add_tables()
        self.__add_images()

    def __add_texts(self):
        with TextRetrievalFromPdf(self.source_filename) as text_ret:
            texts = text_ret.get_text_elements()

        doc_ids = [str(uuid.uuid4()) for _ in texts]
        docs_from_texts = [
            Document(page_content=text.content, metadata=text.metadata | {self.id_key: doc_ids[i]})
            for i, text in enumerate(texts)
        ]
        encoded_text_contents = [text.content.encode() for text in texts]
        self.retriever.vectorstore.add_documents(docs_from_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, encoded_text_contents)))

    def __add_tables(self):
        with TablesRetrievalFromPdf(self.source_filename) as table_ret:
            tables = table_ret.get_table_elements(max_characters=1000)

        table_ids = [str(uuid.uuid4()) for _ in tables]
        docs_from_tables = [
            Document(page_content=table.content, metadata=table.metadata | {self.id_key: table_ids[i]})
            for i, table in enumerate(tables)
        ]
        encoded_table_contents = [table.content.encode() for table in tables]
        self.retriever.vectorstore.add_documents(docs_from_tables)
        self.retriever.docstore.mset(list(zip(table_ids, encoded_table_contents)))

    def __add_images(self, debug: bool = True):
        if debug:
            img_base64_list, image_summaries = load_summaries()
            img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
            summary_img_documents = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]})
                for i, summary in enumerate(image_summaries)
            ]
            encoded_img_base64_list = [img.encode() for img in img_base64_list]
            self.retriever.vectorstore.add_documents(summary_img_documents)
            self.retriever.docstore.mset(list(zip(img_ids, encoded_img_base64_list)))
        else:
            with ImagesRetrievalFromPdf(self.source_filename) as image_ret:
                images = image_ret.get_image_elements()

            image_ids = [str(uuid.uuid4()) for _ in images]
            docs_from_images = [
                Document(page_content=image.content, metadata=image.metadata | {self.id_key: image_ids[i]})
                for i, image in enumerate(images)
            ]
            encoded_base64_images = [elem.base64_image.encode() for elem in images]
            self.retriever.vectorstore.add_documents(docs_from_images)
            self.retriever.docstore.mset(list(zip(image_ids, encoded_base64_images)))

    def get_relevant_documents(self, text_query):
        return self.retriever.get_relevant_documents(text_query)

    def get_retriever(self):
        return self.retriever
