import base64
import glob
import os
import uuid
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.storage import LocalFileStore
from langchain.schema.document import Document
from langchain_core.stores import BaseStore
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from configs.app_configs import AppConfigs
from libraries.data_processing.ImagesRetrievalFromPdf import summarize_image
from libraries.data_processing.TablesRetrievalFromPdf import TablesRetrievalFromPdf
from libraries.data_processing.TextRetrievalFromPdf import TextRetrievalFromPdf

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)
figures_storage_path = os.path.join(*app_configs.configs.Documents.Images.path)
image_summaries_path = os.path.join(*app_configs.configs.Documents.ImageSummaries.path, app_configs.configs.Documents.ImageSummaries.file_name)


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

        with open(os.path.join(image_summaries_path), 'w') as fp:
            json.dump(json_doc, fp)
            return True

    except Exception:
        return False


def load_summaries() -> (List[str], List[str]):
    import json
    with open(image_summaries_path) as json_file:
        image_summaries = json.load(json_file)
    summaries, base64_images = [], []
    for summary in image_summaries:
        base64_images.append(summary['image_base64'])
        summaries.append(summary['summary'])
    return base64_images, summaries

class MultiModalVectorstore:
    chroma_persist_directory: str
    embeddings: Embeddings
    vectorstore: Chroma
    docstore: BaseStore
    retriever: MultiVectorRetriever
    img_base64_list: List[str]
    image_summaries: List[str]

    def __init__(self, from_summaries: bool = True):
        self.chroma_persist_directory = os.path.join(*app_configs.configs.MultiChromaConfigs.path)
        self.embeddings = OpenCLIPEmbeddings(
            model_name="ViT-H-14",
            checkpoint="laion2b_s32b_b79k"
        )

        '''Google Multi-modal embeddings API
        # self.embeddings = os.system("python3 predict_request_gapic.py --image_file 'IMAGE_FILE' --text 'TEXT' --project 'PROJECT_ID'")
        # self.embeddings = VertexAIEmbeddings(model_name="multimodalembedding")
        '''

        self.vectorstore = Chroma(
            persist_directory=self.chroma_persist_directory,
            collection_name="multi_modal_rag",
            embedding_function=self.embeddings
        )

        # The storage layer for the parent documents
        self.docstore = InMemoryStore()
        # local_filestore_path = os.path.join("db", "local_file_store")
        # self.docstore = LocalFileStore(root_path=local_filestore_path)
        id_key = "doc_id"

        # The retriever (empty to start)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=id_key,
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        if from_summaries:
            self.img_base64_list, self.image_summaries = load_summaries()
        else:
            for image_path in glob.glob(os.path.join(figures_storage_path, "*")):
                # List of base64 strings of the images retrieved from the PDF
                self.img_base64_list.append(encode_image(image_path))
                # List of summaries of the images generated with gemini-pro-vision
                self.image_summaries.append(summarize_image(image_path))

            _ = save_summaries(self.img_base64_list, self.image_summaries)

        # Add image summaries
        img_ids = [str(uuid.uuid4()) for _ in self.img_base64_list]

        summary_img_documents = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(self.image_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_img_documents)
        # decoded_images = [decode_image(img) for img in self.img_base64_list]
        self.retriever.docstore.mset(list(zip(img_ids, self.img_base64_list)))

        # Add text
        with TextRetrievalFromPdf(source_filename="Manuale operativo iliad FTTH (1).pdf") as text_ret:
            texts = text_ret.get_text_elements()
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            docs_from_texts = [
                Document(page_content=text.content, metadata=text.metadata | {id_key: doc_ids[i]})
                for i, text in enumerate(texts)
            ]
        text_contents = [text.content for text in texts]
        self.retriever.vectorstore.add_documents(docs_from_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, text_contents)))

        # Add tables
        with TablesRetrievalFromPdf(source_filename="Manuale operativo iliad FTTH (1).pdf") as table_ret:
            tables = table_ret.get_table_elements(max_characters=1000)
            table_ids = [str(uuid.uuid4()) for _ in tables]
            # texts = text_ret.get_text_elements_by_titles()
            docs_from_tables = [
                Document(page_content=table.content, metadata=table.metadata | {id_key: table_ids[i]})
                for i, table in enumerate(tables)
            ]
        table_contents = [table.content for table in tables]
        self.retriever.vectorstore.add_documents(docs_from_tables)
        self.retriever.docstore.mset(list(zip(table_ids, table_contents)))

    def get_relevant_documents(self, text_query):
        return self.retriever.get_relevant_documents(text_query)

    def get_retriever(self):
        return self.retriever
