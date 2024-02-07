import os
from typing import List

from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document

from libraries.data_processing.ImagesRetrievalFromPdf import ImagesRetrievalFromPdf
from libraries.data_processing.TablesRetrievalFromPdf import TablesRetrievalFromPdf, TableElement
from libraries.data_processing.TextRetrievalFromPdf import TextRetrievalFromPdf, TextElement


def get_docs_from_text(texts: List[TextElement]) -> List[Document]:
    docs = []
    for text in texts:
        docs.append(Document(
            page_content=text.content,
            metadata=text.metadata
        ))
    return docs


def get_docs_from_tables(tables: List[TableElement]) -> List[Document]:
    docs = []
    for table in tables:
        docs.append(Document(
            page_content=table.content,
            metadata=table.metadata
        ))
    return docs

# def get_docs_from_images(self):


class TokenElementSplitter:
    source_filename: str

    def __init__(self, source_filename: str):
        self.source_filename = source_filename

    def split_document(self,
                       chunk_size: int = 2000,
                       chunk_overlap: int = 0,
                       include_tables: bool = True,
                       include_images: bool = True):
        docs = []

        with TextRetrievalFromPdf(self.source_filename) as text_ret:
            texts = text_ret.get_text_elements()
            # texts = text_ret.get_text_elements_by_titles()
            docs_from_texts = get_docs_from_text(texts)
            docs.extend(docs_from_texts)

        if include_tables:
            with TablesRetrievalFromPdf(self.source_filename) as table_ret:
                tables = table_ret.get_table_elements(max_characters=1000)
                docs_from_tables = get_docs_from_tables(tables)
                docs.extend(docs_from_tables)

        # TODO - To retrieve images
        if include_images:
            with ImagesRetrievalFromPdf(self.source_filename) as ret:
                images = ret.get_image_elements()
                print(images)
                # docs_from_images = get_docs_from_images(images)
                # docs.extend(docs_from_images)

        token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        token_splits = token_splitter.split_documents(docs)
        return token_splits


