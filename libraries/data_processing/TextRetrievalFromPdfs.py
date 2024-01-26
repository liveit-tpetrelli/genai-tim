import os
import re
from typing import Any, Dict, List

from fitz import fitz, Document, Page
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title

from configs.app_configs import AppConfigs

app_configs = AppConfigs()
documents_dir_path = os.path.join(*app_configs.configs.Documents.path)


class TextElement:
    content: str
    metadata: Dict[str, Any]

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


class TextRetrievalFromPdfs:
    source_filename: str
    source_dir: str
    document: Document
    pages: List[Page]

    def __init__(self, source_filename: str, source_dir: str = documents_dir_path):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = self.__get_document(path=os.path.join(source_dir, source_filename))
        self.pages = [*self.document.pages()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    @staticmethod
    def __get_document(path: str):
        return fitz.Document(path)

    def get_text_elements(self):
        all_text = ""
        for page in self.pages:
            text = page.get_text()
            all_text += text

        chunks = chunk_elements(all_text)
        return chunk_by_title(all_text)