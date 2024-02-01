import os
import re
from typing import Any, Dict, List

from fitz import fitz, Document, Page
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from configs.app_configs import AppConfigs

app_configs = AppConfigs()
documents_dir_path = os.path.join(*app_configs.configs.Documents.path)


def get_document(path: str):
    return fitz.Document(path)


class TextElement:
    content: str
    metadata: Dict[str, Any]

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


class TextRetrievalFromPdf:
    source_filename: str
    source_dir: str
    document: Document
    pages: List[Page]

    def __init__(self, source_filename: str, source_dir: str = documents_dir_path):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = get_document(path=os.path.join(source_dir, source_filename))
        self.pages = [*self.document.pages()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    def get_text_elements(self) -> List[TextElement]:
        text_elements = []
        elements = partition_pdf(
            filename=os.path.join(self.source_dir, self.source_filename),
            # strategy='hi_res',
            chunking_strategy="by_title",

        )

        chunks = chunk_by_title(
            elements=elements,
            combine_text_under_n_chars=2000,
            max_characters=6000,
            new_after_n_chars=5500,
            overlap=500,)

        for chunk in chunks:
            text_elements.append(TextElement(
                content=chunk.text,
                metadata={'source': chunk.metadata.fields['filename'],
                          'page_number': chunk.metadata.fields['page_number']}
            ))

        return text_elements
