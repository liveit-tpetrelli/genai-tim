import os
import re
from typing import Any, Dict, List

import pdfplumber
from fitz import fitz, Document, Page
from pdfplumber import PDF
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from configs.app_configs import AppConfigs

app_configs = AppConfigs()
documents_dir_path = os.path.join(*app_configs.configs.Documents.path)


def get_document(path: str):
    return pdfplumber.open(path)


class TableElement:
    content: str
    metadata: Dict[str, Any]

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


class TablesRetrievalFromPdf:
    source_filename: str
    source_dir: str
    document: PDF
    pages: List[Page]

    def __init__(self, source_filename: str, source_dir: str = documents_dir_path):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = get_document(path=os.path.join(source_dir, source_filename))
        # self.pages = [*self.document.pages()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    def get_table_elements(self, max_characters: int) -> List[TableElement]:
        tables = []
        for page in self.document.pages:
            tabs = page.extract_tables()
            if tabs:
                for t in tabs:
                    for i in range(1, len(t)):
                        if None in t[i]:
                            cleaned_row = [(i, text) for i, text in enumerate(t[i]) if text]
                            t[i] = ['']
                            for k, item in enumerate(cleaned_row):
                                t[i-1][item[0]] += f"\n{item[1]}"

                    cleaned_table = [elem for elem in t if elem != ['']]
                    if len(f'Table: {cleaned_table}') > max_characters:
                        diff_len = len(f'Table: {cleaned_table}')
                        header, i = [cleaned_table[0]], 1
                        while diff_len > max_characters:
                            temp = header
                            while len(str(temp)) <= max_characters:
                                temp.append(cleaned_table[i])
                                i += 1
                            tables.append(
                                TableElement(
                                    content=f'SplittedTable: {temp}',
                                    metadata={'source': self.document.path.name, 'page': page.page_number}
                                )
                            )
                            diff_len -= max_characters
                        if i < len(t):  # non ho aggiunto tutte le righe
                            last_chunk = [cleaned_table[0]] + cleaned_table[i:]
                            tables.append(
                                TableElement(
                                    content=f'SplittedTable: {last_chunk}',
                                    metadata={'source': self.document.path.name, 'page': page.page_number}
                                )
                            )
                    else:
                        tables.append(
                            TableElement(
                                content=f'Table: {cleaned_table}',
                                metadata={'source': self.document.path.name, 'page': page.page_number}
                            )
                        )

        return tables
