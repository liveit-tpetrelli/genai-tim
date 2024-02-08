import os
import re
from typing import List

from fitz import fitz, Document, Page
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from configs.app_configs import AppConfigs
from libraries.data_processing.RetrievedElement import RetrievedElement

app_configs = AppConfigs()
documents_dir_path = os.path.join(*app_configs.configs.Documents.path)


def get_document(path: str):
    return fitz.Document(path)


class TextElement(RetrievedElement):
    pass


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

    # Experimental get_text_elements_by_titles method
    '''
    def get_text_elements_by_titles(self) -> List[TextElement]:
        elements = partition_pdf(
            filename=os.path.join(self.source_dir, self.source_filename),
            # strategy='hi_res',
            chunking_strategy="by_title",

        )
        # Prendere il testo pdf e salvarlo in una stringa
        # Trovare il ToC (Table of Content)
        all_text = ""
        toc = []
        for page in self.pages:
        # for element in elements:
            current_text = page.get_text()
            # current_text = element.text
            all_text += f"\n{current_text}"
            if "sommario" in current_text.lower():
                text = current_text
                index = text.lower().find("sommario")
                text = text[index+len("sommario"):]
                regex_numbering = re.compile(r'^(\d+(\.\d+)*)\.?')
                regex_title = re.compile(r'\b(\D+)\b')

                titles = []
                numbers = []
                for line in list(filter(None, ["" if line == " " else line for line in text.split('\n')])):

                    title_match = regex_title.search(line)
                    numbering_match = regex_numbering.search(line)

                    if numbering_match:
                        current_numbering = line.strip()
                        numbers.append(current_numbering)

                    if title_match:
                        title = re.sub(r'\.+', '', title_match.group(1)).strip()
                        if title:
                            # Remove series of dots from the title
                            titles.append(title)

                toc = list(zip(numbers, titles))
                # toc = [f"{n} {t}" for n, t in toc]
                print(toc)
        chunks = []
        # Splittare il testo seguendo i titoli del ToC
        all_text = all_text.replace("\n", " ")
        for i in range(len(toc)-1):

            # index = all_text.find(toc[i])
            try:
                index = re.search(fr"{toc[i][0]}\s+{toc[i][1]}\s+[^.]", all_text).start()
                # end_index = all_text.find(toc[i+1])
                end_index = re.search(rf"{toc[i+1][0]}\s+{toc[i+1][1]}\s+[^.]", all_text).start()

                if i == 0:
                    chunks.append(all_text[:index])

                chunk = all_text[index:end_index]
                chunks.append(chunk)

                if i == len(toc) - 1:
                    chunks.append(all_text[end_index:])
            except:
                print("exception.")

        print(chunks)

        # Es: for title in toc: chunk = search_for_title(title); append(chunk)
    '''