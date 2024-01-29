import os
import re
from typing import Any, Dict, List

from fitz import fitz, Document, Page
from langchain_core.messages import HumanMessage
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.app_configs import AppConfigs

app_configs = AppConfigs()
documents_dir_path = os.path.join(*app_configs.configs.Documents.path)


class ImageElement:
    content: str
    metadata: Dict[str, Any]

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


class ImagesRetrievalFromPdf:
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

    # def __summarize_image(self, image_path: str) -> str:
    #     llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    #     message = HumanMessage(
    #         content=[
    #             {
    #                 "type": "text",
    #                 "text": "What's in this image?",
    #             },  # You can optionally provide text parts
    #             {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    #         ]
    #     )
    #     summary = llm.invoke([message])
    #     return summary.content

    def get_image_elements(self):
        all_images = []
        for page in self.pages:  # iterate over pdf pages
            image_list = page.get_images()
            all_images.extend(image_list)

            for image_index, img in enumerate(image_list, start=1):  # enumerate the image list
                xref = img[0]  # get the XREF of the image
                pix = fitz.Pixmap(self.document, xref)  # create a Pixmap

                if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                pix.save(os.path.join(self.source_dir, "figures", "page_%s-image_%s.png" % (page.number, image_index)))  # save the image as png
                pix = None

        return all_images