import os
import re
from typing import Any, Dict, List

from fitz import fitz, Document, Page
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import VertexAI, ChatVertexAI
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.data_processing.RetrievedElement import RetrievedElement
from libraries.exceptions.ImageRetrievalException import ImageRetrievalException

app_configs = AppConfigs()
prompt_retrieval = PromptsRetrieval()

# API_KEY = app_configs.configs.GoogleApplicationCredentials.api_key
# os.environ["GOOGLE_API_KEY"] = API_KEY

gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

documents_dir_path = os.path.join(*app_configs.configs.Documents.path)
figures_storage_path = os.path.join(*app_configs.configs.Documents.Images.path)


def get_document(path: str):
    return fitz.Document(path)


class ImageElement(RetrievedElement):
    pass


class ImagesRetrievalFromPdf:
    source_filename: str
    source_dir: str
    document: Document
    pages: List[Page]
    figures_storage: str
    llm_model: BaseChatModel

    def __init__(self,
                 source_filename: str,
                 source_dir: str = documents_dir_path,
                 figures_storage: str = figures_storage_path,
                 llm: BaseChatModel = ChatVertexAI(model_name="gemini-pro-vision", temperature=0)):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = get_document(path=os.path.join(source_dir, source_filename))
        self.pages = [*self.document.pages()]
        self.figures_storage = figures_storage
        self.llm_model = llm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    def summarize_image(self, image_path: str) -> str:
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt_retrieval.prompts["summary_image"].template,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_path}"}
                },
            ]
        )
        summary = self.llm_model.invoke([message])
        return summary.content

    def get_image_elements(self):
        summaries = []
        for page in self.pages:  # iterate over pdf pages
            image_list = page.get_images()

            # for image in the page I need to save them and get a summary
            # try to swap the image in the pdf with the summary generated
            for image_index, img in enumerate(image_list, start=1):  # enumerate the image list
                try:
                    xref = img[0]  # get the XREF of the image
                    pix = fitz.Pixmap(self.document, xref)  # create a Pixmap

                    if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # save the image as png
                    file_name, ext = f"page_{page.number}-image_{image_index}", "png"
                    image_path = os.path.join(self.figures_storage, f"{file_name}.{ext}")
                    pix.save(image_path)

                    # Invoke LLM to get a summary
                    summary = self.summarize_image(image_path=image_path)

                    summaries.append(ImageElement(
                            content=summary.strip(),
                            metadata={'location': image_path, 'file_name': file_name,
                                      'ext': ext, 'index': image_index,
                                      'source': self.document.name, 'page': page.number})
                    )
                except ImageRetrievalException as e:
                    print(e.args[0])

        return summaries
