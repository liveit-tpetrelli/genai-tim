import os
import re
from typing import Any, Dict, List

from fitz import fitz, Document, Page
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import VertexAI, ChatVertexAI
from unstructured.chunking import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval

app_configs = AppConfigs()
prompt_retrieval = PromptsRetrieval()

API_KEY = app_configs.configs.GoogleApplicationCredentials.api_key
os.environ["GOOGLE_API_KEY"] = API_KEY

gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

documents_dir_path = os.path.join(*app_configs.configs.Documents.path)
figures_storage_path = os.path.join(*app_configs.configs.Documents.Images.path)


def get_document(path: str):
    return fitz.Document(path)


class ImageElement:
    summary: str
    file_name: str
    ext: str
    metadata: Dict[str, Any]

    def __init__(self, summary: str, file_name: str, ext: str, metadata: Dict[str, Any]):
        self.summary = summary
        self.file_name = file_name
        self.ext = ext
        self.metadata = metadata


class ImagesRetrievalFromPdf:
    source_filename: str
    source_dir: str
    document: Document
    pages: List[Page]
    figures_storage: str

    def __init__(self, source_filename: str, source_dir: str = documents_dir_path, figures_storage: str = figures_storage_path):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = get_document(path=os.path.join(source_dir, source_filename))
        self.pages = [*self.document.pages()]
        self.figures_storage = figures_storage

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    def summarize_image(self, image_path: str) -> str:
        # llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
        llm = ChatVertexAI(model_name="gemini-pro-vision", temperature=0)
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
        summary = llm.invoke([message])
        return summary.content

    def get_image_elements(self):
        summaries = []
        for page in self.pages:  # iterate over pdf pages
            image_list = page.get_images()

            # for image in the page I need to save them and get a summary
            # try to swap the image in the pdf with the summary generated
            for image_index, img in enumerate(image_list, start=1):  # enumerate the image list
                xref = img[0]  # get the XREF of the image
                pix = fitz.Pixmap(self.document, xref)  # create a Pixmap

                if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # save the image as png
                file_name = f"page_{page.number}-image_{image_index}"
                ext = "png"
                image_path = os.path.join(self.figures_storage, f"{file_name}.{ext}")
                pix.save(image_path)

                # Invoke LLM to get a summary
                summary = self.summarize_image(image_path=image_path)

                summaries.append(ImageElement(
                        summary=summary.strip(),
                        file_name=file_name,
                        ext=ext,
                        metadata={'source': image_path, 'index': image_index, 'page': page.number}
                    )
                )

        return summaries
