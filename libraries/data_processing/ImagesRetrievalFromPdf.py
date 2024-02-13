import base64
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

_DOCUMENTS_DIR_PATH = os.path.join(*app_configs.configs.Documents.path)
_FIGURES_STORAGE_PATH = os.path.join(*app_configs.configs.Documents.Images.path)
_IMAGE_SUMMARIES_PATH = os.path.join(*app_configs.configs.Documents.ImageSummaries.path, app_configs.configs.Documents.ImageSummaries.file_name)



def get_document(path: str):
    return fitz.Document(path)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ImageElement(RetrievedElement):
    base64_image: str

    def __init__(self, content: str, metadata: Dict[str, Any], base64_image: str):
        super().__init__(content, metadata)
        self.base64_image = base64_image


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


def summarize_image(image_path: str) -> str:
    llm_model = ChatVertexAI(model_name="gemini-pro-vision", temperature=0)
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
    summary = llm_model.invoke([message])
    return summary.content


class ImagesRetrievalFromPdf:
    source_filename: str
    source_dir: str
    document: Document
    pages: List[Page]
    figures_storage: str

    def __init__(self,
                 source_filename: str,
                 source_dir: str = _DOCUMENTS_DIR_PATH,
                 figures_storage: str = _FIGURES_STORAGE_PATH):
        self.source_filename = source_filename
        self.source_dir = source_dir
        self.document = get_document(path=os.path.join(source_dir, source_filename))
        self.pages = [*self.document.pages()]
        self.figures_storage = figures_storage

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.document.close()

    def get_image_elements(self, save_result: bool = True):
        image_elements = []
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
                    summary = summarize_image(image_path=image_path)

                    image_elements.append(ImageElement(
                        content=summary.strip(),
                        base64_image=encode_image(image_path),
                        metadata={'location': image_path, 'file_name': file_name,
                                  'ext': ext, 'index': image_index,
                                  'source': self.document.name, 'page': page.number})
                    )
                except Exception as e:
                    print(e.args[0])

        if save_result:
            base64_images = [elem.base64_image for elem in image_elements]
            image_summaries = [elem.content for elem in image_elements]
            _ = save_summaries(base64_images, image_summaries)

        return image_elements
