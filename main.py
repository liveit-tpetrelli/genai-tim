import os

import pdfplumber
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TokenTextSplitter
from langchain_google_vertexai import (VertexAI,
                                       VertexAIEmbeddings)
from langchain_community.vectorstores.chroma import Chroma
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

from configs.app_configs import AppConfigs
from libraries.data_processing.ImagesRetrievalFromPdf import ImagesRetrievalFromPdf
from libraries.data_processing.TextRetrievalFromPdfs import TextRetrievalFromPdfs

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    *gcloud_credentials.dir_path,
    gcloud_credentials.file_name
)

path = "./documents/"
chroma_persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.path)
# notion_persist_directory = os.path.join(*app_configs.configs.NotionLocalDbConfigs.persist_directory)
vertex_embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")


class Table:
    text: str
    metadata: dict

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get elements
    # raw_pdf_elements = partition_pdf(
    #     filename=path + "Manuale operativo iliad FTTH (1).pdf",
    #     # Unstructured first finds embedded image blocks
    #     extract_images_in_pdf=True,
    #     strategy="hi_res",
    #     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    #     # Titles are any sub-section of the document
    #     infer_table_structure=True,
    #     # Post processing to aggregate text once we have the title
    #     chunking_strategy="by_title",
    #     # Chunking params to aggregate text blocks
    #     # Attempt to create a new chunk 3800 chars
    #     # Attempt to keep chunks > 2000 chars
    #     max_characters=4000,
    #     new_after_n_chars=3800,
    #     combine_text_under_n_chars=2000,
    #     extract_image_block_types=["Image", "Table"],
    #     extract_image_block_to_payload=False,
    #     extract_image_block_output_dir=path + 'images/',
    # )
    # print(raw_pdf_elements)
    #
    # category_counts = {}
    #
    # for element in raw_pdf_elements:
    #     category = str(type(element))
    #     if category in category_counts:
    #         category_counts[category] += 1
    #     else:
    #         category_counts[category] = 1
    #
    # # Unique_categories will have unique elements
    # unique_categories = set(category_counts.keys())
    # print(category_counts)
    # loader = UnstructuredFileLoader(path + 'Manuale operativo iliad FTTH (1).pdf', mode="elements")
    # docs = loader.load()
    # # print(docs)
    #
    # token_splitter = TokenTextSplitter(
    #     chunk_size=2000,
    #     chunk_overlap=0
    # )
    # token_splits = token_splitter.split_documents(docs)
    # print(token_splits)
    # with TextRetrievalFromPdfs("Manuale operativo iliad FTTH (1).pdf") as ret:
    #     blocks = ret.get_text_elements()

    with ImagesRetrievalFromPdf("Manuale operativo iliad FTTH (1).pdf") as ret:
        images = ret.get_image_elements()

    elements = partition_pdf(
        filename=path + "Manuale operativo iliad FTTH (1).pdf",
        infer_table_structure=True,
        chunking_strategy="by_title",
        strategy='hi_res',
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    tables = []
    with pdfplumber.open(path + "Manuale operativo iliad FTTH (1).pdf",) as pdf:
        for page in pdf.pages:
            tabs = page.extract_tables()
            if len(tabs) > 0:
                for t in tabs:
                    tables.append(Table(text=f'Tabella: {t}', metadata={'source': pdf.path.name, 'page': page.page_number}))

    # tables = [el for el in elements if el.category == "Table"]
    texts = [el for el in elements if el.category == "CompositeElement"]

    # print(tables[0].text)
    # print(tables[0].metadata.text_as_html)

    docs = []
    for table in tables:
        docs.append(Document(page_content=table.text, metadata=table.metadata))
    for text in texts:
        docs.append(Document(page_content=text.text, metadata={'source': text.metadata.fields['filename'], 'page_number': text.metadata.fields['page_number']}))

    print(len(docs))


    # Split
    token_splitter = TokenTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    token_splits = token_splitter.split_documents(docs)

    vector_chromadb = Chroma.from_documents(
        documents=token_splits,
        persist_directory=chroma_persist_directory,
        embedding=vertex_embeddings_model
    )

    google_llm_model = VertexAI(model_name="gemini-pro", temperature=0)

    # chain = RetrievalQAWithSourcesChain.from_llm(
    #     llm=google_llm_model,
    #     retriever=vector_chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    #     verbose=True
    # )

    chain = ConversationalRetrievalChain.from_llm(
        llm=google_llm_model,
        retriever=vector_chromadb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        ),
        # condense_question_prompt=condense_question_prompt,
        # return_source_documents=True,
        rephrase_question=True,
        chain_type="stuff",
        # combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
        verbose=True
    )

    want_to_chat = True
    while want_to_chat:
        user_input = input("You: ")
        if user_input.lower() != "quit":
            question = user_input
            res = chain.invoke({"question": question})
            answer = res['answer']
            print(f"Chatbot: {answer}")
        else:
            want_to_chat = False

