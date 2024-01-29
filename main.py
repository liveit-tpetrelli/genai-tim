import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter
from langchain_google_vertexai import (VertexAI,
                                       VertexAIEmbeddings)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from configs.app_configs import AppConfigs
from libraries.data_processing.TablesRetrievalFromPdf import TablesRetrievalFromPdf
from libraries.data_processing.TextRetrievalFromPdf import TextRetrievalFromPdf

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    *gcloud_credentials.dir_path,
    gcloud_credentials.file_name
)

path = "./documents/"
chroma_persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.path)
vertex_embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with TextRetrievalFromPdf("Manuale operativo iliad FTTH (1).pdf") as ret:
        texts = ret.get_text_elements()

    # TODO - To retrieve images
    # with ImagesRetrievalFromPdf("Manuale operativo iliad FTTH (1).pdf") as ret:
    #     images = ret.get_image_elements()

    with TablesRetrievalFromPdf("Manuale operativo iliad FTTH (1).pdf") as ret:
        tables = ret.get_table_elements(max_characters=1000)

    docs = []
    for table in tables:
        docs.append(Document(
            page_content=table.content,
            metadata=table.metadata))
    for text in texts:
        docs.append(Document(
            page_content=text.text,
            metadata={'source': text.metadata.fields['filename'], 'page_number': text.metadata.fields['page_number']}))

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

    # TODO - Altra chain proposta da prendere in considerazione
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
        rephrase_question=True,
        chain_type="stuff",
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
