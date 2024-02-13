import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.prompts import load_prompt
from langchain_google_vertexai import (VertexAI,
                                       VertexAIEmbeddings)
from langchain_community.vectorstores.chroma import Chroma

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.splitters.TokenElementSplitter import TokenElementSplitter
from libraries.retrievals.MultiModalRetrieval import MultiModalRetrieval

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

prompt_retrieval = PromptsRetrieval()

chroma_persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.path)
vertex_embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")
google_llm_model = VertexAI(model_name="gemini-pro", temperature=0)


def start_chatbot():
    token_splitter = TokenElementSplitter(source_filename="Manuale operativo iliad FTTH (1).pdf")
    print(token_splitter.split_document())
    if not os.listdir(chroma_persist_directory):
        token_splitter = TokenElementSplitter(source_filename="Manuale operativo iliad FTTH (1).pdf")
        token_splits = token_splitter.split_document()

        vector_chromadb = Chroma.from_documents(
            documents=token_splits,
            persist_directory=chroma_persist_directory,
            embedding=vertex_embeddings_model
        )
    else:
        vector_chromadb = Chroma(
            persist_directory=chroma_persist_directory,
            embedding_function=vertex_embeddings_model
        )

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
        condense_question_prompt=prompt_retrieval.prompts["condense_question"],
        combine_docs_chain_kwargs={"prompt": prompt_retrieval.prompts["combine_docs"]},
        rephrase_question=True,
        chain_type="stuff",
        verbose=True
    )
    return chain


if __name__ == '__main__':

    want_to_chat = True
    qa_chain = start_chatbot()

    while want_to_chat:
        user_input = input("You: ")
        if user_input.lower() != "quit":
            question = user_input
            res = qa_chain.invoke({"question": question})
            answer = res['answer']
            print(f"Chatbot: {answer}")
        else:
            want_to_chat = False

    print("""
    |----------------------------------------------------------|
    |                 CHATBOT SESSION TERMINATED.              |
    |----------------------------------------------------------|
    """)
