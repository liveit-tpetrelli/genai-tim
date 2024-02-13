import os
import chainlit as cl

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableConfig
from langchain_google_vertexai import (VertexAI,
                                       VertexAIEmbeddings)
from langchain_community.vectorstores.chroma import Chroma

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.chains.multi_modal_rag_chain import multi_modal_rag_chain, conversational_multi_modal_rag_chain
from libraries.retrievals.MultiModalRetrieval import MultiModalRetrieval

from libraries.splitters.TokenElementSplitter import TokenElementSplitter


app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

prompt_retrieval = PromptsRetrieval()

chroma_persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.path)
vertex_embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")
google_llm_model = VertexAI(model_name="gemini-pro", temperature=0, streaming=True)


@cl.on_chat_start
async def on_conversational_chat_start():
    memory = ConversationBufferMemory(return_messages=True)
    memory.load_memory_variables({})
    multi_modal_retrieval = MultiModalRetrieval()
    chain = conversational_multi_modal_rag_chain(multi_modal_retrieval.get_retriever(), memory)
    cl.user_session.set("memory", memory)
    cl.user_session.set("chain", chain)


# @cl.on_chat_start
# async def on_chat_start():
#     multi_modal_retrieval = MultiModalRetrieval()
#     chain = multi_modal_rag_chain(multi_modal_retrieval.get_retriever())
#
#     cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")

    # msg = cl.Message(content="")

    # async for chunk in chain.astream(
    #         {"question": message.content},
    #         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    answer = chain.invoke(message.content)
    memory.save_context({"input": message.content}, {"output": answer})
    memory.load_memory_variables({})

    await cl.Message(content=str(answer)).send()

