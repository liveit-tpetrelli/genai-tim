import base64
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
from libraries.chains.multi_modal_rag_chain import multi_modal_rag_chain, conversational_multi_modal_rag_chain, \
    invoke_multi_modal_chain, check_image_relevancy
from libraries.exceptions.CheckImageRelevancyException import CheckImageRelevancyException
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
    multi_modal_retrieval = MultiModalRetrieval()

    chain = conversational_multi_modal_rag_chain(
        retriever=multi_modal_retrieval.get_retriever(),
        memory=memory,
        get_source_documents=True
    )

    cl.user_session.set("chain", chain)

# @cl.on_chat_start
# async def on_conversational_chat_start():
#     memory = ConversationBufferMemory(return_messages=True)
#     memory.load_memory_variables({})
#     multi_modal_retrieval = MultiModalRetrieval()
#     chain = conversational_multi_modal_rag_chain(multi_modal_retrieval.get_retriever(), memory)
#     cl.user_session.set("memory", memory)
#     cl.user_session.set("chain", chain)


# @cl.on_chat_start
# async def on_chat_start():
#     multi_modal_retrieval = MultiModalRetrieval()
#     chain = multi_modal_rag_chain(multi_modal_retrieval.get_retriever())
#
#     cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    # memory = cl.user_session.get("memory")

    # msg = cl.Message(content="")

    # async for chunk in chain.astream(
    #         {"question": message.content},
    #         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # answer = chain.invoke(message.content)
    # memory.save_context({"input": message.content}, {"output": answer})
    # memory.load_memory_variables({})

    response = invoke_multi_modal_chain(
        question=message.content,
        chain=chain
    )

    # if "image" in response.keys():
    try:

        print("1 - ", response["answer"])
        # image = cl.Image(path="documents/figures/page_5-image_4.png", name="image1", display="inline")
        image = cl.Image(content=base64.b64decode(response["image"]), display="inline", size="large")

        # This method can raise a CheckImageRelevancyException if the image is useless.
        check_image_relevancy(text=str(response["answer"]), image=response["image"])

        await cl.Message(content=str(response["answer"]), elements=[image],).send()

    except CheckImageRelevancyException as e:
        print(e)
    # else:
        print("2 - ", response["answer"])
        await cl.Message(content=str(response["answer"])).send()

    except Exception as e:
        print("General Exception", e)

