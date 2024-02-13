import base64
import os

from PIL import Image
from io import BytesIO

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import VertexAI, ChatVertexAI

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.chains.multi_modal_rag_chain import multi_modal_rag_chain, conversational_multi_modal_rag_chain
from libraries.retrievals.MultiModalRetrieval import MultiModalRetrieval
from langchain.memory import ConversationBufferMemory

# app_configs = AppConfigs()
# gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)
#
# prompt_retrieval = PromptsRetrieval()
#
#
# def split_image_text_types(d):
#     base64s = []
#     text = []
#     for doc in d:
#         try:
#             base64.b64decode(doc)
#             base64s.append(doc)
#         except Exception as e:
#             text.append(doc)
#     return {
#         "images": base64s,
#         "texts": text
#     }
#
#
# def img_prompt_func(dictionary):
#     context = "\n".join(dictionary["context"]["texts"])
#     try:
#         return [
#             HumanMessage(
#                 content=[
#                     {
#                         "type": "text",
#                         "text": prompt_retrieval.prompts["combine_docs_and_images"].format(
#                             context=context,
#                             question=dictionary['question']),
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{dictionary['context']['images'][0]}"
#                         }
#                     },
#                 ]
#             )
#         ]
#     except Exception:
#         return [
#             HumanMessage(
#                 content=[
#                     {
#                         "type": "text",
#                         "text": prompt_retrieval.prompts["combine_docs"].format(
#                             context=context,
#                             question=dictionary['question']),
#                     }
#                 ]
#             )
#         ]
#
# def multi_modal_rag_chain(retriever):
#     """
#     Multi-modal RAG chain,
#
#     :param retriever: A function that retrieves the necessary context for the model.
#     :return: A chain of functions representing the multi-modal RAG process.
#     """
#     # Initialize the multi-modal Large Language Model with specific parameters
#     model = ChatVertexAI(model_name="gemini-pro-vision", temperature=0)
#
#     # Define the RAG pipeline
#     chain = (
#         {
#             "context": retriever | RunnableLambda(split_image_text_types),
#             "question": RunnablePassthrough(),
#         }
#         | RunnableLambda(img_prompt_func)
#         | model
#         | StrOutputParser()
#     )
#
#     return chain


def run_test():
    multi = MultiModalRetrieval()
    # multi = MultiModalRetrieval(from_persistent=False)

    # chain = multi_modal_rag_chain(multi.get_retriever())

    memory = ConversationBufferMemory(return_messages=True)
    memory.load_memory_variables({})
    chain = conversational_multi_modal_rag_chain(multi.get_retriever(), memory)

    inputs1 = "Come è fatta la schermata di login di ILIAD?"
    # inputs1 = {"input": "hi im bob"}
    response = chain.invoke(inputs1)
    print(response)

    memory.save_context({"input": inputs1}, {"output": response.content})
    memory.load_memory_variables({})
    print(memory)

    # inputs = {"input": "whats my name"}
    inputs = "A cosa serve il bottone rosso?"
    response = chain.invoke(inputs)
    print(response)


    # async for chunk in chain.astream(
    #         {"question": message.content},
    #         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # print(chain.invoke("Come è fatta la schermata di login di ILIAD?"))

    # print(chain.invoke("Cosa devo fare se dopo la sostituzione della CPE in scenario attivo (MI-TO-BO) la nuova CPE rimane bloccata in step 6-7?"))



run_test()

