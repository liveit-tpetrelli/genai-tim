import base64
import binascii
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.exceptions.NoImagesRetrievedException import NoImagesRetrievedException

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(*gcloud_credentials.dir_path, gcloud_credentials.file_name)

prompt_retrieval = PromptsRetrieval()


def __split_image_text_types(retrieved_documents):
    base64_images = []
    texts = []
    for doc in retrieved_documents:
        try:
            base64.b64decode(doc)
            base64_images.append(doc.decode())
        except binascii.Error:
            texts.append(doc.decode())
    return {
        "images": base64_images,
        "texts": texts
    }


def __history_prompt_func(data):
    print(data)
    if data['history']:
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt_retrieval.prompts["condense_question"].format(
                            chat_history=data['history'],
                            question=data['question'])
                    }
                ]
            )
        ]
        model = ChatVertexAI(model_name="gemini-pro", temperature=0)
        result = model.invoke(messages)
        return {"condensed_question": result.content, "history": data['history'], "context": data['context']}
    else:
        return {"condensed_question": data['question'], "history": data['history'], "context": data['context']}


def __img_prompt_func(dictionary):
    context = "\n".join(dictionary["context"]["texts"])
    try:
        return [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt_retrieval.prompts["combine_docs_and_images"].format(
                            context=context,
                            question=dictionary['condensed_question']),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{dictionary['context']['images'][0]}"
                        }
                    },
                ]
            )
        ]
    except IndexError:
        return [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt_retrieval.prompts["combine_docs"].format(
                            context=context,
                            question=dictionary['condensed_question']),
                    }
                ]
            )
        ]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain,

    :param retriever: A function that retrieves the necessary context for the model.
    :return: A chain of functions representing the multi-modal RAG process.
    """
    # Initialize the multi-modal Large Language Model with specific parameters
    model = ChatVertexAI(model_name="gemini-pro-vision", temperature=0, streaming=True)

    # Define the RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(__split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(__img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain


def conversational_multi_modal_rag_chain(retriever, memory):

    from operator import itemgetter
    # Initialize the multi-modal Large Language Model with specific parameters
    model = ChatVertexAI(model_name="gemini-pro-vision", temperature=0, streaming=True)
    chain = (
        {
            "context": retriever | RunnableLambda(__split_image_text_types),
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
        | RunnableLambda(__history_prompt_func)
        | RunnableLambda(__img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain
