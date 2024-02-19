import base64
import binascii
import os
from operator import itemgetter
from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI

from configs.app_configs import AppConfigs
from configs.prompts.PromptsRetrieval import PromptsRetrieval
from libraries.exceptions.CheckImageRelevancyException import CheckImageRelevancyException

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
                            question=dictionary['condensed_question'],)
                            # few_shots="    \n\n".join(prompt_retrieval.few_shots_examples)),
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


def __conversational_multi_modal_rag_chain(retriever, memory):
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


def invoke_multi_modal_chain(question, chain):
    response = {}

    context = chain["retrieval_chain"].invoke(question)
    response["answer"] = chain["chain"].invoke({"context": context, "question": question})

    if context["images"]:
        response["image"] = context["images"][0]

    chain["memory"].save_context({"input": question}, {"output": response["answer"]})
    chain["memory"].load_memory_variables({})

    return response


def conversational_multi_modal_rag_chain(
        retriever: BaseRetriever,
        memory: BaseMemory,
        get_source_documents: bool = False,
        llm_model: BaseChatModel = ChatVertexAI(model_name="gemini-pro-vision", temperature=0, streaming=True)
) -> Dict[str, Any]:

    result = {
        "memory": memory,
        "retriever": retriever
    }

    if get_source_documents:
        retrieval_chain = (retriever | RunnableLambda(__split_image_text_types))
        chain = (
            {
                "context": itemgetter("context") | RunnablePassthrough(),
                "question": itemgetter("question") | RunnablePassthrough()
            }
            | RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
            | RunnableLambda(__history_prompt_func)
            | RunnableLambda(__img_prompt_func)
            | llm_model
            | StrOutputParser()
        )

        result |= {
            "retrieval_chain": retrieval_chain,
            "chain": chain,
        }
    else:
        chain = __conversational_multi_modal_rag_chain(retriever, memory)
        result |= {
            "chain": chain,
        }

    return result


def check_image_relevancy(text: str, image: str) -> None:
    llm_model = ChatVertexAI(model_name="gemini-pro-vision", temperature=0)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_retrieval.prompts["check_image_relevancy"].format(
                    text=text
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            },
        ]
    )
    relevancy = llm_model.invoke([message])
    print(relevancy)
    if "FALSE" in relevancy.content:
        raise CheckImageRelevancyException("The image retrieved is not relevant and will not be displayed.")
