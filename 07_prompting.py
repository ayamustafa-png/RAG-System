"""
المرحلة 7: تجهيز الـ LLM (Qwen2.5)، بناء الـ prompt، وتكوين سلسلة الـ
RAG (retrieval + generation) اللي بترجع الإجابة النهائية.
نفس المنطق اللي كان في app.py قبل التقسيم — من غير أي تغيير في القيم أو السلوك.
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from config import (
    SYSTEM_PROMPT,
    LLM_REPO_ID,
    LLM_TASK,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
)


def build_llm(hf_token):
    """بتبني كائن الـ LLM (Qwen2.5-7B-Instruct) باستخدام مفتاح HuggingFace."""
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task=LLM_TASK,  # this model is only served as a chat model on the "together" provider
        provider=LLM_PROVIDER,  # let Hugging Face route to whichever partner currently serves this model
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm_endpoint)


def build_prompt_template():
    """بتبني الـ prompt template اللي بيتحط فيه السياق المسترجع + سؤال المستخدم."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )


def build_rag_chain(llm, retriever):
    """بتربط الـ retriever بالـ LLM في سلسلة واحدة (retrieval + generation)."""
    prompt_template = build_prompt_template()
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, question_answer_chain)


def generate_answer(rag_chain, query):
    """بتنفّذ السلسلة كاملة وترجع dict فيه answer + context (المصادر المستخدمة)."""
    return rag_chain.invoke({"input": query})
