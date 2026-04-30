from config import DASHSCOPE_API_KEY

from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings


embeddings = DashScopeEmbeddings(
    model= "text-embedding-v2",
    dashscope_api_key="your-api-key"
)
from langchain_core.prompts import ChatPromptTemplate

import os
index_path = "C:\\Users\\Luna3\\Desktop\\Experience\\ai-rag\\faiss_index"
db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# 初始化模型
llm = ChatTongyi(dashscope_api_key=DASHSCOPE_API_KEY)

retriever = db.as_retriever(search_kwargs={"k": 3})

# RAG Prompt
prompt = ChatPromptTemplate.from_template("""
你是一个AI领域助手，请根据以下资料回答问题。

资料：
{context}

问题：
{question}

请用简洁清晰的语言回答。
""")

while True:
    query = input("\n请输入问题（输入q退出）：")
    if query == "q":
        break

    #检索
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    print("检索到的内容：")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}")

    final_prompt = prompt.format(context=context, question=query)
    result = llm.invoke(final_prompt)

    print("回答：")
    print(result)
