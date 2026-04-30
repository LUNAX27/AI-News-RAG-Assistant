from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

#读取数据
with open("data/ai_news.txt", "r", encoding="utf-8") as f:
    texts = f.read().split("\n\n")

#初始化
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key= "your-api-key"
)


#构建向量数据库
db = FAISS.from_texts(texts, embeddings)

#保存
db.save_local("faiss_index")

print("✅ 向量数据库构建完成！")
