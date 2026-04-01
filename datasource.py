import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
# 导入特定的读取器
from llama_index.readers.file import DocxReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata

def create_doc_tool(llm, docs_path="docs", top_k=3, debug=True):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5", trust_remote_code=True)
    Settings.llm = llm
    Settings.embed_model = embed_model

    persist_dir = "./storage"
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        if debug: print(f"🔍 正在从 {docs_path} 重新构建索引...")
        
        # 显式映射后缀名到解析器
        file_extractor = {".docx": DocxReader()}
        
        reader = SimpleDirectoryReader(
            input_dir=docs_path, 
            recursive=True,
            file_extractor=file_extractor  # 使用我们指定的提取器
        )
        documents = reader.load_data()
        
        # 调试：打印出每个文档读取到的前 100 个字符
        if debug:
            print(f"📄 成功加载了 {len(documents)} 个文档片段")
            for i, doc in enumerate(documents):
                preview = doc.text[:100].replace('\n', ' ')
                print(f"   [文档 {i+1} 预览]: {preview}...")

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        if debug: print("正在加载本地缓存索引...")
        storage_context = StorageContext.from_defaults(persist_directory=persist_dir)
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    return QueryEngineTool(
        metadata=ToolMetadata(
            name="knowledge_base",
            description="查询本地文档库，包括 NLP 介绍和项目提案内容。",
        ),
        query_engine=query_engine,
    )