# memory.py
from llama_index.core.memory import (
    Memory,
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def create_long_short_memory(llm, session_id="default_user", vector_store=None):
    """
    创建一个集成了 静态、事实提取、向量检索 的混合记忆系统
    """
    # 1. 初始化 Embedding 模型
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    
    # 2. 如果没传入 vector_store，默认使用内存存储
    if vector_store is None:
        vector_store = SimpleVectorStore()
    vector_store.stores_text = True

    # 3. 定义记忆块 (Blocks)
    blocks = [
        # 静态信息：人设、基础背景
        StaticMemoryBlock(
            name="core_info",
            static_content="My name is Logan", #可以自定义修改
            priority=0,
        ),
        # 事实提取：自动总结对话细节
        FactExtractionMemoryBlock(
            name="extracted_info",
            llm=llm,
            max_facts=50,
            priority=1,
        ),
        # 长期向量记忆：基于语义检索历史
        VectorMemoryBlock(
            name="vector_memory",
            vector_store=vector_store,
            embed_model=embed_model,
            priority=2,
            similarity_top_k=5,
            retrieval_context_window=5,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        ),
    ]

    # 4. 封装成 Memory 对象
    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=8000, #要存储的短期和长期记忆的最大数量
        token_flush_size=2000, # 每次短期记忆满了，挪出 2000 token 存入向量库
        memory_blocks=blocks,
        chat_history_token_ratio=0.7, # 70% 的空间留给短期记忆（最近的对话）
        insert_method="system", # 记忆将以 System Message 的形式插入
    )
    
    return memory