# main.py
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
# from llama_index.core.agent.workflow import ReActAgent
from memory import create_long_short_memory # 导入你的封装函数
import os,asyncio
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike

load_dotenv()
api_key = os.getenv("API_KEY")
print(f"检查 API Key: {api_key[:10]}******")
# 1. 初始化基础组件
llm = OpenAILike(
    model="deepseek-chat", 
    api_key=api_key, 
    api_base="https://api.deepseek.com",
    temperature=0.1,
    is_chat_model=True,
    is_function_calling_model=True
    )

# 2. 获取封装好的记忆系统
memory = create_long_short_memory(llm=llm, session_id="user_123")

# 3. 初始化 Agent
agent = FunctionAgent(llm=llm)

# 4. 运行 Agent
# 在 Workflow 模式下，记忆通常通过 ctx 管理，或者直接在 run 时传递
async def main():
    print("ok")
    while True:
        # 3. 获取用户输入
        user_msg = input("\n👤 你: ")
        if user_msg.lower() in ["exit", "quit", "退出"]:
            break

        # 4. Agent 运行 (它会自动读取并更新传入的 memory)
        response = await agent.run(user_msg, memory=memory)

        # 5. 输出回答
        print(f"\n🤖 Agent: {response}")
    
    # print("\n--- 🔍 记忆状态监控 ---")
    # for block in memory.memory_blocks:
    #     if block.name == "extracted_info":
    #         # 打印出事实提取块里的内容
    #         print(f"📌 已提取的事实: {block.get_content()}")
    #     if block.name == "vector_memory":
    #         # 看看向量库里现在有几条“陈年旧账”
    #         print(f"📦 长期向量库大小: {len(block.vector_store.to_dict().get('embedding_dict', {}))} 条")
    # print("----------------------")
    # response = await agent.run(
    #     "Hey, remember what my job is? Also, I like coffee.", 
    #     memory=memory
    # )
    # print(response)
if __name__ == "__main__":
    asyncio.run(main())
