# main.py
import os, asyncio
from dotenv import load_dotenv

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent

from memory import create_long_short_memory
from datasource import create_doc_tool

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("API_KEY")
hf_token = os.getenv("HF_TOKEN")

print(f"检查 API Key: {api_key[:10]}******")
print(f"检查 HF Token: {hf_token[:10]}******")

# 2. 初始化 LLM (DeepSeek)
llm = OpenAILike(
    model="deepseek-chat",
    api_key=api_key,
    api_base="https://api.deepseek.com",
    temperature=0.1,
    is_chat_model=True,
    is_function_calling_model=True,
)

# 3. 初始化记忆系统
memory = create_long_short_memory(llm=llm, session_id="user_123")

# 4. 初始化文档工具
doc_tool = create_doc_tool(llm, docs_path="docs", top_k=3, debug=True)

# 5. 初始化 Agent，注册工具
agent = FunctionAgent(llm=llm, tools=[doc_tool])

# 6. 主循环
async def main():
    print("ok")
    while True:
        user_msg = input("\n👤 你: ")
        if user_msg.lower() in ["exit", "quit", "退出"]:
            break

        response = await agent.run(user_msg, memory=memory)
        print(f"\n🤖 Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
