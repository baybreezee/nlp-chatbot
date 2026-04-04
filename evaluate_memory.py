import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
# 导入你的 memory 组件
from memory import create_long_short_memory # 导入你的封装函数

# 测评数据集：每一项包含用户输入、是否是测试问题，以及期望的回答核心点
EVAL_DATASET = [
    # --- 阶段 1: 建立事实记忆 (Fact) ---
    {"turn": 1, "role": "user", "content": "你好，我是张三，我是一名在深圳工作的后端工程师。", "is_test": False},
    {"turn": 2, "role": "user", "content": "我非常喜欢喝美式咖啡，每天早上都要来一杯。", "is_test": False},

    # --- 阶段 2: 短期记忆测试 (Short-term) ---
    {"turn": 3, "role": "user", "content": "我刚才说我喜欢喝什么来着？", "is_test": True, "expected": "美式咖啡",
     "dimension": "短期记忆"},

    # --- 阶段 3: 制造大量对话“冲刷”短期记忆，触发进入向量库 (Long-term Flush) ---
    {"turn": 4, "role": "user", "content": "随便聊聊吧，你知道量子力学吗？请给我讲500字以上的科普。", "is_test": False},
    {"turn": 5, "role": "user", "content": "那相对论呢？再给我讲500字。", "is_test": False},
    {"turn": 6, "role": "user", "content": "继续，讲讲宇宙大爆炸，越长越好。", "is_test": False},

    # --- 阶段 4: 长期向量记忆测试 (Long-term Vector) ---
    {"turn": 7, "role": "user", "content": "由于聊了太多物理，你还记得我叫什么名字，在哪里工作吗？", "is_test": True,
     "expected": "张三，深圳的后端工程师", "dimension": "长期记忆/事实提取"},

    # --- 阶段 5: 记忆更新测试 (Memory Update) ---
    {"turn": 8, "role": "user",
     "content": "我上个月辞职了，现在搬到了成都，做一名全职自由撰稿人。并且因为肠胃不好，我把咖啡戒了，现在只喝绿茶。",
     "is_test": False},
    {"turn": 9, "role": "user", "content": "朋友想请我喝东西，你觉得我会点什么？另外我现在的工作是什么？", "is_test": True,
     "expected": "点绿茶，工作是自由撰稿人（不能提咖啡和后端工程师）", "dimension": "记忆更新与冲突解决"},
]


async def evaluate():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    llm = OpenAILike(
        model="deepseek-chat",
        api_key=api_key,
        api_base="https://api.deepseek.com",
        temperature=0.1,
        is_chat_model=True,  # 声明这是一个对话模型
        is_function_calling_model=True  # 声明它支持函数调用
    )

    # ⚠️ 测评核心：为了快速触发向量库，覆盖默认参数，调小 token_limit
    print("⏳ 初始化极速记忆系统 (Token Limit=500, Flush=200)...")
    memory = create_long_short_memory(llm=llm, session_id="eval_user_001")
    # 强制修改 memory 的阈值参数进行压力测试
    memory.token_limit = 500
    memory.token_flush_size = 200

    agent = FunctionAgent(llm=llm)

    results = []
    print("\n🚀 开始自动化测评...\n" + "=" * 40)

    for step in EVAL_DATASET:
        user_input = step["content"]
        print(f"\n👤 User (Turn {step['turn']}): {user_input[:50]}...")

        response = await agent.run(user_input, memory=memory)

        if step["is_test"]:
            print(f"🤖 Agent 回答: {response}")
            print(f"🎯 期望关键点: {step['expected']}")

            # 使用 LLM 作为裁判 (LLM-as-a-Judge) 自动评分
            judge_prompt = f"""
            你是一个严苛的裁判。请判断Agent的回答是否包含了期望的关键信息。
            用户问题: {user_input}
            期望关键点: {step['expected']}
            Agent回答: {response}
            如果回答正确且包含了期望信息，请仅输出 "PASS"。否则仅输出 "FAIL"。
            """
            eval_res = await llm.acomplete(judge_prompt)
            score = "✅ PASS" if "PASS" in eval_res.text else "❌ FAIL"

            print(f"📊 维度: [{step['dimension']}] -> {score}")
            results.append({"turn": step["turn"], "dimension": step["dimension"], "score": score})
        else:
            print(f"🤖 Agent:{response}")

    # 打印最终记忆状态
    print("\n" + "=" * 40 + "\n🧠 测评结束，记忆系统最终状态：")
    for block in memory.memory_blocks:
        if block.name == "extracted_info":
            print(f"\n📌 事实提取库:\n{block.facts}")
        if block.name == "vector_memory":
            vec_size = len(block.vector_store.to_dict().get('embedding_dict', {}))
            print(f"\n📦 向量长期库: 共挤入 {vec_size} 个节点")

    # 打印总成绩单
    print("\n📈 测评成绩单:")
    for r in results:
        print(f" - Turn {r['turn']} [{r['dimension']}]: {r['score']}")


if __name__ == "__main__":
    asyncio.run(evaluate())