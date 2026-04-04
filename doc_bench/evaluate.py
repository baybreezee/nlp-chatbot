from typing import Tuple, List, Dict, Literal
from pathlib import Path
import zipfile
import json
import asyncio

from llama_index.core import Document, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.readers.file import PDFReader      # use pypdf internally
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.agent.workflow import FunctionAgent, BaseWorkflowAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import CorrectnessEvaluator

DIR = Path(__file__).parent
DATA_ZIP = DIR / "data.zip"
DATA_FOLDER = DIR / "data"
MIN_IDX = 0
MAX_IDX = 228
QATypes = Literal["text-only", "multimodal-f", "multimodal-t", "unanswerable", "meta-data", "una-web"]
REF_TEMPLATE = """### Answer
{answer}

### Evidence (Evidence can help you make better judgement. It's not a mandatory part of a correct answer.)
{evidence}"""


def load_data_path(idx: int) -> Tuple[Path, Path]:
    assert idx >= MIN_IDX and idx <= MAX_IDX
    if not DATA_FOLDER.exists():
        with zipfile.ZipFile(DATA_ZIP) as zip_ref:
            zip_ref.extractall(DIR)
    folder = next(DATA_FOLDER.glob(f"{idx}"), None)
    assert folder is not None and folder.is_dir()
    qa_path = next(folder.glob(f"{idx}_qa.jsonl"), None)
    pdf_path = next(folder.glob("*.pdf"), None)
    return pdf_path, qa_path


def read_qa_file(qa_path: Path) -> List[Dict]:
    with open(qa_path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def load_pdf(path: Path) -> List[Document]:
    return SimpleDirectoryReader.load_file(
        input_file=path,
        file_metadata=default_file_metadata_func,
        file_extractor={".pdf": PDFReader()}
    )


async def get_response(agent: BaseWorkflowAgent, question: str):
    return await agent.run(question)


def run(
    model: str, 
    api_key: str, 
    base_url: str,
    temperature: float = 0.1,
    extra_body: Dict | None = None,     # openai extra_body
    context_window: int = 32000,
    mode: Literal["full_text", "chunks"] = "chunks",
    agentic: bool = False,      # single-turn + prompt template, or multi-turn + tool-calling
    chunk_size: int = 1024,     # be careful: chunk_size <= embed_model.max_seq_length
    chunk_overlap: int = 200,
    top_k: int = 5,
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5",
    idx_list: List[int] | None = None,
    qa_types: List[QATypes] | None = None,
    # judge model if specified
    eval_model: str | None = None,
    eval_api_key: str | None = None,
    eval_base_url: str | None = None
):
    llm = OpenAILike(
        model=model,
        api_key=api_key,
        api_base=base_url,
        temperature=temperature,
        additional_kwargs={"extra_body": extra_body},
        is_chat_model=True,
        is_function_calling_model=agentic,
        context_window=context_window
    )
    if eval_model and eval_api_key and eval_base_url:
        eval_llm = OpenAILike(
            model=eval_model,
            api_key=eval_api_key,
            api_base=eval_base_url,
            is_chat_model=True
        )
    else:
        eval_llm = llm
    if mode == "full_text":
        embed_model = None
    else:
        embed_model = HuggingFaceEmbedding(model_name=embed_model, trust_remote_code=True)
    idx_list = idx_list if idx_list else list(range(5))
    qa_types = qa_types if qa_types else ["text-only"]
    results = []
    for idx in idx_list:
        try:
            pdf_path, qa_path = load_data_path(idx)
            documents = load_pdf(pdf_path)
            qa = read_qa_file(qa_path)
            if mode == "full_text":
                index = SummaryIndex.from_documents(documents)
            else:
                index = VectorStoreIndex.from_documents(    # gpt-3.5-turbo tokenizer by default
                    documents,
                    transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)],
                    similarity_top_k=top_k,
                    embed_model=embed_model
                )
        except Exception as e:
            print(f"Failed to load {idx}: {e}")
            continue
        for data in qa:
            if not data["type"] in qa_types:
                continue
            results.append({
                "doc_idx": idx,
                "question": data["question"],
                "type": data["type"],
                "ref_answer": data["answer"],
                "evidence": data.get("evidence", ""),
                "generation_error": "",
                "evaluation_error": ""
            })
            try:
                if agentic:
                    query_engine = index.as_query_engine(llm=llm)
                    metadata = ToolMetadata(name="search_docs", description="Useful for answering questions.")
                    agent = FunctionAgent(
                        llm=llm, tools=[QueryEngineTool(query_engine=query_engine, metadata=metadata)]
                    )
                    response = asyncio.run(get_response(agent=agent, question=data["question"]))
                    response = response.response.content
                else:
                    chat_engine = index.as_chat_engine(chat_mode="context", llm=llm)
                    response = chat_engine.chat(data["question"]).response
            except Exception as e:
                print(f"Failed to generate answer (idx: {idx}, question: '{data['question']}'): {e}")
                results[-1]["generation_error"] = str(e)
                continue
            else:
                results[-1]["answer"] = response
            evaluator = CorrectnessEvaluator(llm=eval_llm)
            if data.get("evidence", ""):
                reference = REF_TEMPLATE.format(answer=data["answer"], evidence=data["evidence"])
            else:
                reference = data["answer"]
            try:
                result = evaluator.evaluate(query=data["question"], response=response, reference=reference)
            except Exception as e:
                print(f"Failed to evaluate (idx: {idx}, question: '{data['question']}'): {e}")
                results[-1]["evaluation_error"] = str(e)
            else:
                results[-1]["feedback"] = result.feedback
                results[-1]["score"] = result.score
                results[-1]["passing"] = result.passing
    return results


# test: questions for 10
if __name__ == "__main__":
    print(json.dumps(
        run(
            model="qwen3.5-0.8b",
            api_key="none",
            base_url="http://127.0.0.1:1234/v1",
            mode="chunks",
            agentic=False,
            idx_list=[10]
        ),
        ensure_ascii=False,
        indent=4
    ))
