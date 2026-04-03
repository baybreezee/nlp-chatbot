"""DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems."""

from .evaluate import run, load_data_path, read_qa_file, load_pdf

__all__ = ["run", "load_data_path", "read_qa_file", "load_pdf"]
