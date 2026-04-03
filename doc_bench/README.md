# DocBench Evaluation

## Data Download

- Download [zip file](https://drive.google.com/file/d/1HwPbi9-7Zo-nLzUassUEVaMxyckjcMwS/view?usp=share_link).
- Or download from the original link below.
- Put the zip file or the folder under this directory.

```
doc_bench/
├── data.zip
├── data
│   └── ...
└── ...
```

## Usage

DocBench can be run as a Python module from the command line. Results are saved to a JSON file with metadata and evaluation scores.

### Quick Start

```bash
# Install dependencies first
pip install -r requirements.txt

# Basic usage (tests first 5 documents with text-only questions)
python -m doc_bench \
    --model qwen2.5-7b \
    --api-key YOUR_API_KEY \
    --base-url http://localhost:1234/v1
```

### Command-Line Arguments

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | **Model name** to use for evaluation | `gpt-4`, `qwen2.5-7b`, `llama3.1-8b` |
| `--api-key` | **API key** for authentication | `sk-...` or `none` for local models |
| `--base-url` | **API endpoint URL** | `http://localhost:1234/v1` (LM Studio), `https://api.openai.com/v1` |

#### Document & Question Selection

| Argument | Description | Default |
|----------|-------------|---------|
| `--idx-list` | **Which documents to test** (space-separated indices, 0-228) | First 5 documents |
| `--qa-types` | **Question categories** to include (see table below) | `text-only` |

**Question Types Explained:**

| Type | Description |
|------|-------------|
| `text-only` | Questions answerable from text alone |
| `multimodal-f` | Questions requiring figures/images in the document |
| `multimodal-t` | Questions requiring tables/visual elements |
| `unanswerable` | Questions intentionally unanswerable from the document |
| `meta-data` | Questions about document metadata (number of pages, etc.) |
| `una-web` | Questions requiring external/web knowledge |

#### Processing Mode

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | How the model reads documents: `full_text` or `chunks` | `chunks` |
| `--agentic` | Enable **multi-turn agent** with tool calling | Disabled |

**Mode Explanation:**
- **`chunks`** (recommended): Splits documents into smaller chunks, retrieves only relevant sections for each question. Faster and more focused.
- **`full_text`**: Feeds the entire document to the model. May exceed context window for long documents.

**Agentic Mode:**
- **Off (default)**: Single-turn question answering with context provided upfront.
- **On (`--agentic`)**: Multi-turn conversation where the model can search the document multiple times to find answers.

#### Chunking Parameters (only for `--mode chunks`)

| Argument | Description | Default | What it means |
|----------|-------------|---------|---------------|
| `--chunk-size` | Tokens per chunk | 1024 | How big each document slice is |
| `--chunk-overlap` | Overlap between chunks | 200 | Prevents cutting sentences at boundaries |
| `--top-k` | Chunks to retrieve | 5 | How many relevant chunks feed into the answer |
| `--embed-model` | Embedding model | `nomic-ai/nomic-embed-text-v1.5` | Model that converts text to vectors for search |

> **Tip:** If your embedding model has a max sequence length of 512, reduce `--chunk-size` to 512 or less.

#### LLM Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--temperature` | Randomness in generation (0-1) | 0.1 |
| `--context-window` | Model's maximum token capacity | 32000 |

#### Output

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Path to save results JSON | `results_<timestamp>.json` |

### Examples

```bash
# Test specific documents (indices 10, 20, 30)
python -m doc_bench --model gpt-4 --api-key KEY --base-url URL --idx-list 10 20 30

# Use full text mode instead of chunks
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL --mode full_text

# Enable agentic mode with tool calling
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL --agentic

# Test only multimodal questions on first 3 documents
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL \
    --idx-list 0 1 2 --qa-types multimodal-f multimodal-t

# Custom output file
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL -o my_results.json
```

### Understanding Results

The output JSON contains:

```json
{
  "metadata": {
    "model": "qwen2.5-7b",
    "mode": "chunks",
    "total_questions": 15,
    "timestamp": "2025-04-03T10:30:00"
  },
  "results": [
    {
      "doc_idx": 0,
      "question": "What is the main finding?",
      "ref_answer": "...",
      "answer": "...",
      "score": 4.5,
      "passing": true,
      "feedback": "..."
    }
  ]
}
```

**Score interpretation:**
- **Score**: 1-5 rating of answer quality (higher is better)
- **Passing**: Boolean indicating if the answer meets quality threshold
- **Feedback**: Explanation of the evaluation

## Reference

- [repo link](https://github.com/Anni-Zou/DocBench)
- [paper link](https://arxiv.org/pdf/2407.10701)
- [data link](https://drive.google.com/drive/folders/1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr)
- **citation**

```
@misc{zou2024docbenchbenchmarkevaluatingllmbased,
      title={DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems}, 
      author={Anni Zou and Wenhao Yu and Hongming Zhang and Kaixin Ma and Deng Cai and Zhuosheng Zhang and Hai Zhao and Dong Yu},
      year={2024},
      eprint={2407.10701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10701}, 
}
```