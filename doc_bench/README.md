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
| `--extra-body` | JSON string for OpenAI extra_body parameters | None |

#### Evaluation (Judge) LLM

By default, the same model is used for both generation and evaluation. Use these to specify a different model for judging answer correctness:

| Argument | Description | Default |
|----------|-------------|---------|
| `--eval-model` | Model name for evaluation/judging | Same as `--model` |
| `--eval-api-key` | API key for evaluation model | Same as `--api-key` |
| `--eval-base-url` | Base URL for evaluation model | Same as `--base-url` |

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

# Use extra_body for provider-specific parameters
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL \
    --extra-body '{"enable_thinking": true}'

# Use a stronger model (e.g., GPT-4) for evaluation while testing a smaller model
python -m doc_bench --model qwen2.5-7b --api-key KEY --base-url URL \
    --eval-model gpt-4 --eval-api-key GPT4_KEY --eval-base-url https://api.openai.com/v1
```

### Understanding Results

The output JSON contains:

```json
{
  "metadata": {
    "model": "qwen2.5-7b",
    "mode": "chunks",
    "eval_model": "gpt-4",
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

## Experiment

### PDFs and Questions

```
Error reading PDF for idx 72: cryptography>=3.1 is required for AES algorithm
Error reading PDF for idx 73: cryptography>=3.1 is required for AES algorithm
Total documents: 227
Documents passing filter (<= 12 pages): 101
Documents filtered out: 126

Top 10 longest documents (filtered out):
  Index 53: 844 pages
  Index 83: 564 pages
  Index 64: 449 pages
  Index 66: 382 pages
  Index 81: 339 pages
  Index 120: 330 pages
  Index 110: 318 pages
  Index 63: 316 pages
  Index 62: 312 pages
  Index 54: 293 pages

Page count distribution:
  0-10 pages: 83 documents
  11-20 pages: 51 documents
  21-30 pages: 15 documents
  31-50 pages: 9 documents
  51-100 pages: 25 documents
  101+ pages: 44 documents
```

- Remove 142, 187, 203, because it triggers "Sensitive Content" API error 😅.
- Questions types: `text-only` and `unanswerable`.
- Number of questions: 202.

### Models

- Evaluated LLM 1: [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) (non-thinking)
- Evaluated LLM 2: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (non-thinking)
- Judge LLM: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- Embedding Model: [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### Results

#### Baseline

**PDFs**

```
0 1 2 3 4 8 9 10 12 13 14 15 16 17 18 19 20 21 22 24 25 26 28 29 31 32 34 35 36 38 40 41 44 45 46 47 48 85 89 97 103 123 125 141 145 146 148 152 155 175 179 180 181 182 183 184 185 186 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228
```

**Params**

- `mode="chunks"`
- `agentic=False`
- `chunk_size=1024`
- `chunk_overlap=200`
- `top_k=2`

**DeepSeek-V3.2**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 148 | 131 | 4.66 |
| unanswerable | 54 | 40 | 4.24 |
| all | 202 | 171 | 4.54 |

**Qwen3.5-27B**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 148 | 129 | 4.62 |
| unanswerable | 54 | 47 | 4.67 |
| all | 202 | 176 | 4.63 |

#### Improvement

**PDFs**

```
0 1 3 10 12 13 16 17 21 28 34 35 36 41 44 45 46 47 85 89 97 123 145 146 148 155 175 192 193 194 196 200 204 228
```

**Params**

- `mode="chunks"`
- `agentic=True`
- `chunk_size=2048`
- `chunk_overlap=200`
- `top_k=5`

**Baseline (DeepSeek-V3.2)**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 56 | 39 | 4.09 |
| unanswerable | 20 | 6 | 2.95 |
| all | 76 | 45 | 3.79 |

**Improvement (DeepSeek-V3.2)**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 56 | 44 | 4.35 |
| unanswerable | 20 | 12 | 3.75 |
| all | 76 | 56 | 4.19 |

**Baseline (Qwen3.5-27B)**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 56 | 37 | 3.99 |
| unanswerable | 20 | 13 | 4.1 |
| all | 76 | 50 | 4.02 |

**Improvement (Qwen3.5-27B)**

| type | count | passing | avg score |
| --- | --- | --- | --- |
| text-only | 56 | 43 | 4.34 |
| unanswerable | 20 | 13 | 4.03 |
| all | 76 | 56 | 4.26 |

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