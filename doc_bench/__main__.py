"""CLI entry point for DocBench evaluation."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from .evaluate import run, MIN_IDX, MAX_IDX


def main():
    parser = argparse.ArgumentParser(
        description="Run DocBench evaluation on LLM-based document reading systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on first 5 documents (default)
  python -m doc_bench --model qwen2.5-7b --api-key YOUR_KEY --base-url http://localhost:1234/v1

  # Test specific documents
  python -m doc_bench --model gpt-4 --api-key YOUR_KEY --base-url https://api.openai.com/v1 --idx-list 10 20 30

  # Use full document text instead of chunks
  python -m doc_bench --model qwen2.5-7b --api-key YOUR_KEY --base-url http://localhost:1234/v1 --mode full_text

  # Enable agentic mode with multi-turn tool calling
  python -m doc_bench --model qwen2.5-7b --api-key YOUR_KEY --base-url http://localhost:1234/v1 --agentic

  # Test only multimodal questions
  python -m doc_bench --model qwen2.5-7b --api-key YOUR_KEY --base-url http://localhost:1234/v1 --qa-types multimodal-f multimodal-t
        """
    )

    # Required LLM connection arguments
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4, qwen2.5-7b)")
    parser.add_argument("--api-key", required=True, help="API key for the LLM service")
    parser.add_argument("--base-url", required=True, help="Base URL for the LLM API (e.g., http://localhost:1234/v1)")

    # LLM parameters
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    parser.add_argument("--context-window", type=int, default=32000, help="Model context window size (default: 32000)")

    # Processing mode
    parser.add_argument(
        "--mode",
        choices=["full_text", "chunks"],
        default="chunks",
        help='How to process documents: "full_text" (summarize entire doc) or "chunks" (semantic search) (default: chunks)'
    )
    parser.add_argument("--agentic", action="store_true", help="Enable multi-turn agent with tool calling instead of single-turn QA")

    # Chunking parameters (only used in chunks mode)
    parser.add_argument("--chunk-size", type=int, default=1024, help="Size of each text chunk in tokens (default: 1024)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between consecutive chunks in tokens (default: 200)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of most relevant chunks to retrieve (default: 5)")
    parser.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5", help="HuggingFace embedding model name (default: nomic-ai/nomic-embed-text-v1.5)")

    # Document selection
    parser.add_argument(
        "--idx-list",
        type=int,
        nargs="+",
        help=f"Specific document indices to evaluate (default: first 5). Range: {MIN_IDX}-{MAX_IDX}"
    )

    # Question type filtering
    qa_choices = ["text-only", "multimodal-f", "multimodal-t", "unanswerable", "meta-data", "una-web"]
    parser.add_argument(
        "--qa-types",
        nargs="+",
        choices=qa_choices,
        help=f"Types of questions to evaluate (default: text-only). Choices: {', '.join(qa_choices)}"
    )

    # Output
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file path (default: results_<timestamp>.json)")

    args = parser.parse_args()

    # Validate idx_list if provided
    if args.idx_list:
        for idx in args.idx_list:
            if idx < MIN_IDX or idx > MAX_IDX:
                parser.error(f"Index {idx} is out of range. Valid range: {MIN_IDX}-{MAX_IDX}")

    print(f"Starting DocBench evaluation...")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Mode: {args.mode}")
    print(f"  Agentic: {args.agentic}")
    print(f"  Documents: {args.idx_list if args.idx_list else 'first 5 (default)'}")
    print(f"  Question types: {args.qa_types if args.qa_types else 'text-only (default)'}")
    print()

    # Run evaluation
    results = run(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        context_window=args.context_window,
        mode=args.mode,
        agentic=args.agentic,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embed_model=args.embed_model,
        idx_list=args.idx_list,
        qa_types=args.qa_types
    )

    # Determine output path
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results_{timestamp}.json")

    # Save results
    output_data = {
        "metadata": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "context_window": args.context_window,
            "mode": args.mode,
            "agentic": args.agentic,
            "chunk_size": args.chunk_size if args.mode == "chunks" else None,
            "chunk_overlap": args.chunk_overlap if args.mode == "chunks" else None,
            "top_k": args.top_k if args.mode == "chunks" else None,
            "embed_model": args.embed_model if args.mode == "chunks" else None,
            "idx_list": args.idx_list if args.idx_list else list(range(5)),
            "qa_types": args.qa_types if args.qa_types else ["text-only"],
            "total_questions": len(results),
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    answered = sum(1 for r in results if "answer" in r)
    eval_passed = sum(1 for r in results if r.get("passing", False))
    avg_score = sum(r.get("score", 0) for r in results if "score" in r) / max(1, sum(1 for r in results if "score" in r))

    print(f"\nSummary:")
    print(f"  Total questions: {len(results)}")
    print(f"  Successfully answered: {answered}/{len(results)}")
    print(f"  Passing evaluations: {eval_passed}/{len(results)}")
    print(f"  Average score: {avg_score:.2f}")


if __name__ == "__main__":
    main()
