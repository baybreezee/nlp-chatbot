"""
Filter document indices based on PDF page count.
Filters out documents with too many pages to save on budget.
"""

from pathlib import Path
from pypdf import PdfReader
import json

DIR = Path(__file__).parent
DATA_FOLDER = DIR / "data"
MIN_IDX = 0
MAX_IDX = 228


def get_pdf_page_count(idx: int) -> int | None:
    """Get the number of pages in a PDF for a given document index."""
    folder = next(DATA_FOLDER.glob(f"{idx}"), None)
    if folder is None or not folder.is_dir():
        return None

    pdf_path = next(folder.glob("*.pdf"), None)
    if pdf_path is None:
        return None

    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF for idx {idx}: {e}")
        return None


def filter_indices(max_pages: int = 50) -> list[int]:
    """
    Return list of document indices with page count <= max_pages.

    Args:
        max_pages: Maximum number of pages to include (default 50)

    Returns:
        List of document indices that pass the filter
    """
    valid_indices = []
    page_counts = {}

    for idx in range(MIN_IDX, MAX_IDX + 1):
        count = get_pdf_page_count(idx)
        if count is not None:
            page_counts[idx] = count
            if count <= max_pages:
                valid_indices.append(idx)

    # Print summary statistics
    print(f"Total documents: {len(page_counts)}")
    print(f"Documents passing filter (<= {max_pages} pages): {len(valid_indices)}")
    print(f"Documents filtered out: {len(page_counts) - len(valid_indices)}")

    # Show some examples of filtered out documents
    long_docs = [(idx, count) for idx, count in page_counts.items() if count > max_pages]
    long_docs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 longest documents (filtered out):")
    for idx, count in long_docs[:10]:
        print(f"  Index {idx}: {count} pages")

    print(f"\nPage count distribution:")
    ranges = [(0, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, float('inf'))]
    for low, high in ranges:
        count = sum(1 for c in page_counts.values() if low <= c <= high)
        label = f"{low}-{int(high)}" if high != float('inf') else f"{low}+"
        print(f"  {label} pages: {count} documents")

    return valid_indices


if __name__ == "__main__":
    # Default threshold of 30 pages
    # Adjust this based on your budget constraints
    max_pages = 30

    indices = filter_indices(max_pages=max_pages)

    # Output as a Python list that can be copied into evaluate.py
    print(f"\n\nFiltered indices (max {max_pages} pages):")
    print(json.dumps(indices))
