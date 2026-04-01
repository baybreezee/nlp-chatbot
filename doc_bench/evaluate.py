from typing import Tuple, List, Dict
from pathlib import Path
import zipfile
import json

DIR = Path(__file__).parent
DATA_ZIP = DIR / "data.zip"
DATA_FOLDER = DIR / "data"
MIN_IDX = 0
MAX_IDX = 228


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


def create_questions(qa: List[Dict]) -> str:
    q_string = (
        "Based on the uploaded information, answer the following questions. "
        "You should answer all above questions line by line with numerical numbers. \n"
    )
    for i, d in enumerate(qa):
        q_string += f"{i+1}. {d['question']}\n"
    return q_string


# test: print questions for 66
if __name__ == "__main__":
    print(create_questions(read_qa_file(load_data_path(66)[1])))
