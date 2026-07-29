from pathlib import Path

try:
    from .spectra_process_utils import msp2jsonl
except ImportError:
    from utils.spectra_process_utils import msp2jsonl

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    input_msp = repo_root / "data/own/single.msp"
    output_jsonl = repo_root / "data/own/single.jsonl"

    msp2jsonl(
        path_msp=input_msp,
        path_jsonl=output_jsonl,
    )