"""日本語ASR検証用データセットのローダー。各推論スクリプトから共通で使う。"""

from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_reazonspeech() -> list[tuple[str, Path, str]]:
    data_dir = DATA_ROOT / "reazonspeech_tiny"
    tsv_path = data_dir / "tiny.tsv"

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            filename, transcription = line.rstrip("\n").split("\t")
            samples.append((filename, data_dir / filename, transcription))
    return samples


def load_jsut() -> list[tuple[str, Path, str]]:
    subset_dir = DATA_ROOT / "jsut" / "jsut_ver1.1" / "basic5000"
    transcript_path = subset_dir / "transcript_utf8.txt"

    samples = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            sample_id, transcription = line.rstrip("\n").split(":", 1)
            samples.append((sample_id, subset_dir / "wav" / f"{sample_id}.wav", transcription))
    return samples


DATASETS = {
    "reazonspeech": load_reazonspeech,
    "jsut": load_jsut,
}
