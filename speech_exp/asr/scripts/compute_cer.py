"""推論結果CSV(id,gt,pred,inference_time_sec)からCER(文字誤り率)を算出する。

使い方:
    uv run python scripts/compute_cer.py results/parakeet_ja_jsut_results.csv
    uv run python scripts/compute_cer.py results/parakeet_ja_jsut_results.csv --reading
"""

import argparse
import re
import unicodedata

import jiwer
import pandas as pd
from janome.tokenizer import Tokenizer

# モデルカード記載の評価方法に合わせ、句読点・記号を除去した正規化版CERも算出する。
# (数字→単語変換(num2words)までは厳密には再現していない点に注意)
_PUNCT_RE = re.compile(r"[、。!!??,.・「」『』()()【】\[\]…\s]")

_tokenizer = Tokenizer()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _PUNCT_RE.sub("", text)


def to_reading(text: str) -> str:
    """漢字/かな表記ゆれを吸収するため、カタカナ読みに変換する。
    (例: 「歪めた」も「ゆがめた」も同じ「ユガメタ」になる)
    """
    return "".join(
        token.reading if token.reading != "*" else token.surface
        for token in _tokenizer.tokenize(text)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument(
        "--reading",
        action="store_true",
        help="漢字/かな表記ゆれを読み(カタカナ)に正規化してからCERを算出する",
    )
    args = parser.parse_args()

    # predが空文字の行はpandasにNaN(float)として読まれ、jiwerがエラーになるため空文字に戻す。
    df = pd.read_csv(args.csv_path, dtype={"gt": str, "pred": str}).fillna({"gt": "", "pred": ""})

    raw_cer = jiwer.cer(df["gt"].tolist(), df["pred"].tolist())

    norm_gt = df["gt"].map(normalize)
    norm_pred = df["pred"].map(normalize)
    norm_cer = jiwer.cer(norm_gt.tolist(), norm_pred.tolist())

    print(f"samples       : {len(df)}")
    print(f"mean inference_time_sec: {df['inference_time_sec'].mean():.4f}")
    print(f"raw CER       : {raw_cer * 100:.2f}%")
    print(f"normalized CER: {norm_cer * 100:.2f}%  (句読点等除去後、モデルカードの評価方法に近似)")

    if args.reading:
        reading_gt = norm_gt.map(to_reading)
        reading_pred = norm_pred.map(to_reading)
        reading_cer = jiwer.cer(reading_gt.tolist(), reading_pred.tolist())
        print(f"reading CER   : {reading_cer * 100:.2f}%  (句読点除去+読み正規化後、漢字/かな表記ゆれを無視)")


if __name__ == "__main__":
    main()
