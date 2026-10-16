"""
InnerJev - 実際にダウンロード済みのInnerJev-4B-Full checkpointで推論する例
================================================================

model.py の `option_labels`/`build_prompt`/`Readout` はスタンドインLLMで
テストしたが、本ファイルはローカルにダウンロード済みの本物の公開チェックポイント
(`jylin001206/InnerJev-4B-Full`, Qwen3.5-4Bベースのfull-rank学習済みモデル) を
ロードして、全く同じロジックで実際に「choice/noul/score」の3種類の決定を
1回のforward passで読み出す。

公式コード対応: gh/InnerJev/innerjev/infer.py::DecisionEngine と同じロード/読み出し
手順 (AutoConfigでアーキテクチャ判定 -> Qwen3_5ForConditionalGeneration をbf16で
ロード -> backbone=model.model.language_model -> 最後の有効トークン位置から
label_idsだけのロジットを読み出す)。full_rank=True (LoRAではなくFull-rank
チェックポイントのため、本ファイルのReadoutもfull_rank=Trueで使う)。

============================================================
Shape Convention
============================================================
B : バッチサイズ (本デモは1問ずつ処理)
T : プロンプトのトークン長
K : 選択肢ラベル語彙サイズ (255)
"""

from __future__ import annotations

import torch

from model import Readout, build_prompt, option_labels

MODEL_PATH = (
    "/home/ym/.cache/huggingface/hub/models--jylin001206--InnerJev-4B-Full/"
    "snapshots/a8d77e5a3cebfc92feeae93821efa6a22739c9e8"
)


def load_real_model(model_path: str = MODEL_PATH):
    """公式 infer.py::DecisionEngine.__init__ と同じ手順で本物のチェックポイントをロードする。"""
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"  # Readoutの「最後の有効トークン位置」読み出しに必要 (公式と同一)

    config = AutoConfig.from_pretrained(model_path)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else Qwen3_5ForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cls.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


def example_usage() -> None:
    print("=== hf_inference_example.py Example Usage ===")
    print(f"モデル: jylin001206/InnerJev-4B-Full (ローカルキャッシュ: {MODEL_PATH})\n")

    tokenizer, model = load_real_model()
    labels = option_labels(tokenizer)
    print(f"単一トークン選択肢ラベル数: {len(labels)} (先頭5個: {labels[:5]})")

    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    # full_rank=True: Full-rank学習済みチェックポイントなので、出力埋め込み(LM head)の
    # 該当255行は「今ロードした重み」そのものを毎回読む (LoRAのような凍結スナップショットは不要)。
    readout = Readout(model, label_ids, full_rank=True)
    readout.label_ids = readout.label_ids.to(model.device)  # bufferはCPUで作られるためGPUモデルに合わせて移動

    rows = [
        {  # choice: 3択
            "type": "choice",
            "state": "A user is reviewing a pull request that renames a public API without a deprecation path.",
            "question": "What should the reviewer do?",
            "options": [
                {"key": "approve", "desc": "Approve the pull request as-is."},
                {"key": "request_changes",
                 "desc": "Request changes, asking the author to add a deprecation path for the renamed API."},
                {"key": "close", "desc": "Close the pull request without comment."},
            ],
        },
        {  # noul: 2値
            "type": "noul",
            "state": "The error message is: 'ConnectionResetError: [Errno 104] Connection reset by peer' "
                     "occurring intermittently on 2% of requests to a third-party payment API.",
            "question": "Is this error most likely caused by a bug in our own application code?",
            "options": [{"key": "yes", "desc": "Yes, our code has a bug."},
                        {"key": "no", "desc": "No, it is most likely a transient network/peer issue."}],
        },
        {  # score: 5段階
            "type": "score",
            "state": "A code review comment reads: 'This function is 300 lines long, has 8 levels of nested "
                     "if-statements, and no unit tests.'",
            "question": "On a scale of 0 (no concern) to 4 (severe concern), how concerned should a "
                        "reviewer be about maintainability?",
            "options": [{"key": str(i), "desc": f"Concern level {i}"} for i in range(5)],
        },
    ]

    for row in rows:
        prompt_text = build_prompt(tokenizer, row, labels)
        print(f"\n--- type={row['type']} ---")
        print(f"プロンプト末尾: ...{prompt_text[-80:]!r}")

        encoded = tokenizer([prompt_text], padding=True, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = readout(encoded["input_ids"], encoded["attention_mask"])  # (1, K=255)

        n = len(row["options"])
        probs = logits[0, :n].float().softmax(-1).cpu().tolist()
        keys = [o["key"] for o in row["options"]]
        for key, p in zip(keys, probs):
            print(f"  {key:16s}: {p:.4f}")

        if row["type"] == "noul":
            print(f"  => P(yes) = {probs[0]:.4f}  (公式の'noul'出力と同じ定義)")
        if row["type"] == "score":
            expected = sum(i * p for i, p in enumerate(probs))
            print(f"  => 期待スコア E[level] = {expected:.4f}  (公式の'score'出力と同じ定義)")


if __name__ == "__main__":
    example_usage()
