"""
OneStreamer - メインフロー (学習時フォワードパス全体)
================================================================

OneStreamer は Qwen3-VL-4B-Instruct をバックボーンとし、以下だけを追加する:
  1. 5つの制御トークン (control_tokens.py)
  2. 学習時のラベル構築における PSTL マスキング (pstl_loss.py)
  3. 推論時の Recent-N FIFO + PHCM キャプション記憶 (streaming_session.py)

バックボーン自体 (SigLIP系 Vision Encoder + Interleaved MRoPE + 36層 Qwen3 LLM) は
再実装しない。これは手を抜いているのではなく、公開チェックポイント
huggingface.co/MCG-NJU/OneStreamer-4B の `config.json` が
`"architectures": ["Qwen3VLForConditionalGeneration"]` のみで、カスタム
modeling_*.py を一切同梱していない (= 学習済み重みを読み込むだけで標準の
`transformers.Qwen3VLForConditionalGeneration` がそのまま使える) ことを
実際に確認した上での判断であり、バックボーンの完全な実装は姉妹フォルダ
`survey/Qwen3VL/main_flow.py` 等を参照してほしい。

公式コード: 学習コードは非公開。本ファイルの `OneStreamerForTraining` は、
論文 Sec 3 / Appendix A.1,A.3 の記述 (ラベル構築・損失) を実装した再構築である。

============================================================
Shape Convention
============================================================
B        : バッチサイズ (本ファイルの例では主に B=1 のシーケンス単位で説明)
T        : トークン列長 (視覚トークン + テキストトークン)
D_llm    : LLM 隠れ次元 (Qwen3-VL-4B: 2560, config.json で実測確認済み)
V        : 語彙サイズ (151936, うち 151669-151673 が本サーベイの5制御トークン)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from control_tokens import CONTROL_TOKENS, TASK_SPECS
from pstl_loss import ControlTokenEvent, build_labels_with_pstl, compute_loss, IGNORE_INDEX


# ============================================================
# 1. OneStreamer 本体: バックボーンのラッパー
# ============================================================

class OneStreamerForTraining(nn.Module):
    """Qwen3-VL バックボーンに PSTL ラベル構築を組み込んだ学習用ラッパー。

    実運用:
        from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        control_tokens.resize_embeddings_for_control_tokens(backbone, tokenizer)
        model = OneStreamerForTraining(backbone)

    `backbone` に要求するインターフェースは HuggingFace の CausalLM 標準:
        backbone(input_ids=..., attention_mask=..., pixel_values=..., labels=...)
            -> オブジェクト (.logits: (B,T,V), .loss: scalar or None)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.LongTensor,            # (B, T)
        attention_mask: torch.LongTensor,       # (B, T)
        base_labels: torch.LongTensor,          # (B, T) 通常のフルスーパービジョンラベル
        control_events: Sequence[Sequence[ControlTokenEvent]],  # バッチ内各サンプルの制御トークン列
        task: Sequence[str],                    # バッチ内各サンプルのタスク名
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rng=None,
    ) -> Dict[str, torch.Tensor]:
        """1バッチ分の forward (PSTL ラベル構築 -> バックボーン呼び出し)。

        ========================================
        処理フロー & Shape
        ========================================
        Step 1: 各サンプルについて PSTL を適用したラベルを構築
            base_labels[b]: (T,) -> labels[b]: (T,)
        Step 2: バックボーンへ forward (labels を渡すと内部で shift+CE を計算)
            pixel_values: (N_patches, C*P^2) [視覚トークンがある場合]
            -> logits: (B, T, V)
            -> loss: scalar (バックボーンが計算する「全ラベルに対する」CE loss)
        """
        B, T = input_ids.shape
        adjusted_labels = base_labels.clone()
        all_stats: List[Dict[str, int]] = []

        for b in range(B):
            spec = TASK_SPECS[task[b]]
            labels_b, stats = build_labels_with_pstl(
                base_labels[b], control_events[b], spec.anchors, rng
            )
            adjusted_labels[b] = labels_b
            all_stats.append(stats)

        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=adjusted_labels,
        )
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.backbone(**kwargs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        return {"loss": loss, "logits": logits, "labels": adjusted_labels, "pstl_stats": all_stats}


# ============================================================
# 2. 実デモ用の最小バックボーン (ネットワーク接続なしで動かすためのスタンドイン)
# ============================================================
#
# Qwen3-VL-4B-Instruct 本体 (Vision Encoder 24層 + LLM 36層, 隠れ次元2560,
# 語彙151936) をここで再実装すると、姉妹フォルダ survey/Qwen3VL の内容を
# そのまま重複させることになり、本サーベイが説明したい PSTL/PHCM のロジックが
# 埋もれてしまう。そのため、ここでは「本物のHF APIと同じ呼び出し方・同じ
# forward(labels=...) -> loss計算という契約」を持つ、隠れ次元・層数だけを
# 小さくした自前の nn.Module で代替する (重みはランダム初期化だが、
# forward/backward は実際に計算され、PSTL が本当に loss を変化させることを
# 以下の example_usage() で確認できる)。

class _TinyMultimodalBackboneForDemo(nn.Module):
    """shapeの契約だけ本物に合わせた極小バックボーン (デモ専用)。"""

    def __init__(self, vocab_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def resize_token_embeddings(self, new_vocab_size: int) -> nn.Embedding:
        old = self.embed
        new = nn.Embedding(new_vocab_size, old.embedding_dim)
        with torch.no_grad():
            new.weight[: old.weight.shape[0]] = old.weight
        self.embed = new
        self.lm_head = nn.Linear(old.embedding_dim, new_vocab_size, bias=False)
        return new

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.LongTensor,              # (B, T)
        attention_mask: torch.LongTensor,         # (B, T)
        labels: Optional[torch.LongTensor] = None,  # (B, T)
        **_unused,
    ) -> Dict[str, torch.Tensor]:
        x = self.embed(input_ids)  # (B, T, D)
        T_len = x.shape[1]
        # bool causal mask (True=見てはいけない位置)。key_padding_maskもbool型に
        # 揃えることで torch の型不一致警告を避ける。
        causal_mask = torch.triu(
            torch.ones(T_len, T_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        key_padding_mask = (attention_mask == 0)  # (B, T) True=パディング
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if labels is not None:
            loss = compute_loss(logits, labels)
        return {"logits": logits, "loss": loss}


# ============================================================
# 使用例: 合成 PHCM シーケンスで PSTL が実際に loss を変えることを確認する
# ============================================================

def example_usage() -> None:
    print("=== main_flow.py Example Usage ===\n")

    torch.manual_seed(0)

    # ------------------------------------------------------------
    # 1. 語彙の設計: 通常トークンID 0..49, 制御トークンID 50..54
    # ------------------------------------------------------------
    BASE_VOCAB = 50
    control_ids = {tok: BASE_VOCAB + i for i, tok in enumerate(CONTROL_TOKENS)}
    V = BASE_VOCAB + len(CONTROL_TOKENS)
    print(f"語彙サイズ V = {V} (通常トークン{BASE_VOCAB}語 + 制御トークン{len(CONTROL_TOKENS)}語)")

    # ------------------------------------------------------------
    # 2. 合成 PHCM 学習シーケンスを1本構築
    #    構造: [Silence]*3 </Observe> caption(5tok) [Silence]*3 </Observe> caption(5tok)
    #          [Silence]*3 </Summary> caption(5tok)
    #    (画像トークンは省略し、テキストのみの疑似シーケンスとする:
    #     PSTL のロジック自体はテキスト列上のインデックス操作なので、
    #     視覚トークンの有無は本質に影響しない。視覚統合は streaming_session.py /
    #     finetune_example.py 側で real に扱う)
    # ------------------------------------------------------------
    def rand_text_tokens(n: int) -> List[int]:
        return torch.randint(0, BASE_VOCAB, (n,)).tolist()

    seq: List[int] = []
    labels_full: List[int] = []
    events: List[ControlTokenEvent] = []

    def add_silence_block(k: int) -> None:
        for _ in range(k):
            pos = len(seq)
            seq.append(control_ids["</Silence>"])
            labels_full.append(control_ids["</Silence>"])  # 通常は常に教師信号あり
            events.append(ControlTokenEvent(position=pos, state="</Silence>"))

    def add_record(tok: str, caption_len: int) -> None:
        pos = len(seq)
        seq.append(control_ids[tok])
        labels_full.append(control_ids[tok])
        events.append(ControlTokenEvent(position=pos, state=tok))
        caption = rand_text_tokens(caption_len)
        seq.extend(caption)
        labels_full.extend(caption)  # キャプション本文は常に全トークン教師信号 (PSTL対象外)

    add_silence_block(3)
    add_record("</Observe>", 5)
    add_silence_block(3)
    add_record("</Observe>", 5)
    add_silence_block(3)
    add_record("</Summary>", 5)

    input_ids = torch.tensor([seq], dtype=torch.long)       # (1, T)
    attention_mask = torch.ones_like(input_ids)              # (1, T)
    base_labels = torch.tensor([labels_full], dtype=torch.long)  # (1, T)
    T = input_ids.shape[1]
    print(f"シーケンス長 T = {T}, 制御トークン数 = {len(events)}")

    # ------------------------------------------------------------
    # 3. モデル構築 (デモ用の極小バックボーン)
    # ------------------------------------------------------------
    backbone = _TinyMultimodalBackboneForDemo(vocab_size=V, hidden_size=32, num_layers=2)
    model = OneStreamerForTraining(backbone)

    out_pstl = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        base_labels=base_labels,
        control_events=[events],
        task=["phcm"],
    )
    print(f"\nPSTL適用後のloss: {out_pstl['loss'].item():.4f}")
    print(f"PSTL統計 (1サンプル目): {out_pstl['pstl_stats'][0]}")

    # 比較用: PSTLを使わず全制御トークンを教師信号に使った場合
    out_dense = backbone(input_ids=input_ids, attention_mask=attention_mask, labels=base_labels)
    print(f"PSTLなし(全状態トークンsupervise)のloss: {out_dense['loss'].item():.4f}")

    # 逆伝播も実際に通ることを確認 (勾配がNoneでない)
    out_pstl["loss"].backward()
    grad_norm = sum(
        p.grad.norm().item() for p in backbone.parameters() if p.grad is not None
    )
    print(f"\nbackward後の勾配ノルム合計: {grad_norm:.4f} (0でなければ実際に学習可能)")

    print(f"\n出力 logits shape: {out_pstl['logits'].shape}  (B={1}, T={T}, V={V})")


if __name__ == "__main__":
    example_usage()
