"""
Context Language Models (CLMs) - Prefix-Reuse FLOPs (効率指標)
================================================================

論文が全ての実験で報告する計算コスト指標 "prefix-reuse FLOPs" を実装する。
これは「標準的な prefix-cache を使うサーバーが、1つのエージェント軌跡を処理する
のに実際に行う計算量」の理論値であり、CLM が任意の位置でコンテキストを編集する
と、その編集点以降のトークンが毎回「再プリフィル」される点を正しく課金する。

論文: Context Language Models (arXiv:2609.37725)
      - Eq.3 (Sec 3.1): FLOPs_prefix-reuse = FLOPs_prefill(unmatched suffix) + FLOPs_decode(generated)
      - Appendix B "Prefix-Reuse FLOPs Computation" の全ての式 (C_token, C_attn, F_t)
公式コード: gh/context-language-models/clm/clm_harness/flops_metrics/kv_cache_flops.py
      (vLLM/SGLang の実運用ログを食わせる本番用実装)。本ファイルは Appendix の
      数式をそのまま、ログ無しでも動くようにシンプルな trie ベースの prefix-cache
      シミュレーションとして再実装したもの (アルゴリズムは公式と同一、
      I/O 周り(usage.json 等の優先ソース切り替え)は簡略化している)。

============================================================
直感
============================================================
ターン t でプロンプトが P_t トークン、生成が G_t トークン。サーバーは「これより
前のどこかのターンのプロンプトと共有する最長の先頭一致」を再利用し、残り
U_t = P_t - R_t トークンだけを新たにプリフィルする。
  - 末尾に追記するだけのターン (CLMが編集しない) : R_t はほぼ P_{t-1} (ほぼ無料)
  - コンテキストの途中を編集するターン: 編集点より後が全て不一致になり、
    U_t が急増する (再プリフィルのペナルティ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

# ============================================================
# 1. モデル固有の定数 (論文 Appendix B の Qwen3.6-27B 例をそのまま採用)
# ============================================================

# C_token = 6*L*d*d_ff (MLP) + L_attn*[2d(2h_q*d_h+2h_kv*d_h)+2h_q*d_h*d] (full-attn projections)
#           + L_lin*[2d(2h_k*d_k+2h_v*d_v+2h_v)+2h_v*d_v*d] (Gated DeltaNet projections)
MODEL_GEOMETRY: Dict[str, Dict[str, float]] = {
    "qwen3.6-27b": dict(
        L=64, d=5120, d_ff=17408,
        L_attn=16, h_q=24, h_kv=4, d_h=256,          # full-attention layers
        L_lin=48, h_k=16, h_v=48, d_k=128, d_v=128,   # Gated DeltaNet layers
    ),
}


def c_token_and_c_attn(geom: Dict[str, float]) -> Tuple[float, float]:
    """1トークンあたりの線形射影コスト C_token と、Query-Keyペアあたりの
    attentionコスト C_attn を論文 Appendix B の式通りに計算する。

    ========================================
    式 (論文 Appendix B)
    ========================================
    C_token = 6*L*d*d_ff
             + L_attn*[2*d*(2*h_q*d_h + 2*h_kv*d_h) + 2*h_q*d_h*d]
             + L_lin*[2*d*(2*h_k*d_k + 2*h_v*d_v + 2*h_v) + 2*h_v*d_v*d]
    C_attn  = 4 * L_attn * h_q * d_h
    """
    L, d, d_ff = geom["L"], geom["d"], geom["d_ff"]
    L_attn, h_q, h_kv, d_h = geom["L_attn"], geom["h_q"], geom["h_kv"], geom["d_h"]
    L_lin, h_k, h_v, d_k, d_v = geom["L_lin"], geom["h_k"], geom["h_v"], geom["d_k"], geom["d_v"]

    mlp = 6.0 * L * d * d_ff
    full_attn_proj = L_attn * (2.0 * d * (2 * h_q * d_h + 2 * h_kv * d_h) + 2.0 * h_q * d_h * d)
    gdn_proj = L_lin * (2.0 * d * (2 * h_k * d_k + 2 * h_v * d_v + 2 * h_v) + 2.0 * h_v * d_v * d)
    c_token = mlp + full_attn_proj + gdn_proj
    c_attn = 4.0 * L_attn * h_q * d_h
    return c_token, c_attn


# ============================================================
# 2. prefix-cache trie シミュレーション (公式 accumulate_prefill を再実装)
# ============================================================

Units = List[Tuple[Hashable, int]]


def units_from_prompt(prompt_text: str) -> Units:
    """1ターンの「フルプロンプト文字列」を語単位 (疑似トークン) のユニット列に変換する。

    実運用ではトークンID単位でブロック化する (vLLM/SGLangのKVキャッシュブロック)。
    ここでは外部トークナイザへの依存を避けるため、空白区切りの単語を1ユニット
    (重み1) として扱う。重要なのは「ターン間で同じ接頭辞が続く限りキャッシュが
    当たる」という trie の性質であり、単位の粒度そのものは結果の定性的な傾向を
    変えない。
    """
    words = prompt_text.split()
    return [(w, 1) for w in words] if words else []


def accumulate_prefill(turns: List[Units]) -> Tuple[int, int, List[int]]:
    """ターンを順番に trie (prefix木) に通し、naive / cache-aware なプリフィル
    トークン数を積算する。

    ========================================
    出力
    ========================================
    (naive_prefill_tokens, cache_aware_prefill_tokens, per_turn_seq_lengths)
        per_turn_seq_lengths[t] = ターン t のプロンプト長 P_t (attention項で使う)
    """
    naive_total = 0
    cache_aware_total = 0
    seq_lengths: List[int] = []
    root: Dict[Hashable, Any] = {}

    for units in turns:
        total = sum(w for _, w in units)
        naive_total += total
        seq_lengths.append(total)

        node = root
        matched = 0
        for key, w in units:
            nxt = node.get(key)
            if nxt is None:
                break
            node = nxt
            matched += w
        cache_aware_total += total - matched

        node = root
        for key, _ in units:
            node = node.setdefault(key, {})

    return naive_total, cache_aware_total, seq_lengths


# ============================================================
# 3. ターンごとの FLOPs (論文 Appendix B Eq. for F_t)
# ============================================================

@dataclass
class TurnFlops:
    prefill_new_tokens: int   # U_t = P_t - R_t
    attn_prefill_pairs: float  # (P_t^2 - R_t^2) / 2
    decode_pairs: float        # G_t*P_t + G_t^2/2


def per_turn_flops(
    seq_lengths: List[int],
    reused_lengths: List[int],
    gen_lengths: List[int],
) -> List[TurnFlops]:
    """各ターンの (新規プリフィル量, attentionペア数) を計算する。

    ========================================
    式 (論文 Appendix B)
    ========================================
    F_t = C_token*(U_t + G_t) + C_attn*[ (P_t^2 - R_t^2)/2 + G_t*P_t + G_t^2/2 ]
    のうち、角括弧の中身 (attentionペア数) と U_t をここで用意する。
    """
    out = []
    for P_t, R_t, G_t in zip(seq_lengths, reused_lengths, gen_lengths):
        U_t = max(P_t - R_t, 0)
        attn_prefill = max(P_t * P_t - R_t * R_t, 0) / 2.0
        decode = G_t * P_t + (G_t * G_t) / 2.0
        out.append(TurnFlops(prefill_new_tokens=U_t, attn_prefill_pairs=attn_prefill, decode_pairs=decode))
    return out


def trajectory_flops(
    turns: List[TurnFlops],
    c_token: float,
    c_attn: float,
) -> Dict[str, float]:
    """1軌跡分の合計 FLOPs = sum_t F_t (論文 Appendix B)。

    ========================================
    出力
    ========================================
    {"flops_total": float, "flops_linear": float, "flops_attn": float,
     "prefill_new_tokens_total": int}
    """
    linear = 0.0
    attn = 0.0
    prefill_new = 0

    for t in turns:
        # F_t の線形項: C_token*(U_t + G_t) をここでは U_t だけ加算し、
        # G_t (生成量) は呼び出し側が別途加算する設計にしてもよいが、
        # per_turn_flops は G_t を decode_pairs に折り込んでいるため、ここでは
        # U_t 分の線形コストのみ加算する (生成トークンの線形コストは
        # trajectory_flops_from_steps 側で明示的に加える)。
        linear += c_token * t.prefill_new_tokens
        attn += c_attn * (t.attn_prefill_pairs + t.decode_pairs)
        prefill_new += t.prefill_new_tokens

    return {
        "flops_linear": linear,
        "flops_attn": attn,
        "flops_total": linear + attn,
        "prefill_new_tokens_total": prefill_new,
    }


# ============================================================
# 4. 軌跡全体の高レベルAPI: メッセージ列の履歴から FLOPs を計算する
# ============================================================

def compute_trajectory_flops(
    prompt_snapshots: List[str],
    generated_lengths: List[int],
    model_key: str = "qwen3.6-27b",
) -> Dict[str, Any]:
    """軌跡全体 (ターンごとのフルプロンプト) から prefix-reuse FLOPs を計算する。

    ========================================
    入力
    ========================================
    prompt_snapshots: List[str]
        ターン t で実際にサーバーへ送られた「フルプロンプト全文」(system+全履歴)。
        CLM が途中を編集すると、ターン t の文字列はターン t-1 の文字列と
        先頭の共通する語数のところで枝分かれする (末尾に追記しただけなら
        ターン t-1 の全文がターン t の文字列の接頭辞として完全に一致する)。
    generated_lengths: List[int]
        各ターンで生成されたトークン数 G_t。
    model_key: str
        MODEL_GEOMETRY のキー。

    ========================================
    出力
    ========================================
    {"naive_prefill_tokens", "cache_aware_prefill_tokens", "cache_saved_frac",
     "flops_total", "flops_linear", "flops_attn", "flops_generation_linear"}
    """
    geom = MODEL_GEOMETRY[model_key]
    c_token, c_attn = c_token_and_c_attn(geom)

    turns_units = [units_from_prompt(p) for p in prompt_snapshots]
    naive_total, cache_aware_total, seq_lengths = accumulate_prefill(turns_units)

    # R_t (再利用された長さ) = P_t - U_t。U_t はターンごとの trie マッチから逆算する。
    # ここでは per_turn の cache-aware prefill を再度計算してR_tを得る。
    reused_lengths: List[int] = []
    root: Dict[Hashable, Any] = {}
    for units in turns_units:
        node = root
        matched = 0
        for key, w in units:
            nxt = node.get(key)
            if nxt is None:
                break
            node = nxt
            matched += w
        reused_lengths.append(matched)
        node = root
        for key, _ in units:
            node = node.setdefault(key, {})

    turns = per_turn_flops(seq_lengths, reused_lengths, generated_lengths)
    result = trajectory_flops(turns, c_token, c_attn)

    # 生成トークン自体の線形コスト (C_token * G_t) を加算する。
    gen_linear = c_token * sum(generated_lengths)
    result["flops_linear"] += gen_linear
    result["flops_total"] += gen_linear
    result["flops_generation_linear"] = gen_linear

    result["naive_prefill_tokens"] = naive_total
    result["cache_aware_prefill_tokens"] = cache_aware_total
    result["cache_saved_frac"] = (
        (naive_total - cache_aware_total) / naive_total if naive_total else 0.0
    )
    return result


# ============================================================
# 使用例: append-only / 途中編集 / 先頭編集のFLOPsを比較する
# ============================================================

def example_usage() -> None:
    """論文 Appendix B Fig. "flops_cache_bars" の説明と同じ構造の比較例。

    P_t=20000語, G_t=500語 相当のターンを模した軌跡で、
    (a) 追記のみ、(b) 途中編集 (再利用10000語)、(c) 先頭編集 (再利用0語)
    の3パターンのFLOPsを実際に計算して比較する。
    """
    print("=== prefix_reuse_flops.py Example Usage ===\n")

    def make_turn(n_words: int, tag: str) -> str:
        # 語ごとに一意な番号を振ることで、trieが「位置ごとの一致」を正しく追えるようにする。
        return " ".join(f"{tag}{i}" for i in range(n_words))

    # --- (a) append-only: ターン2はターン1の全文をそのまま含み、後ろに追記 ---
    t1 = make_turn(20000, "A")
    t2_append = t1 + " " + make_turn(500, "A_more")

    flops_append = compute_trajectory_flops(
        prompt_snapshots=[t1, t2_append], generated_lengths=[500, 500],
    )

    # --- (b) 途中編集: 先頭10000語だけ共有、残り10000語は別の内容に置き換え ---
    shared = make_turn(10000, "A")
    t2_mid_edit = shared + " " + make_turn(10000, "B_edited")
    flops_mid = compute_trajectory_flops(
        prompt_snapshots=[t1, t2_mid_edit], generated_lengths=[500, 500],
    )

    # --- (c) 先頭編集: 何も共有しない (再利用0語) ---
    t2_head_edit = make_turn(20000, "C_rewritten")
    flops_head = compute_trajectory_flops(
        prompt_snapshots=[t1, t2_head_edit], generated_lengths=[500, 500],
    )

    for name, flops in [("追記のみ", flops_append), ("途中編集", flops_mid), ("先頭編集", flops_head)]:
        print(f"[{name}]")
        print(f"  naive_prefill_tokens:       {flops['naive_prefill_tokens']:>10,}")
        print(f"  cache_aware_prefill_tokens: {flops['cache_aware_prefill_tokens']:>10,}")
        print(f"  cache_saved_frac:           {flops['cache_saved_frac']:.1%}")
        print(f"  flops_total:                {flops['flops_total']:.3e}")
        print()

    ratio = flops_head["flops_total"] / flops_append["flops_total"]
    print(f"=> 先頭編集は追記のみの {ratio:.2f} 倍の FLOPs "
          "(定性的な傾向は論文 Appendix B の例 [同条件で約7.7倍] と同じ: "
          "編集点が前にあるほど再利用できる接頭辞が減り、FLOPsが増える。"
          "絶対比は本例のターン構成が異なるため論文の数値とは一致しない)")


if __name__ == "__main__":
    example_usage()
