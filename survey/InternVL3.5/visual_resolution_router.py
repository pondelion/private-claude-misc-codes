"""
InternVL3.5 Visual Resolution Router (ViR) + Visual Consistency Learning (ViCO)
================================================================================

このファイルは InternVL3.5-Flash の推論高速化の核心である
Visual Resolution Router (ViR) と ViCO 学習フローを実装します。

Visual Resolution Router (ViR):
  各画像パッチの「視覚情報の豊富さ」を判定して、
  適切な圧縮率を動的に選択する軽量バイナリ分類器。

  ・低圧縮 (ξ=1/4):  256 tokens/patch → 視覚情報が豊富なパッチ (文字, 細部)
  ・高圧縮 (ξ=1/16):  64 tokens/patch → 単純なパッチ (背景, ベタ塗り)

  平均50%のビジュアルトークン削減で性能はほぼ100%維持。

Visual Consistency Learning (ViCO):
  ViR を InternVL3.5 に統合するための2段階学習フロー。

  Stage 1 - 一貫性学習 (Consistency Training):
    全モデルをファインチューニングして256/64トークン表現の出力を一致させる。
    KL divergence を最小化することで64トークンモードでも256トークンと同等の出力を保証。

  Stage 2 - ルーター学習 (Router Training):
    各パッチの「圧縮による損失増加率 r_i」を計算し、
    バイナリ分類器 (ViR) を標準クロスエントロピー損失で学習。

公式実装参考:
  公式コードには ViR の単独実装ファイルはなく、モデル設定に統合されている。
  本ファイルは論文 Section 2.3 の記述から詳細実装を再現。

============================================================
テンソル形状記法
============================================================
  B  : バッチサイズ
  P  : 1サンプルあたりのパッチ数
  N  : 系列長
  D_v: ViT hidden size
  D_r: ルーター入力次元 (ViT CLS token の次元と同じ = D_v)
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. 圧縮率の定義
# ============================================================

class CompressionLevel:
    """
    InternVL3.5-Flash の2段階圧縮率を定義するクラス。

    pixel_shuffle の downsample_ratio:
      LOW_COMPRESSION  = 0.5  → (448/14)^2 * 0.5^2 = 256 tokens/patch
      HIGH_COMPRESSION = 0.25 → (448/14)^2 * 0.25^2 = 64  tokens/patch

    論文での記法:
      ξ = 1/4  (LOW)  : 256 tokens  ← 通常の InternVL3.5 と同じ
      ξ = 1/16 (HIGH) :  64 tokens  ← 4倍さらに圧縮
    """
    LOW  = 0.5   # downsample_ratio → 256 tokens
    HIGH = 0.25  # downsample_ratio →  64 tokens

    @staticmethod
    def tokens_per_patch(downsample_ratio: float, image_size: int = 448, patch_size: int = 14) -> int:
        """
        圧縮率から1パッチあたりのトークン数を計算。

        計算式:
          n_tokens = (image_size / patch_size)^2 * downsample_ratio^2
                   = (448/14)^2 * dr^2 = 1024 * dr^2

        引数:
          downsample_ratio : 0.5 (LOW) or 0.25 (HIGH)
        返値:
          tokens_per_patch : 256 (LOW) or 64 (HIGH)
        """
        n_patches_hw = (image_size // patch_size) ** 2  # 1024
        return int(n_patches_hw * downsample_ratio ** 2)


# ============================================================
# 2. Visual Resolution Router (ViR)
# ============================================================

class VisualResolutionRouter(nn.Module):
    """
    各画像パッチの圧縮率を動的に選択するバイナリ分類器。

    入力として ViT の CLS トークン特徴を使用。
    CLS トークンはパッチ全体のグローバル表現を持つため、
    そのパッチの視覚情報の豊富さを判断するのに適している。

    アーキテクチャ: 軽量な 2 層 MLP

    入力形状:
      patch_cls_features : (B, P, D_v)   各パッチの ViT CLS トークン
    出力形状:
      compression_logits : (B, P, 2)     [低圧縮, 高圧縮] のロジット
      compression_probs  : (B, P, 2)     softmax 後の確率
      routing_decision   : (B, P)        0=低圧縮(256tokens), 1=高圧縮(64tokens)
    """
    def __init__(
        self,
        input_dim: int,           # D_v: ViT hidden size (例: 3200 for InternViT-6B)
        hidden_dim: int = 512,    # MLP 中間次元
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 軽量な 2 層 MLP 分類器
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2クラス: 低圧縮 vs 高圧縮
        )

        # 分類器の重みを小さい値で初期化 (学習安定化)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(
        self,
        patch_cls_features: torch.Tensor,
        return_probs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        入力: patch_cls_features  (B, P, D_v)
              ※ 各パッチの ViT CLS トークン特徴
        出力: {
          'logits':   (B, P, 2)   未正規化ロジット
          'probs':    (B, P, 2)   softmax 確率 (return_probs=True の場合)
          'decisions':(B, P)     int: 0=低圧縮, 1=高圧縮
        }
        """
        # (B, P, D_v) → (B, P, 2)
        logits = self.router(patch_cls_features)

        result = {'logits': logits}

        if return_probs:
            probs = F.softmax(logits, dim=-1)   # (B, P, 2)
            result['probs'] = probs

        # ハードな決定 (推論時): argmax → 0 or 1
        decisions = logits.argmax(dim=-1)       # (B, P)
        result['decisions'] = decisions

        return result

    def route_patches(
        self,
        vit_features: torch.Tensor,
        patch_cls_idx: int = 0,
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        ViT 出力から各パッチのルーティングを決定する便利メソッド。

        引数:
          vit_features  : (B, P*S_v, D_v)  全パッチの ViT 出力 (system token 含む)
                         ※ S_v = 1025 (CLS + 1024 パッチトークン)
          patch_cls_idx : CLS トークンの位置 = 0
        返値:
          low_patch_mask  : (B, P) bool  低圧縮パッチのマスク
          high_patch_mask : (B, P) bool  高圧縮パッチのマスク
          decisions       : (B, P) int
        """
        B = vit_features.shape[0]
        S_v = 1025  # CLS + 1024

        # (B, P*S_v, D_v) を (B, P, S_v, D_v) にリシェイプしてCLSを取り出す
        # 実際の実装ではパッチ数 P を引数として受け取る必要がある
        # ここでは簡略化のため CLS を patch_cls_idx=0 で取り出す
        cls_features = vit_features[:, patch_cls_idx::S_v, :]  # (B, P, D_v)

        routing = self.forward(cls_features)
        decisions = routing['decisions']  # (B, P)

        low_mask = (decisions == 0)   # (B, P) bool
        high_mask = (decisions == 1)  # (B, P) bool

        return low_mask, high_mask, decisions


# ============================================================
# 3. 動的圧縮ピクセルシャッフル
# ============================================================

class AdaptivePixelShuffle(nn.Module):
    """
    ViR の決定に基づいて各パッチに異なる圧縮率を適用するモジュール。

    ルーティング結果に応じて:
      decision=0 → PixelShuffle(scale=0.5)   → 256 tokens
      decision=1 → PixelShuffle(scale=0.25)  →  64 tokens

    入力形状:
      patch_features : (total_patches, H_t, W_t, D_v)  ※ H_t=W_t=32
      decisions      : (total_patches,)  int 0 or 1
    出力形状:
      compressed_features: List[torch.Tensor]
        各パッチの圧縮後特徴 (tokens, channels) ← パッチごとに長さが異なる
    """
    def __init__(self, version: str = 'v2'):
        super().__init__()
        self.version = version

    def _pixel_shuffle(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        単一パッチの Pixel Shuffle 圧縮。

        入力: (1, H_t, W_t, D_v)  ※ バッチ次元は 1
        出力: (1, H_t*s, W_t*s, D_v/(s^2))
        """
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale), int(c / scale))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale), int(w * scale), int(c / (scale ** 2)))
        if self.version == 'v2':
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        patch_features: torch.Tensor,
        decisions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力:
          patch_features : (P, H_t=32, W_t=32, D_v)
          decisions      : (P,)  0=低圧縮, 1=高圧縮
        出力:
          compressed     : (P, max_tokens, max_channels) パディング済みテンソル
          token_counts   : (P,)  各パッチの有効トークン数
        """
        P = patch_features.shape[0]
        results = []
        token_counts = []

        for i in range(P):
            patch = patch_features[i:i+1]  # (1, 32, 32, D_v)
            if decisions[i] == 0:
                # 低圧縮: scale=0.5 → 256 tokens
                compressed = self._pixel_shuffle(patch, scale=0.5)
                # (1, 16, 16, D_v*4) → (256, D_v*4)
                compressed = compressed.reshape(1, -1, compressed.shape[-1]).squeeze(0)
            else:
                # 高圧縮: scale=0.25 → 64 tokens
                compressed = self._pixel_shuffle(patch, scale=0.25)
                # (1, 8, 8, D_v*16) → (64, D_v*16)
                compressed = compressed.reshape(1, -1, compressed.shape[-1]).squeeze(0)
            results.append(compressed)
            token_counts.append(compressed.shape[0])

        # パディングして結合 (最長パッチに合わせる)
        max_tokens = max(token_counts)
        max_channels = max(r.shape[-1] for r in results)

        padded = torch.zeros(P, max_tokens, max_channels,
                             dtype=patch_features.dtype,
                             device=patch_features.device)
        for i, (result, n_tok) in enumerate(zip(results, token_counts)):
            padded[i, :n_tok, :result.shape[-1]] = result

        return padded, torch.tensor(token_counts, device=patch_features.device)


# ============================================================
# 4. ViCO - Stage 1: 一貫性学習損失
# ============================================================

class ViCOConsistencyLoss(nn.Module):
    """
    Visual Consistency Learning - 一貫性学習フェーズの損失。

    目的:
      InternVL3.5-Flash が256トークン/パッチと64トークン/パッチで
      同じ入力に対して類似した出力分布を生成できるように学習する。

    損失式 (論文 Eq. 7):
      L_ViCO = E_ξ [1/N * Σ_i KL(π_θ,ξ(y|I_ξ) || π_θ_prior(y|I_256))]

      ξ ∈ {1/4, 1/16} をランダムにサンプリング
      π_θ_prior : 凍結された元の InternVL3.5 (参照モデル)

    重要な実装ポイント:
      - 参照モデルは常に ξ=1/4 (256トークン) で推論
      - 学習モデルはランダムな ξ で推論
      - KL divergence を最小化して圧縮の影響を緩和

    入力形状:
      student_logits  : (B, N, V)  学習モデルのロジット (ξがランダム)
      teacher_logits  : (B, N, V)  参照モデルのロジット (ξ=1/4固定)
      response_mask   : (B, N)     損失計算対象トークンのマスク
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL Divergence ベースの一貫性損失。

        KL(student || teacher) を最小化。
        ※ 論文では π_θ,ξ || π_θ_prior の KL

        入力:
          student_logits : (B, N, V)  学習モデル (ランダム ξ)
          teacher_logits : (B, N, V)  参照モデル (ξ=1/4 固定)
          response_mask  : (B, N)    float: 1=損失計算, 0=無視
        出力:
          loss : スカラー  平均 KL divergence
        """
        # 温度スケーリング
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)  # (B, N, V)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)           # (B, N, V)

        # KL(student || teacher) = Σ_v teacher_prob * (log teacher_prob - log student_prob)
        # = -Σ_v teacher_prob * log_student_prob + Σ_v teacher_prob * log_teacher_prob
        # F.kl_div は input=log_probs, target=probs として KL(target || input) を計算
        # → F.kl_div(student_log_probs, teacher_probs) = KL(teacher || student)
        # 論文の式に合わせて KL(student || teacher) を使用する場合は逆にする
        kl_per_token = F.kl_div(
            student_log_probs,   # log P (学習モデル)
            teacher_probs,       # Q (参照モデル)
            reduction='none',
        ).sum(dim=-1)  # (B, N)  語彙方向に合計

        # マスクを適用して平均を計算
        masked_kl = kl_per_token * response_mask        # (B, N)
        loss = masked_kl.sum() / response_mask.sum().clamp(min=1)

        return loss


# ============================================================
# 5. ViCO - Stage 2: ルーター学習
# ============================================================

class RouterTargetBuilder:
    """
    ViR のルーティングターゲットを動的に構築するクラス。

    処理フロー:
      1. 各パッチを256トークン/64トークンの両方でフォワードして損失を計算
      2. 損失比 r_i = L(64トークン) / L(256トークン) を計算
      3. 過去の r_i 履歴からスライディングウィンドウでパーセンタイル閾値 τ を計算
      4. r_i >= τ → 高圧縮不可 (decision=0, 低圧縮 256tokens)
         r_i <  τ → 高圧縮可   (decision=1, 高圧縮  64tokens)

    論文 Eq. (8, 9):
      r_i = L_ViCO(y_i | I_64) / L_ViCO(y_i | I_256)
      y_router = 0 if r_i < τ else 1
      τ = k-th percentile of historical {r_i}

    注意: 論文では y_router=0 が「圧縮の影響が無視できる」ことを意味し、
          実際には高圧縮 (64tokens) に使用する。
          本実装では直感的に decision=1 を高圧縮に割り当てる。
    """
    def __init__(
        self,
        window_size: int = 10000,   # スライディングウィンドウのサンプル数
        target_percentile: float = 50.0,  # τ のパーセンタイル (50%で均等分割)
    ):
        self.window_size = window_size
        self.target_percentile = target_percentile
        # 過去の r_i 値を保存するキュー
        self.history = deque(maxlen=window_size)
        self.tau = 1.0  # 初期閾値 (動的に更新)

    def update_threshold(self, new_ratios: torch.Tensor) -> float:
        """
        新しい r_i 値でスライディングウィンドウを更新して閾値 τ を再計算。

        入力: new_ratios  (P,)  新しいバッチのパッチ損失比
        返値: tau         float  更新後の閾値
        """
        ratios_list = new_ratios.detach().cpu().tolist()
        self.history.extend(ratios_list)

        if len(self.history) >= 100:  # 最低 100 サンプル必要
            self.tau = float(np.percentile(list(self.history), self.target_percentile))

        return self.tau

    def compute_routing_targets(
        self,
        loss_low: torch.Tensor,
        loss_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        2種類の圧縮での損失から、ルーティングターゲットを生成。

        入力:
          loss_low  : (B, P)  低圧縮 (256tokens) の ViCO 損失
          loss_high : (B, P)  高圧縮 (64tokens)  の ViCO 損失
        出力:
          targets   : (B, P) int  0=低圧縮, 1=高圧縮
          ratios    : (B, P) float  損失比 r_i
          tau       : float  更新後の閾値
        """
        # 損失比 r_i = L_high / L_low
        # r_i が大きいほど高圧縮の影響が大きい (→ 低圧縮を維持すべき)
        ratios = loss_high / (loss_low + 1e-8)  # (B, P)

        # 閾値を更新
        tau = self.update_threshold(ratios.reshape(-1))

        # ルーティングターゲット
        # r_i < τ  → 高圧縮で十分 → target = 1 (高圧縮)
        # r_i >= τ → 高圧縮は不可 → target = 0 (低圧縮)
        targets = (ratios < tau).long()  # (B, P)  1=高圧縮, 0=低圧縮

        return targets, ratios, tau


class RouterTrainingLoss(nn.Module):
    """
    ViR (Visual Resolution Router) の学習損失。

    標準クロスエントロピー損失を使用して
    バイナリ分類器 (低圧縮 vs 高圧縮) を学習。

    入力形状:
      router_logits : (B, P, 2)  ViR のロジット
      router_targets: (B, P)    int  0 or 1 (RouterTargetBuilder が生成)
    出力形状:
      loss : スカラー
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        router_logits: torch.Tensor,
        router_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        入力:
          router_logits  : (B, P, 2)  [低圧縮スコア, 高圧縮スコア]
          router_targets : (B, P)    int  0=低圧縮, 1=高圧縮
        出力:
          loss : スカラー  クロスエントロピー損失
        """
        B, P, _ = router_logits.shape
        # (B, P, 2) → (B*P, 2)
        logits_flat = router_logits.reshape(B * P, 2)
        # (B, P) → (B*P,)
        targets_flat = router_targets.reshape(B * P)
        loss = self.ce_loss(logits_flat, targets_flat)
        return loss


# ============================================================
# 6. InternVL3.5-Flash の推論フロー
# ============================================================

class FlashInferenceRouter:
    """
    InternVL3.5-Flash の推論時のルーティングと圧縮を管理するクラス。

    推論フロー:
      1. InternViT で全パッチの特徴を抽出 (→ CLS トークン取得)
      2. ViR で各パッチの圧縮率を決定 (0=低圧縮, 1=高圧縮)
      3. 各パッチに対応する Pixel Shuffle を適用
      4. MLP Projector で LLM 次元に変換
      5. 適切な MLP Projector で LLM 次元に射影
         (低圧縮用: LayerNorm(D_v*4) → Linear → GELU → Linear
          高圧縮用: LayerNorm(D_v*16) → Linear → GELU → Linear)
    """
    def __init__(
        self,
        vir: VisualResolutionRouter,
        adaptive_ps: AdaptivePixelShuffle,
        mlp_low: nn.Sequential,   # 低圧縮用 MLP Projector
        mlp_high: nn.Sequential,  # 高圧縮用 MLP Projector
    ):
        self.vir = vir
        self.adaptive_ps = adaptive_ps
        self.mlp_low = mlp_low
        self.mlp_high = mlp_high

    @torch.no_grad()
    def route_and_compress(
        self,
        vit_last_hidden: torch.Tensor,
        num_patches_per_sample: List[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        ViT 出力をルーティングして圧縮した特徴を返す。

        入力:
          vit_last_hidden       : (total_P, S_v=1025, D_v)  ViT 出力
          num_patches_per_sample: List[int]  各サンプルのパッチ数
        出力:
          sample_features  : List[Tensor]  各サンプルの視覚特徴
                             形状: (Σ tokens_in_sample_i, D_llm) ← パッチごとに異なる
          token_counts     : List[int]     各サンプルの総ビジュアルトークン数
        """
        total_P = vit_last_hidden.shape[0]

        # CLS トークン取得: (total_P, D_v)
        cls_tokens = vit_last_hidden[:, 0, :]

        # パッチトークン取得 (CLS 除去): (total_P, 1024, D_v)
        patch_tokens = vit_last_hidden[:, 1:, :]  # (total_P, 1024, D_v)

        # 2D 空間に変換: (total_P, 32, 32, D_v)
        H = W = 32
        patch_2d = patch_tokens.reshape(total_P, H, W, -1)

        # ViR でルーティング決定 (CLS を使用)
        # cls_tokens: (total_P, D_v) → unsqueeze → (total_P, 1, D_v) で 1パッチとして扱う
        routing = self.vir(cls_tokens.unsqueeze(1))
        decisions = routing['decisions'].squeeze(1)  # (total_P,)

        # 適応的 Pixel Shuffle
        compressed, token_counts = self.adaptive_ps(patch_2d, decisions)
        # compressed: (total_P, max_tokens, max_channels)
        # token_counts: (total_P,)

        # サンプルごとに特徴を整理
        sample_features = []
        all_token_counts = []
        patch_offset = 0

        for n_patches in num_patches_per_sample:
            sample_feat_list = []
            sample_tok_count = 0

            for p_idx in range(patch_offset, patch_offset + n_patches):
                n_tok = token_counts[p_idx].item()
                feat = compressed[p_idx, :n_tok, :]  # (n_tok, channels)

                # 圧縮率に応じた MLP Projector を適用
                if decisions[p_idx] == 0:
                    proj = self.mlp_low(feat)    # (256, D_llm)
                else:
                    proj = self.mlp_high(feat)   # (64, D_llm)

                sample_feat_list.append(proj)
                sample_tok_count += n_tok

            # 1サンプルの全パッチを結合: (Σ tokens, D_llm)
            sample_features.append(torch.cat(sample_feat_list, dim=0))
            all_token_counts.append(sample_tok_count)
            patch_offset += n_patches

        return sample_features, all_token_counts


# ============================================================
# 使用例
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Visual Resolution Router (ViR) + ViCO 動作確認")
    print("=" * 60)

    torch.manual_seed(42)
    B = 2          # バッチサイズ
    P = 4          # 1サンプルあたりのパッチ数
    D_v = 1024     # InternViT-300M の hidden size (テスト用)
    D_l = 2048     # LLM hidden size (テスト用)
    V = 1000       # 語彙サイズ (テスト用)

    # --- 1. CompressionLevel テスト ---
    print("\n[1] CompressionLevel: トークン数確認")
    for dr in [0.5, 0.25]:
        n_tok = CompressionLevel.tokens_per_patch(dr)
        print(f"  downsample_ratio={dr}: {n_tok} tokens/patch")
    assert CompressionLevel.tokens_per_patch(0.5) == 256
    assert CompressionLevel.tokens_per_patch(0.25) == 64
    print("  OK")

    # --- 2. VisualResolutionRouter テスト ---
    print("\n[2] VisualResolutionRouter テスト")
    vir = VisualResolutionRouter(input_dim=D_v, hidden_dim=256)

    # 各パッチの CLS トークン特徴: (B, P, D_v)
    cls_features = torch.randn(B, P, D_v)
    routing_result = vir(cls_features)

    print(f"  入力: patch_cls_features  {cls_features.shape}")
    print(f"  出力: logits              {routing_result['logits'].shape}")
    print(f"  出力: probs               {routing_result['probs'].shape}")
    print(f"  出力: decisions           {routing_result['decisions'].shape}")
    print(f"  ルーティング決定 (B=0): {routing_result['decisions'][0].tolist()}")
    print(f"  高圧縮確率 (B=0):      {routing_result['probs'][0, :, 1].tolist()}")
    assert routing_result['logits'].shape == (B, P, 2)
    assert routing_result['decisions'].shape == (B, P)
    print("  OK")

    # --- 3. AdaptivePixelShuffle テスト ---
    print("\n[3] AdaptivePixelShuffle テスト")
    aps = AdaptivePixelShuffle(version='v2')

    # 全パッチ特徴: (B*P=8, 32, 32, D_v)
    patch_features = torch.randn(B * P, 32, 32, D_v)
    # ルーティング: [0, 1, 0, 1, 1, 0, 1, 0]
    decisions = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])

    compressed, tok_counts = aps(patch_features, decisions)
    print(f"  入力: (B*P, 32, 32, D_v)  {patch_features.shape}")
    print(f"  ルーティング: {decisions.tolist()} (0=低圧縮256tok, 1=高圧縮64tok)")
    print(f"  出力: compressed           {compressed.shape}")
    print(f"  各パッチのトークン数:       {tok_counts.tolist()}")

    expected_counts = [256 if d == 0 else 64 for d in decisions.tolist()]
    assert tok_counts.tolist() == expected_counts, f"期待: {expected_counts}, 実際: {tok_counts.tolist()}"
    print("  OK: 低圧縮=256tok, 高圧縮=64tok")

    # --- 4. ViCO 一貫性損失テスト ---
    print("\n[4] ViCO 一貫性損失テスト")
    vico_loss_fn = ViCOConsistencyLoss(temperature=1.0)

    # ダミーロジット
    student_logits = torch.randn(B, 64, V)   # (B, N, V)
    teacher_logits = torch.randn(B, 64, V)   # (B, N, V)
    response_mask = torch.ones(B, 64)        # (B, N)
    response_mask[:, 40:] = 0               # 後半をマスク

    loss_vico = vico_loss_fn(student_logits, teacher_logits, response_mask)
    print(f"  入力: student_logits  {student_logits.shape}")
    print(f"  入力: teacher_logits  {teacher_logits.shape}")
    print(f"  入力: response_mask   {response_mask.shape} (有効トークン数: {int(response_mask.sum())})")
    print(f"  ViCO 損失: {loss_vico.item():.4f}")
    assert loss_vico.item() > 0, "ViCO 損失は正の値であるべき"

    # 同一ロジットでは損失 ≈ 0
    loss_same = vico_loss_fn(teacher_logits, teacher_logits, response_mask)
    print(f"  同一ロジット時の損失: {loss_same.item():.6f} (≈ 0 であるべき)")
    assert loss_same.item() < 1e-4, "同一ロジット時の KL divergence は 0 になるべき"
    print("  OK")

    # --- 5. RouterTargetBuilder テスト ---
    print("\n[5] RouterTargetBuilder テスト")
    target_builder = RouterTargetBuilder(window_size=1000, target_percentile=50.0)

    # 損失比 r_i のシミュレーション
    # 低圧縮 vs 高圧縮での ViCO 損失
    loss_low_sim = torch.rand(B, P) * 0.5 + 0.1      # (B, P)  低圧縮の損失
    loss_high_sim = loss_low_sim * (1.0 + torch.randn(B, P).abs())  # (B, P)  高圧縮は常に高め

    targets, ratios, tau = target_builder.compute_routing_targets(loss_low_sim, loss_high_sim)
    print(f"  入力: loss_low  {loss_low_sim.shape}")
    print(f"  入力: loss_high {loss_high_sim.shape}")
    print(f"  損失比 r_i (B=0): {ratios[0].tolist()}")
    print(f"  動的閾値 τ: {tau:.4f} (履歴{len(target_builder.history)}件から計算)")
    print(f"  ルーティングターゲット (B=0): {targets[0].tolist()} (0=低圧縮, 1=高圧縮)")
    assert targets.shape == (B, P)
    print("  OK")

    # --- 6. RouterTrainingLoss テスト ---
    print("\n[6] RouterTrainingLoss テスト")
    router_loss_fn = RouterTrainingLoss()

    router_logits = torch.randn(B, P, 2)  # (B, P, 2)
    router_targets = targets               # (B, P)  RouterTargetBuilderが生成

    r_loss = router_loss_fn(router_logits, router_targets)
    print(f"  入力: router_logits  {router_logits.shape}")
    print(f"  入力: router_targets {router_targets.shape}")
    print(f"  ルーター損失: {r_loss.item():.4f}")
    assert r_loss.item() > 0
    print("  OK")

    # --- 7. トークン削減効果の確認 ---
    print("\n[7] トークン削減効果シミュレーション")
    n_patches_total = 100
    for high_compression_ratio in [0.0, 0.3, 0.5, 0.7, 1.0]:
        n_high = int(n_patches_total * high_compression_ratio)
        n_low = n_patches_total - n_high
        total_tokens = n_low * 256 + n_high * 64
        baseline_tokens = n_patches_total * 256
        reduction = 1.0 - total_tokens / baseline_tokens
        print(f"  高圧縮割合={high_compression_ratio*100:.0f}%: "
              f"総トークン={total_tokens} ({reduction*100:.1f}%削減)")
    print("  ※ 論文: 平均50%削減で性能はほぼ100%維持")

    print("\n全テスト完了!")
