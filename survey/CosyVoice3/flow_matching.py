"""
CosyVoice3 Conditional Flow Matching (DiTベース) - 簡略化疑似コード
====================================================================

離散音声トークンからメルスペクトログラムを生成する
非自己回帰モジュール (粗→細の「細」段階)。

論文: CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
公式実装:
  - cosyvoice/flow/flow.py (CausalMaskedDiffWithDiT)
  - cosyvoice/flow/flow_matching.py (CausalConditionalCFM)
  - cosyvoice/flow/DiT/dit.py (DiT)
  - cosyvoice/flow/DiT/modules.py (DiTBlock等)

CosyVoice2との違い:
- CFMアーキテクチャ: U-Netベース → DiT (Diffusion Transformer) に変更
- パラメータ数: ~100M → 300M
- テキストエンコーダ不要 (DiT内でトークン特徴を直接条件付け)
- 長さ正規化モジュール不要 (単純な補間で解決)

Shape Convention
============================================================
B: バッチサイズ
T_speech: 音声トークン長 (25Hz)
T_mel: メルスペクトログラムのフレーム数 (= T_speech × token_mel_ratio)
D_mel: メル周波数ビン数 (80)
D_dit: DiT隠れ次元 (1024)
D_head: アテンションヘッド次元 (64)
D_spk: 話者埋め込み次元 (192 → 80に射影)
N_heads: アテンションヘッド数 (16)
N_layers: DiTレイヤー数 (22)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class CausalMaskedDiffWithDiT(nn.Module):
    """
    Conditional Flow Matching with Diffusion Transformer

    音声トークン → メルスペクトログラム変換の全体フロー:

    ┌──────────────────────────────────────────────────────────┐
    │ 音声トークン (B, T_speech)                                │
    │     ↓                                                    │
    │ Token Embedding (6561 → 896)                             │
    │     ↓                                                    │
    │ PreLookahead Layer (3トークン先読み)                       │
    │     ↓                                                    │
    │ token_features: (B, T_speech, 896)                       │
    │     ↓                                                    │
    │ Encoder Projection (896 → 80)                            │
    │     ↓                                                    │
    │ mu: (B, T_speech, 80)                                    │
    │     ↓                                                    │
    │ Interpolation (T_speech → T_mel, ×2)                     │
    │     ↓                                                    │
    │ mu: (B, T_mel, 80)                                       │
    │                                                          │
    │ 話者埋め込み (B, 192) → Linear → (B, 80)                  │
    │                                                          │
    │ ┌────────────────────────────────────────────────────┐   │
    │ │ Conditional Flow Matching (DiT)                     │   │
    │ │                                                    │   │
    │ │ 初期ノイズ z ~ N(0, I): (B, T_mel, 80)              │   │
    │ │     ↓                                              │   │
    │ │ Euler ODE Solver (10ステップ)                        │   │
    │ │   t: 0.0 → 1.0                                    │   │
    │ │   各ステップで DiT が速度場を推定                     │   │
    │ │     ↓                                              │   │
    │ │ mel: (B, T_mel, 80)                                │   │
    │ └────────────────────────────────────────────────────┘   │
    │     ↓                                                    │
    │ 転置: (B, 80, T_mel) - メルスペクトログラム                │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        input_size: int = 80,                # メル次元
        output_size: int = 80,               # 出力メル次元
        spk_embed_dim: int = 192,            # 話者埋め込み次元
        vocab_size: int = 6561,              # 音声トークン語彙サイズ
        token_embed_dim: int = 896,          # トークン埋め込み次元
        token_mel_ratio: int = 2,            # トークン→メルの倍率
        pre_lookahead_len: int = 3,          # 先読みトークン数
        dit_dim: int = 1024,                 # DiT隠れ次元
        dit_depth: int = 22,                 # DiTレイヤー数
        dit_heads: int = 16,                 # アテンションヘッド数
        n_timesteps: int = 10,               # ODE推論ステップ数
    ):
        super().__init__()

        self.token_mel_ratio = token_mel_ratio
        self.n_timesteps = n_timesteps

        # ========================================
        # 1. トークン埋め込み
        # ========================================
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size,       # 6561
            embedding_dim=token_embed_dim,   # 896
        )
        # 入力: (B, T_speech) → 出力: (B, T_speech, 896)

        # ========================================
        # 2. PreLookahead Layer
        # ========================================
        # ストリーミング対応のため、先読みを制限
        self.pre_lookahead_layer = PreLookaheadLayer(
            embed_dim=token_embed_dim,       # 896
            lookahead_len=pre_lookahead_len,  # 3
        )
        # 入力: (B, T_speech, 896) → 出力: (B, T_speech, 896)

        # ========================================
        # 3. エンコーダ射影
        # ========================================
        self.encoder_proj = nn.Linear(token_embed_dim, input_size)
        # 入力: (B, T_speech, 896) → 出力: (B, T_speech, 80)

        # ========================================
        # 4. 話者埋め込み射影
        # ========================================
        self.spk_proj = nn.Linear(spk_embed_dim, input_size)
        # 入力: (B, 192) → 出力: (B, 80)

        # ========================================
        # 5. Conditional Flow Matching デコーダ
        # ========================================
        self.decoder = CausalConditionalCFM(
            input_size=input_size,           # 80
            dit_dim=dit_dim,                 # 1024
            dit_depth=dit_depth,             # 22
            dit_heads=dit_heads,             # 16
            n_timesteps=n_timesteps,         # 10
        )

    def forward(
        self,
        speech_tokens: torch.Tensor,         # (B, T_speech)
        speech_tokens_len: torch.Tensor,     # (B,)
        target_mel: torch.Tensor,            # (B, 80, T_mel)
        target_mel_len: torch.Tensor,        # (B,)
        speaker_embedding: torch.Tensor,     # (B, 192)
    ) -> torch.Tensor:
        """
        学習時のフォワードパス

        入力:
            speech_tokens: (B, T_speech) - 正解音声トークン
            speech_tokens_len: (B,) - トークン長
            target_mel: (B, 80, T_mel) - 正解メルスペクトログラム
            target_mel_len: (B,) - メル長
            speaker_embedding: (B, 192) - 話者埋め込み

        出力:
            loss: スカラー - CFMロス (条件付きフローマッチング損失)

        学習の仕組み:
            1. ランダムな時刻 t ~ U(0, 1) をサンプル
            2. ノイズ z ~ N(0, I) をサンプル
            3. 補間パス: x_t = (1 - t) * z + t * target_mel
            4. DiTで速度場 v_theta(x_t, t, mu, spk) を推定
            5. ロス: ||v_theta - (target_mel - z)||^2
        """
        # トークン特徴抽出
        token_embeds = self.input_embedding(speech_tokens)
        # token_embeds: (B, T_speech, 896)

        token_features = self.pre_lookahead_layer(token_embeds)
        # token_features: (B, T_speech, 896)

        mu = self.encoder_proj(token_features)
        # mu: (B, T_speech, 80)

        # 補間: T_speech → T_mel (×2)
        mu = F.interpolate(
            mu.transpose(1, 2),              # (B, 80, T_speech)
            size=target_mel.shape[-1],       # T_mel
            mode='nearest',
        )
        # mu: (B, 80, T_mel)

        # 話者埋め込み射影
        spk = self.spk_proj(speaker_embedding)
        # spk: (B, 80)

        # CFMロス計算
        loss = self.decoder.compute_loss(
            x1=target_mel,    # (B, 80, T_mel) - ターゲット
            mu=mu,            # (B, 80, T_mel) - 条件 (トークン特徴)
            spks=spk,         # (B, 80) - 話者
            mask=None,        # マスク
        )

        return loss

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,         # (B, T_speech)
        speaker_embedding: torch.Tensor,     # (B, 192)
    ) -> torch.Tensor:
        """
        推論: 音声トークン → メルスペクトログラム

        入力:
            speech_tokens: (B, T_speech) - 音声トークン (LLMの出力)
            speaker_embedding: (B, 192) - 話者埋め込み

        出力:
            mel: (B, 80, T_mel) - メルスペクトログラム
                 T_mel = T_speech × token_mel_ratio (= T_speech × 2)

        処理:
            1. トークン → 条件特徴 mu
            2. 補間 (T_speech → T_mel)
            3. Euler ODE solver で z (ノイズ) → mel を推定
        """
        B = speech_tokens.shape[0]
        T_speech = speech_tokens.shape[1]
        T_mel = T_speech * self.token_mel_ratio

        # トークン特徴抽出
        token_embeds = self.input_embedding(speech_tokens)
        token_features = self.pre_lookahead_layer(token_embeds)
        mu = self.encoder_proj(token_features)
        # mu: (B, T_speech, 80)

        # 補間: T_speech → T_mel
        mu = F.interpolate(
            mu.transpose(1, 2),      # (B, 80, T_speech)
            size=T_mel,
            mode='nearest',
        )
        # mu: (B, 80, T_mel)

        # 話者埋め込み
        spk = self.spk_proj(speaker_embedding)
        # spk: (B, 80)

        # CFM推論 (Euler ODE)
        mel = self.decoder.inference(
            mu=mu,            # (B, 80, T_mel)
            spks=spk,         # (B, 80)
            n_timesteps=self.n_timesteps,
        )
        # mel: (B, 80, T_mel)

        return mel


class CausalConditionalCFM(nn.Module):
    """
    Conditional Flow Matching (Matcha-TTSベース)

    Optimal Transport CFM (OT-CFM):
    - 確率パス: p_t(x | x_1) = N(x | t*x_1, (1-(1-σ)t)^2 I)
    - 速度場: u_t(x | x_1) = (x_1 - (1-σ)x) / (1 - (1-σ)t)
    - 推論: z_0 ~ N(0, I) → ODE を解いて x_1 を生成

    Classifier-Free Guidance (CFG):
    - 学習時: training_cfg_rate=0.2 でランダムにmu=0
    - 推論時: v = (1+w)*v_cond - w*v_uncond, w=inference_cfg_rate=0.7
    """

    def __init__(
        self,
        input_size: int = 80,
        dit_dim: int = 1024,
        dit_depth: int = 22,
        dit_heads: int = 16,
        sigma_min: float = 1e-6,
        n_timesteps: int = 10,
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
    ):
        super().__init__()

        self.sigma_min = sigma_min
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate

        # DiT (Diffusion Transformer) - 速度場推定器
        self.estimator = DiT(
            dim=dit_dim,             # 1024
            depth=dit_depth,         # 22
            heads=dit_heads,         # 16
            dim_head=64,
            mel_dim=input_size,      # 80
        )

    def compute_loss(
        self,
        x1: torch.Tensor,       # (B, 80, T_mel) - ターゲットメル
        mu: torch.Tensor,       # (B, 80, T_mel) - 条件 (トークン特徴)
        spks: torch.Tensor,     # (B, 80) - 話者
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CFM学習ロス計算

        入力:
            x1: (B, 80, T_mel) - ターゲットメルスペクトログラム
            mu: (B, 80, T_mel) - 条件 (エンコーダ出力)
            spks: (B, 80) - 話者埋め込み

        出力:
            loss: スカラー - フローマッチング損失

        数式:
            t ~ U(0, 1)
            z ~ N(0, I)
            x_t = (1 - (1-σ)t) * z + t * x1
            target_flow = x1 - (1-σ) * z
            v_theta = DiT(x_t, t, mu, spks)
            loss = ||v_theta - target_flow||^2
        """
        B = x1.shape[0]

        # ランダム時刻サンプリング (cosineスケジュール)
        t = torch.rand(B, device=x1.device)
        t = 1 - torch.cos(t * math.pi / 2)  # cosineスケジュール
        # t: (B,) ∈ [0, 1]

        # ノイズサンプリング
        z = torch.randn_like(x1)
        # z: (B, 80, T_mel)

        # 補間パス
        t_expanded = t[:, None, None]  # (B, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * t_expanded) * z + t_expanded * x1
        # x_t: (B, 80, T_mel)

        # CFG: 学習時にランダムに条件を無効化
        if self.training and self.training_cfg_rate > 0:
            cfg_mask = torch.rand(B, device=x1.device) < self.training_cfg_rate
            mu = mu * (~cfg_mask)[:, None, None]

        # DiTで速度場推定
        v_theta = self.estimator(
            x=x_t.transpose(1, 2),       # (B, T_mel, 80)
            mu=mu.transpose(1, 2),       # (B, T_mel, 80)
            t=t,                         # (B,)
            spks=spks,                   # (B, 80)
        )
        # v_theta: (B, T_mel, 80)

        # ターゲット速度場
        target_flow = x1 - (1 - self.sigma_min) * z
        # target_flow: (B, 80, T_mel)

        # L2ロス
        loss = F.mse_loss(
            v_theta,
            target_flow.transpose(1, 2),
        )

        return loss

    @torch.inference_mode()
    def inference(
        self,
        mu: torch.Tensor,       # (B, 80, T_mel) - 条件
        spks: torch.Tensor,     # (B, 80) - 話者
        n_timesteps: int = 10,  # ODEステップ数
    ) -> torch.Tensor:
        """
        Euler ODE Solver による推論

        入力:
            mu: (B, 80, T_mel) - 条件 (エンコーダ出力)
            spks: (B, 80) - 話者埋め込み
            n_timesteps: ODEソルバーのステップ数

        出力:
            x: (B, 80, T_mel) - 生成されたメルスペクトログラム

        処理:
            1. z ~ N(0, I): 初期ノイズ
            2. dt = 1/n_timesteps: ステップ幅
            3. for t in [0, dt, 2dt, ..., 1-dt]:
                 v = DiT(x_t, t, mu, spks)  # 速度場推定
                 x_{t+dt} = x_t + dt * v    # Eulerステップ
            4. x_1 がメルスペクトログラム
        """
        B, _, T_mel = mu.shape

        # 初期ノイズ
        z = torch.randn(B, 80, T_mel, device=mu.device)
        # z: (B, 80, T_mel)

        x = z
        dt = 1.0 / n_timesteps

        for i in range(n_timesteps):
            t = torch.full((B,), i * dt, device=mu.device)
            # t: (B,) - 現在時刻

            # Classifier-Free Guidance
            # 条件付き速度場
            v_cond = self.estimator(
                x=x.transpose(1, 2),         # (B, T_mel, 80)
                mu=mu.transpose(1, 2),       # (B, T_mel, 80)
                t=t,                         # (B,)
                spks=spks,                   # (B, 80)
            )
            # v_cond: (B, T_mel, 80)

            if self.inference_cfg_rate > 0:
                # 無条件速度場 (mu=0)
                v_uncond = self.estimator(
                    x=x.transpose(1, 2),
                    mu=torch.zeros_like(mu).transpose(1, 2),
                    t=t,
                    spks=spks,
                )
                # v_uncond: (B, T_mel, 80)

                # CFG: v = (1+w)*v_cond - w*v_uncond
                v = (1 + self.inference_cfg_rate) * v_cond \
                    - self.inference_cfg_rate * v_uncond
            else:
                v = v_cond
            # v: (B, T_mel, 80)

            # Eulerステップ
            x = x + dt * v.transpose(1, 2)
            # x: (B, 80, T_mel)

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)

    フローマッチングの速度場を推定するネットワーク。
    ノイズ付きメル + 条件(トークン特徴) + 時刻 + 話者 → 速度場

    アーキテクチャ:
    ┌──────────────────────────────────────────────────┐
    │ InputEmbedding                                    │
    │   x(80) + mu(80) + cond(80) + text(80) → (1024)  │
    │   + spk(80) → bias                              │
    │   + TimestepEmbedding(t → 1024)                  │
    ├──────────────────────────────────────────────────┤
    │ 22 × DiTBlock:                                    │
    │   ┌──────────────────────────────────────────┐   │
    │   │ AdaLayerNormZero (時刻条件付き正規化)      │   │
    │   │   → shift, scale, gate (6つのパラメータ)   │   │
    │   │                                          │   │
    │   │ Multi-Head Self-Attention (16ヘッド)      │   │
    │   │   Q,K,V: (B, T_mel, 1024)                │   │
    │   │   + Rotary Position Embedding             │   │
    │   │                                          │   │
    │   │ Feed-Forward Network (1024 → 2048 → 1024) │   │
    │   │   + SiLU活性化                            │   │
    │   └──────────────────────────────────────────┘   │
    │ + Long Skip Connections (中間層→後半層)           │
    ├──────────────────────────────────────────────────┤
    │ AdaLayerNormZero_Final                            │
    │ Linear Projection (1024 → 80)                    │
    │   → 速度場推定: (B, T_mel, 80)                    │
    └──────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        dim: int = 1024,            # DiT隠れ次元
        depth: int = 22,            # レイヤー数
        heads: int = 16,            # アテンションヘッド数
        dim_head: int = 64,         # ヘッドあたり次元
        ff_mult: int = 2,           # FFN中間倍率
        mel_dim: int = 80,          # メル次元
        dropout: float = 0.1,
    ):
        super().__init__()

        self.depth = depth

        # ========================================
        # 時刻埋め込み
        # ========================================
        self.time_embed = TimestepEmbedding(dim=dim)
        # 入力: t (B,) → 出力: (B, dim)

        # ========================================
        # 入力埋め込み (4つの入力を融合)
        # ========================================
        self.input_embed = InputEmbedding(
            mel_dim=mel_dim,         # 80
            out_dim=dim,             # 1024
        )
        # ノイズ付きメル + 条件 + cond(マスク) + text → (B, T_mel, 1024)
        # 話者 → (B, 1024) bias

        # ========================================
        # 22 × DiTBlock
        # ========================================
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ========================================
        # Long Skip Connections
        # ========================================
        # 前半レイヤーの出力を後半レイヤーにスキップ接続
        # layer[i] → layer[depth-1-i] (i < depth//2)
        self.long_skip_projs = nn.ModuleList([
            nn.Linear(dim * 2, dim)
            for _ in range(depth // 2)
        ])

        # ========================================
        # 出力層
        # ========================================
        self.norm_out = AdaLayerNormZeroFinal(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        # 出力: (B, T_mel, 80)

    def forward(
        self,
        x: torch.Tensor,        # (B, T_mel, 80) - ノイズ付きメル
        mu: torch.Tensor,       # (B, T_mel, 80) - 条件 (エンコーダ出力)
        t: torch.Tensor,        # (B,) - 時刻
        spks: torch.Tensor,     # (B, 80) - 話者埋め込み
        cond: Optional[torch.Tensor] = None,  # (B, T_mel, 80) - マスク条件
    ) -> torch.Tensor:
        """
        DiTフォワードパス

        入力:
            x: (B, T_mel, 80) - ノイズ付きメルスペクトログラム
                B: バッチサイズ
                T_mel: メルフレーム数
                80: メル周波数ビン
            mu: (B, T_mel, 80) - トークン特徴 (補間済み)
            t: (B,) - 拡散時刻 ∈ [0, 1]
            spks: (B, 80) - 話者埋め込み (射影済み)
            cond: (B, T_mel, 80) - マスクされたメル (プロンプト部分)
                  推論時はゼロ

        出力:
            v: (B, T_mel, 80) - 推定速度場
        """
        if cond is None:
            cond = torch.zeros_like(x)

        # 時刻埋め込み
        t_embed = self.time_embed(t)
        # t_embed: (B, 1024)

        # 入力融合
        h, spk_bias = self.input_embed(
            x=x,           # (B, T_mel, 80)
            mu=mu,         # (B, T_mel, 80)
            cond=cond,     # (B, T_mel, 80)
            spks=spks,     # (B, 80)
        )
        # h: (B, T_mel, 1024) - 融合特徴
        # spk_bias: (B, 1024) - 話者バイアス

        h = h + spk_bias.unsqueeze(1)
        # h: (B, T_mel, 1024)

        # Transformer Blocks with Long Skip Connections
        skips = []
        for i in range(self.depth):
            if i < self.depth // 2:
                # 前半: スキップ接続用に保存
                h = self.transformer_blocks[i](h, t_embed)
                skips.append(h)
            else:
                # 後半: スキップ接続を適用
                skip = skips.pop()
                h = torch.cat([h, skip], dim=-1)
                # h: (B, T_mel, 2048)
                h = self.long_skip_projs[i - self.depth // 2](h)
                # h: (B, T_mel, 1024)
                h = self.transformer_blocks[i](h, t_embed)
            # h: (B, T_mel, 1024)

        # 出力射影
        h = self.norm_out(h, t_embed)
        # h: (B, T_mel, 1024)

        v = self.proj_out(h)
        # v: (B, T_mel, 80) - 推定速度場

        return v


class DiTBlock(nn.Module):
    """
    DiT Transformer Block

    AdaLayerNormZero + Multi-Head Attention + FFN

    時刻埋め込みから6つの変調パラメータを生成:
    (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)

    入力: (B, T_mel, 1024)
    出力: (B, T_mel, 1024)
    """

    def __init__(
        self,
        dim: int = 1024,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Adaptive Layer Norm (時刻条件付き)
        self.ada_norm = AdaLayerNormZero(dim)
        # t_embed (B, 1024) → 6つの変調パラメータ

        # Multi-Head Self-Attention + RoPE
        self.attn = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Feed-Forward Network (SiLU)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),     # 1024 → 2048
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),     # 2048 → 1024
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, T, 1024)
        t_embed: torch.Tensor,  # (B, 1024) - 時刻埋め込み
    ) -> torch.Tensor:
        """
        入力:
            x: (B, T, 1024) - 入力特徴
            t_embed: (B, 1024) - 時刻埋め込み

        出力:
            x: (B, T, 1024) - 変換後の特徴
        """
        # AdaLayerNorm: 時刻条件付き正規化 + 変調パラメータ生成
        shift_msa, scale_msa, gate_msa, \
            shift_ffn, scale_ffn, gate_ffn = self.ada_norm(x, t_embed)
        # 各パラメータ: (B, 1, 1024)

        # Self-Attention
        x_norm = x * (1 + scale_msa) + shift_msa
        # x_norm: (B, T, 1024) - 正規化 + シフト/スケール
        x = x + gate_msa * self.attn(x_norm)
        # x: (B, T, 1024)

        # FFN
        x_norm = x * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.ffn(x_norm)
        # x: (B, T, 1024)

        return x


class TimestepEmbedding(nn.Module):
    """
    時刻埋め込み (Sinusoidal + MLP)

    t (スカラー) → 高次元埋め込みベクトル

    入力: t (B,) ∈ [0, 1]
    出力: (B, 1024)

    処理:
    1. Sinusoidal encoding: t → (B, dim)
    2. MLP: Linear → SiLU → Linear
    """
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        # Sinusoidal encoding
        half_dim = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        # emb: (B, dim)
        return self.mlp(emb)
        # 出力: (B, 1024)


class InputEmbedding(nn.Module):
    """
    入力融合モジュール

    4つの入力 (ノイズ付きメル, 条件, マスク, テキスト) を結合して
    DiTの入力次元に射影。

    入力:
        x: (B, T, 80) - ノイズ付きメル
        mu: (B, T, 80) - トークン条件
        cond: (B, T, 80) - マスク条件 (プロンプト部分)
        spks: (B, 80) - 話者

    出力:
        h: (B, T, 1024) - 融合特徴
        spk_bias: (B, 1024) - 話者バイアス
    """
    def __init__(self, mel_dim: int = 80, out_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 3, out_dim)  # x + mu + cond
        self.spk_proj = nn.Linear(mel_dim, out_dim)

    def forward(self, x, mu, cond, spks):
        h = self.proj(torch.cat([x, mu, cond], dim=-1))
        # h: (B, T, 1024)
        spk_bias = self.spk_proj(spks)
        # spk_bias: (B, 1024)
        return h, spk_bias


class AdaLayerNormZero(nn.Module):
    """
    Adaptive Layer Normalization Zero

    時刻埋め込みから6つの変調パラメータを生成:
    shift_msa, scale_msa, gate_msa,
    shift_ffn, scale_ffn, gate_ffn

    入力: x (B, T, 1024), t_embed (B, 1024)
    出力: 6 × (B, 1, 1024)
    """

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 時刻埋め込みから6つの変調パラメータを生成
        # ゼロ初期化: 学習初期はIdentity変換として振る舞う
        self.linear = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor):
        # x: (B, T, 1024), t_embed: (B, 1024)
        params = self.linear(F.silu(t_embed))
        # params: (B, 6 * 1024)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = \
            params.unsqueeze(1).chunk(6, dim=-1)
        # 各パラメータ: (B, 1, 1024)
        return shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn


class AdaLayerNormZeroFinal(nn.Module):
    """最終層用 (2パラメータ: shift, scale のみ)"""

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor):
        # x: (B, T, 1024), t_embed: (B, 1024)
        params = self.linear(F.silu(t_embed))
        shift, scale = params.unsqueeze(1).chunk(2, dim=-1)
        # shift, scale: (B, 1, 1024)
        return self.norm(x) * (1 + scale) + shift
        # 出力: (B, T, 1024)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embedding

    16ヘッド × 64次元 = 1024次元
    RoPEにより位置情報を注入 (外挿性能が高い)

    入力: (B, T, 1024)
    内部:
        Q, K, V: (B, 16, T, 64)
        Attention: softmax(QK^T / sqrt(64)) V
    出力: (B, T, 1024)
    """

    def __init__(self, dim: int = 1024, heads: int = 16, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head  # 1024

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.to_q(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        # q, k, v: (B, 16, T, 64)

        # RoPE適用 (位置に依存した回転行列をQ, Kに乗算)
        # q, k = apply_rope(q, k, freqs)  # 外部から事前計算されたfreqsを使用

        # Scaled Dot-Product Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # attn_out: (B, 16, T, 64)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        # attn_out: (B, T, 1024)

        return self.to_out(attn_out)
        # 出力: (B, T, 1024)


class PreLookaheadLayer(nn.Module):
    """
    先読みレイヤー (ストリーミング対応)

    各トークン位置で、未来の pre_lookahead_len トークンまで参照可能。
    因果マスクに先読みウィンドウを追加。

    入力: (B, T_speech, 896)
    出力: (B, T_speech, 896)

    パラメータ:
        lookahead_len: 3 (3トークン = 120ms先まで参照)
    """

    def __init__(self, dim: int = 896, lookahead_len: int = 3, num_heads: int = 4):
        super().__init__()
        self.lookahead_len = lookahead_len
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 896)
        B, T, D = x.shape

        # 因果マスク + 先読みウィンドウ
        # 通常の因果マスク: position i は [0, i] のみ参照可能
        # 先読み付き:       position i は [0, i + lookahead_len] まで参照可能
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=self.lookahead_len + 1)
        # mask[i, j] = True (マスク) if j > i + lookahead_len

        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        return x + attn_out
        # 出力: (B, T, 896)
