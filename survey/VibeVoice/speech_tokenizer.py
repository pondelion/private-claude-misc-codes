"""
VibeVoice Speech Tokenizer - 超低フレームレート σ-VAE 音声トークナイザ

公式実装: vibevoice/modular/modular_vibevoice_tokenizer.py

VibeVoice の音声トークナイザは2種類:
  1. Acoustic Tokenizer: σ-VAE ベースの音響圧縮（エンコーダ + デコーダ）
     - 24kHz → 7.5Hz（3200倍圧縮）
     - VAE 次元: 64
     - fix_std: 0.5（固定分散）
     - 用途: 音質・音色の保存と復元

  2. Semantic Tokenizer: 決定論的な意味特徴抽出（エンコーダのみ）
     - 24kHz → 7.5Hz（3200倍圧縮）
     - VAE 次元: 128
     - fix_std: 0（サンプリングなし）
     - 用途: 言語的意味の抽出（ASR プロキシタスクで学習）

参照: modular_vibevoice_tokenizer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# ============================================================================
# ストリーミングキャッシュ
# ============================================================================

class VibeVoiceTokenizerStreamingCache:
    """
    畳み込みレイヤーのストリーミングキャッシュ。
    LLM の KVキャッシュに相当する役割を畳み込みネットワークで果たす。

    因果畳み込みでは過去のコンテキストが必要なため、
    各レイヤーの最後の context_size 個のサンプルをキャッシュする。

    構造: Dict[(layer_id, sample_idx) → state_tensor]

    参照: modular_vibevoice_tokenizer.py の VibeVoiceTokenizerStreamingCache
    """

    def __init__(self):
        self.states: Dict[Tuple[int, int], torch.Tensor] = {}

    def get(self, layer_id: int, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        指定レイヤー・サンプルのキャッシュ状態を取得。

        Args:
            layer_id: 畳み込みレイヤーのID
            sample_indices: [B] バッチ内の各サンプルインデックス

        Returns:
            stacked_states: [B, C, context_size] or None
            存在しないサンプルがあれば None を返す
        """
        states = []
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.states:
                return None
            states.append(self.states[key])

        # パディング（長さが異なる場合）
        max_len = max(s.shape[-1] for s in states)
        padded = []
        for s in states:
            if s.shape[-1] < max_len:
                pad = torch.zeros(*s.shape[:-1], max_len - s.shape[-1],
                                  device=s.device, dtype=s.dtype)
                s = torch.cat([pad, s], dim=-1)
            padded.append(s)

        return torch.stack(padded, dim=0)  # [B, C, context_size]

    def set(self, layer_id: int, sample_indices: torch.Tensor,
            states: torch.Tensor):
        """
        キャッシュ状態を保存。

        Args:
            layer_id: 畳み込みレイヤーのID
            sample_indices: [B]
            states: [B, C, context_size]
        """
        for i, idx in enumerate(sample_indices.tolist()):
            self.states[(layer_id, idx)] = states[i].detach()

    def clear(self, layer_id: Optional[int] = None,
              sample_indices: Optional[torch.Tensor] = None):
        """選択的にキャッシュをクリア"""
        if layer_id is not None and sample_indices is not None:
            for idx in sample_indices.tolist():
                self.states.pop((layer_id, idx), None)


# ============================================================================
# 正規化モジュール
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。
    LayerNorm よりも軽量で、平均の減算を省略。

    数式: x_norm = x / sqrt(mean(x²) + eps) * weight

    参照: modular_vibevoice_tokenizer.py の RMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, dim]
        Returns:
            [*, dim] 正規化済みテンソル
        """
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x * norm
        if self.elementwise_affine:
            x_norm = x_norm * self.weight
        return x_norm


class ConvRMSNorm(nn.Module):
    """
    畳み込み出力 [B, C, T] に対応した RMSNorm。
    [B, C, T] → permute → [B, T, C] → RMSNorm → permute → [B, C, T]

    参照: modular_vibevoice_tokenizer.py の ConvRMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] 畳み込み出力
        Returns:
            [B, C, T] 正規化済み
        """
        x = x.permute(0, 2, 1)      # [B, C, T] → [B, T, C]
        x = self.norm(x)             # [B, T, C]
        x = x.permute(0, 2, 1)      # [B, T, C] → [B, C, T]
        return x


# ============================================================================
# 畳み込みモジュール（ストリーミング対応）
# ============================================================================

class SConv1d(nn.Module):
    """
    ストリーミング対応の因果 Conv1d。
    因果パディング（左のみ）により、未来の情報を見ない。

    非ストリーミング: 左に padding_total 分ゼロパディング
    ストリーミング: キャッシュから前回の context_size サンプルを結合

    context_size = (kernel_size - 1) * dilation - (stride - 1)

    参照: modular_vibevoice_tokenizer.py の SConv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = True,
        pad_mode: str = 'constant',
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=bias,
        )
        self.causal = causal
        self.pad_mode = pad_mode
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

        # ストリーミング時のキャッシュサイズ
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)

    def forward(
        self,
        x: torch.Tensor,                              # [B, C, T]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T] 入力テンソル
            cache: ストリーミングキャッシュ
            use_cache: True ならストリーミングモード
            is_final_chunk: True なら最終チャンク（余分なパディング追加）

        Returns:
            out: [B, C_out, T'] ダウンサンプリング後のテンソル
                 T' = ceil(T / stride)
        """
        if not use_cache:
            # === 非ストリーミングモード ===
            padding_total = (self.kernel_size - 1) * self.dilation
            # ceil 動作のための追加パディング
            extra_padding = self._get_extra_padding(x, padding_total)

            if self.causal:
                # 因果: 左のみパディング
                x = F.pad(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # 非因果: 対称パディング
                left = padding_total // 2
                right = padding_total - left + extra_padding
                x = F.pad(x, (left, right), mode=self.pad_mode)

            return self.conv(x)
            # [B, C_out, ceil(T / stride)]
        else:
            # === ストリーミングモード ===
            # キャッシュから前回のコンテキストを取得
            cached = cache.get(layer_id, sample_indices)
            if cached is not None:
                x = torch.cat([cached, x], dim=-1)  # [B, C, context + T]

            if is_final_chunk:
                extra = self._get_extra_padding(x, 0)
                x = F.pad(x, (0, extra))

            out = self.conv(x)
            # [B, C_out, T']

            # キャッシュ更新: 末尾 context_size サンプルを保存
            if self.context_size > 0:
                cache.set(layer_id, sample_indices, x[..., -self.context_size:])

            return out

    def _get_extra_padding(self, x: torch.Tensor, padding_total: int) -> int:
        """ceil ダウンサンプリングのための追加パディング計算"""
        T = x.shape[-1] + padding_total
        T_out = (T + self.stride - 1) // self.stride
        T_needed = (T_out - 1) * self.stride + self.kernel_size * self.dilation - (self.dilation - 1)
        return max(0, T_needed - T)


class SConvTranspose1d(nn.Module):
    """
    ストリーミング対応の ConvTranspose1d（アップサンプリング用）。

    デコーダで使用。stride 倍のアップサンプリングを行う。
    ストリーミング時はキャッシュを使って前回の context_size 入力サンプルを保持。

    context_size = kernel_size - 1

    参照: modular_vibevoice_tokenizer.py の SConvTranspose1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_tr = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, bias=bias,
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.context_size = kernel_size - 1

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T] 入力
        Returns:
            out: [B, C_out, T * stride] アップサンプリング後
        """
        if not use_cache:
            # 非ストリーミング
            out = self.conv_tr(x)
            # パディング除去
            padding = self.kernel_size - self.stride
            out = out[..., padding // 2: -(padding - padding // 2)]
            return out
            # [B, C_out, T * stride]
        else:
            # ストリーミング
            cached = cache.get(layer_id, sample_indices)
            is_first = cached is None

            if cached is not None:
                x_full = torch.cat([cached, x], dim=-1)
            else:
                x_full = x

            out = self.conv_tr(x_full)
            # パディング除去
            padding = self.kernel_size - self.stride
            out = out[..., padding // 2: -(padding - padding // 2)]

            if is_first:
                result = out  # 初回は全出力
            else:
                # 2回目以降は新規部分のみ（T * stride サンプル）
                result = out[..., -x.shape[-1] * self.stride:]

            # キャッシュ更新
            cache.set(layer_id, sample_indices, x_full[..., -self.context_size:])

            return result


# ============================================================================
# 基本ブロック
# ============================================================================

class FFN(nn.Module):
    """
    Feed-Forward Network (GELU 活性化)

    参照: modular_vibevoice_tokenizer.py の FFN
    """

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim]
        Returns:
            [B, T, embed_dim]
        """
        return self.fc2(F.gelu(self.fc1(x)))  # [B, T, embed_dim]


class Block1D(nn.Module):
    """
    Tokenizer の基本残差ブロック。
    Conv ミキサー + FFN の2パスで構成。

    構造:
      パス1（Conv Mixer）: norm → depthwise_conv → γ × residual + x
      パス2（FFN）:        norm → permute → FFN → permute → γ × residual + x

    各パスに Layer Scale (γ) を適用し、初期値 1e-6 で安定した学習。

    参照: modular_vibevoice_tokenizer.py の Block1D
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        ffn_ratio: float = 4.0,
        causal: bool = True,
        layer_scale_init_value: float = 1e-6,
        pad_mode: str = 'constant',
    ):
        super().__init__()
        # パス1: Conv Mixer (Depthwise Conv)
        self.norm1 = ConvRMSNorm(dim)
        self.conv = SConv1d(
            dim, dim, kernel_size,
            causal=causal, pad_mode=pad_mode,
            # groups=dim にすると depthwise conv
        )
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

        # パス2: FFN
        self.norm2 = ConvRMSNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = FFN(dim, ffn_dim)
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, C, T]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T] 同じ形状
        """
        # --- パス1: Conv Mixer ---
        residual = self.conv(
            self.norm1(x), cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            is_final_chunk=is_final_chunk,
        )
        # [B, C, T]
        x = x + self.gamma1.unsqueeze(0).unsqueeze(-1) * residual

        # --- パス2: FFN ---
        normed = self.norm2(x)
        normed = normed.permute(0, 2, 1)    # [B, C, T] → [B, T, C]
        ffn_out = self.ffn(normed)           # [B, T, C]
        ffn_out = ffn_out.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        x = x + self.gamma2.unsqueeze(0).unsqueeze(-1) * ffn_out

        return x  # [B, C, T]


# ============================================================================
# TokenizerEncoder: 音声 → 潜在変数
# ============================================================================

class TokenizerEncoder(nn.Module):
    """
    階層的エンコーダ。7段階の Modified Transformer ブロックで構成。
    1D 深さ方向因果畳み込みで self-attention の代替を実現（ストリーミング対応）。

    構造:
      Stem Conv → [Downsample + Blocks] × 6 stages → Norm → Head Conv

    ダウンサンプリング比率 (reversed): [2, 2, 4, 5, 5, 8]
    → 累積: 2 × 2 × 4 × 5 × 5 × 8 = 3200倍
    → 24kHz / 3200 = 7.5 Hz

    各ステージのブロック深さ: "3-3-3-3-3-3-8" (最終ステージが最も深い)

    パラメータ: 約340M

    参照: modular_vibevoice_tokenizer.py の TokenizerEncoder
    """

    def __init__(
        self,
        channels: int = 1,         # 入力チャンネル (モノラル)
        n_filters: int = 32,       # ベースフィルタ数
        ratios: list = None,       # ダウンサンプリング比率 [8,5,5,4,2,2] → reversed
        depths: list = None,       # 各ステージのブロック数 [3,3,3,3,3,3,8]
        vae_dim: int = 64,         # 出力潜在次元
        causal: bool = True,       # 因果畳み込み
        kernel_size: int = 7,      # Block1D のカーネルサイズ
        pad_mode: str = 'constant',
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 5, 4, 2, 2]  # 設定値（reverse して使用）
        if depths is None:
            depths = [3, 3, 3, 3, 3, 3, 8]

        # encoder_ratios は reversed: [2, 2, 4, 5, 5, 8]
        self.ratios = list(reversed(ratios))
        self.num_stages = len(self.ratios)

        # === Stem ===
        self.stem = SConv1d(channels, n_filters, kernel_size=7, causal=causal)
        # [B, 1, T] → [B, 32, T]

        # === Downsample Stages ===
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        in_ch = n_filters  # 32
        for i, ratio in enumerate(self.ratios):
            out_ch = in_ch * 2  # チャンネル数を2倍に
            # ダウンサンプリング畳み込み
            self.downsample_layers.append(
                SConv1d(in_ch, out_ch, kernel_size=ratio * 2,
                        stride=ratio, causal=causal, pad_mode=pad_mode)
            )
            # Block1D × depth
            blocks = nn.ModuleList([
                Block1D(out_ch, kernel_size=kernel_size, causal=causal)
                for _ in range(depths[i])
            ])
            self.stages.append(blocks)
            in_ch = out_ch
            # Stage 0: 32→64, stride=2
            # Stage 1: 64→128, stride=2
            # Stage 2: 128→256, stride=4
            # Stage 3: 256→512, stride=5
            # Stage 4: 512→1024, stride=5
            # Stage 5: 1024→2048, stride=8

        # === 最終ブロック（depths[-1] = 8） ===
        self.final_blocks = nn.ModuleList([
            Block1D(in_ch, kernel_size=kernel_size, causal=causal)
            for _ in range(depths[-1])
        ])

        # === Norm + Head ===
        self.norm = ConvRMSNorm(in_ch)
        self.head = SConv1d(in_ch, vae_dim, kernel_size=1, causal=causal)
        # [B, 2048, T_latent] → [B, 64, T_latent]

    def forward(
        self,
        x: torch.Tensor,  # [B, 1, T_samples]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> torch.Tensor:
        """
        音声波形を潜在表現にエンコード。

        Args:
            x: [B, 1, T_samples] 24kHz モノラル波形

        Returns:
            out: [B, vae_dim, T_latent]
                 T_latent = ceil(T_samples / 3200) ≈ 7.5 tokens/sec

        データフロー:
            [B, 1, T_samples]
            → Stem: [B, 32, T]
            → Stage 0 (stride=2):  [B, 64, T/2]
            → Stage 1 (stride=2):  [B, 128, T/4]
            → Stage 2 (stride=4):  [B, 256, T/16]
            → Stage 3 (stride=5):  [B, 512, T/80]
            → Stage 4 (stride=5):  [B, 1024, T/400]
            → Stage 5 (stride=8):  [B, 2048, T/3200]
            → Final Blocks (×8):   [B, 2048, T/3200]
            → RMSNorm:            [B, 2048, T/3200]
            → Head:               [B, 64, T/3200]
        """
        x = self.stem(x, cache=cache, sample_indices=sample_indices,
                       use_cache=use_cache, is_final_chunk=is_final_chunk)

        for i in range(self.num_stages):
            x = self.downsample_layers[i](
                x, cache=cache, sample_indices=sample_indices,
                use_cache=use_cache, is_final_chunk=is_final_chunk,
            )
            for block in self.stages[i]:
                x = block(x, cache=cache, sample_indices=sample_indices,
                          use_cache=use_cache, is_final_chunk=is_final_chunk)

        for block in self.final_blocks:
            x = block(x, cache=cache, sample_indices=sample_indices,
                      use_cache=use_cache, is_final_chunk=is_final_chunk)

        x = self.norm(x)
        x = self.head(x, cache=cache, sample_indices=sample_indices,
                       use_cache=use_cache, is_final_chunk=is_final_chunk)
        return x  # [B, vae_dim, T_latent]


# ============================================================================
# TokenizerDecoder: 潜在変数 → 音声
# ============================================================================

class TokenizerDecoder(nn.Module):
    """
    階層的デコーダ。エンコーダの鏡像構造。
    ConvTranspose1d でアップサンプリングし、波形を復元。

    構造:
      Stem Conv → [Upsample + Blocks] × 6 stages → Norm → Head Conv

    アップサンプリング比率: [8, 5, 5, 4, 2, 2]
    → 累積: 8 × 5 × 5 × 4 × 2 × 2 = 3200倍
    → 7.5 Hz → 24kHz

    参照: modular_vibevoice_tokenizer.py の TokenizerDecoder
    """

    def __init__(
        self,
        channels: int = 1,
        n_filters: int = 32,
        ratios: list = None,
        depths: list = None,
        vae_dim: int = 64,
        causal: bool = True,
        kernel_size: int = 7,
        pad_mode: str = 'constant',
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 5, 4, 2, 2]
        if depths is None:
            depths = [8, 3, 3, 3, 3, 3, 3]  # エンコーダの reversed

        self.ratios = ratios  # [8, 5, 5, 4, 2, 2]
        self.num_stages = len(ratios)

        # 最大チャンネル数の計算
        max_ch = n_filters * (2 ** self.num_stages)  # 32 * 64 = 2048

        # === Stem ===
        self.stem = SConv1d(vae_dim, max_ch, kernel_size=7, causal=causal)
        # [B, 64, T_latent] → [B, 2048, T_latent]

        # === 初期ブロック（depths[0] = 8） ===
        self.initial_blocks = nn.ModuleList([
            Block1D(max_ch, kernel_size=kernel_size, causal=causal)
            for _ in range(depths[0])
        ])

        # === Upsample Stages ===
        self.upsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        in_ch = max_ch  # 2048
        for i, ratio in enumerate(ratios):
            out_ch = in_ch // 2
            self.upsample_layers.append(
                SConvTranspose1d(in_ch, out_ch, kernel_size=ratio * 2, stride=ratio)
            )
            blocks = nn.ModuleList([
                Block1D(out_ch, kernel_size=kernel_size, causal=causal)
                for _ in range(depths[i + 1])
            ])
            self.stages.append(blocks)
            in_ch = out_ch
            # Stage 0: 2048→1024, stride=8
            # Stage 1: 1024→512, stride=5
            # Stage 2: 512→256, stride=5
            # Stage 3: 256→128, stride=4
            # Stage 4: 128→64, stride=2
            # Stage 5: 64→32, stride=2

        # === Norm + Head ===
        self.norm = ConvRMSNorm(in_ch)  # in_ch = 32
        self.head = SConv1d(in_ch, channels, kernel_size=1, causal=causal)
        # [B, 32, T_audio] → [B, 1, T_audio]

    def forward(
        self,
        x: torch.Tensor,  # [B, vae_dim, T_latent]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        潜在表現から音声波形を復元。

        Args:
            x: [B, vae_dim, T_latent] (channels-first)
               T_latent = ceil(T_samples / 3200)

        Returns:
            out: [B, 1, T_samples] 復元された24kHz波形

        データフロー:
            [B, 64, T_latent]
            → Stem: [B, 2048, T_latent]
            → Initial Blocks (×8): [B, 2048, T_latent]
            → Stage 0 (stride=8):  [B, 1024, T_latent*8]
            → Stage 1 (stride=5):  [B, 512, T_latent*40]
            → Stage 2 (stride=5):  [B, 256, T_latent*200]
            → Stage 3 (stride=4):  [B, 128, T_latent*800]
            → Stage 4 (stride=2):  [B, 64, T_latent*1600]
            → Stage 5 (stride=2):  [B, 32, T_latent*3200]
            → RMSNorm:            [B, 32, T_audio]
            → Head:               [B, 1, T_audio]
        """
        x = self.stem(x, cache=cache, sample_indices=sample_indices,
                       use_cache=use_cache)

        for block in self.initial_blocks:
            x = block(x, cache=cache, sample_indices=sample_indices,
                      use_cache=use_cache)

        for i in range(self.num_stages):
            x = self.upsample_layers[i](
                x, cache=cache, sample_indices=sample_indices,
                use_cache=use_cache,
            )
            for block in self.stages[i]:
                x = block(x, cache=cache, sample_indices=sample_indices,
                          use_cache=use_cache)

        x = self.norm(x)
        x = self.head(x, cache=cache, sample_indices=sample_indices,
                       use_cache=use_cache)
        return x  # [B, 1, T_audio]


# ============================================================================
# エンコーダ出力（σ-VAE 分布）
# ============================================================================

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    """
    σ-VAE エンコーダの出力。固定分散のガウス分布を表す。

    Fields:
        mean: [B, T, vae_dim] 分布の平均（エンコーダの出力）
        std: float or Tensor 固定標準偏差

    サンプリング方式:
        'fix':      z = mean + std * randn_like(mean)  (固定 std)
        'gaussian': z = mean + per_sample_std * randn_like(mean)
        'none':     z = mean  (決定論的、Semantic Tokenizer 用)

    参照: modular_vibevoice_tokenizer.py の VibeVoiceTokenizerEncoderOutput
    """
    mean: torch.Tensor   # [B, T, vae_dim]
    std: float           # 固定 std (Acoustic: 0.5, Semantic: 0)

    def sample(self, dist_type: str = 'fix') -> Tuple[torch.Tensor, float]:
        """
        σ-VAE からサンプリング。

        数式: z = μ + σ ⊙ ε,  ε ~ N(0, 1)

        Args:
            dist_type: 'fix' | 'gaussian' | 'none'

        Returns:
            (sampled_z, std_used)
            sampled_z: [B, T, vae_dim]
        """
        if dist_type == 'fix':
            # 固定 std でサンプリング
            noise = torch.randn_like(self.mean)
            z = self.mean + self.std * noise
            return z, self.std

        elif dist_type == 'gaussian':
            # バッチごとに異なる std（N(0, C_σ) からサンプリング）
            B = self.mean.shape[0]
            batch_std = torch.randn(B, 1, 1, device=self.mean.device) * self.std
            batch_std = batch_std.abs()  # 正の値に
            noise = torch.randn_like(self.mean)
            z = self.mean + batch_std * noise
            return z, batch_std

        elif dist_type == 'none':
            # 決定論的（Semantic Tokenizer 用）
            return self.mean, self.std

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    def kl(self) -> torch.Tensor:
        """KL散逸: KL(q(z|x) || N(0,1))の簡易近似"""
        return F.mse_loss(self.mean, torch.zeros_like(self.mean))

    def mode(self) -> torch.Tensor:
        """分布のモード（= 平均）"""
        return self.mean


# ============================================================================
# Acoustic Tokenizer Model（完全な VAE）
# ============================================================================

class VibeVoiceAcousticTokenizerModel(nn.Module):
    """
    Acoustic Tokenizer: σ-VAE ベースの音声圧縮モデル。
    エンコーダとデコーダの両方を持ち、音声の圧縮・復元が可能。

    設定値:
        - channels: 1 (モノラル)
        - vae_dim: 64
        - fix_std: 0.5
        - std_dist_type: 'gaussian'
        - encoder_ratios: [8, 5, 5, 4, 2, 2] (3200倍ダウンサンプリング)
        - encoder_depths: "3-3-3-3-3-3-8"
        - layernorm: 'RMSNorm'
        - mixer_layer: 'depthwise_conv'

    学習: DAC [KSL+23] の判別器と損失関数に従う
    推論時は凍結（@torch.no_grad）

    参照: modular_vibevoice_tokenizer.py の VibeVoiceAcousticTokenizerModel
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 深さ解析: "3-3-3-3-3-3-8" → [3, 3, 3, 3, 3, 3, 8]
        encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        # デコーダ深さ: reversed → [8, 3, 3, 3, 3, 3, 3]
        decoder_depths = list(reversed(encoder_depths))

        self.encoder = TokenizerEncoder(
            channels=config.channels,           # 1
            n_filters=config.encoder_n_filters,  # 32
            ratios=config.encoder_ratios,        # [8, 5, 5, 4, 2, 2]
            depths=encoder_depths,               # [3, 3, 3, 3, 3, 3, 8]
            vae_dim=config.vae_dim,             # 64
            causal=config.causal,                # True
        )

        self.decoder = TokenizerDecoder(
            channels=config.channels,           # 1
            n_filters=config.decoder_n_filters,  # 32
            ratios=config.decoder_ratios,        # [8, 5, 5, 4, 2, 2]
            depths=decoder_depths,               # [8, 3, 3, 3, 3, 3, 3]
            vae_dim=config.vae_dim,             # 64
            causal=config.causal,                # True
        )

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,  # [B, 1, T_samples] or [B, T_samples]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> VibeVoiceTokenizerEncoderOutput:
        """
        音声波形をエンコードして σ-VAE 分布を返す。

        Args:
            audio: [B, 1, T_samples] 24kHz モノラル波形

        Returns:
            VibeVoiceTokenizerEncoderOutput
                mean: [B, T_latent, 64]
                std: 0.5

        入出力形状:
            [B, 1, T_samples]
            → Encoder: [B, 64, T_latent]
            → permute: [B, T_latent, 64]
            → VibeVoiceTokenizerEncoderOutput(mean=[B, T_latent, 64], std=0.5)
        """
        encoded = self.encoder(
            audio, cache=cache, sample_indices=sample_indices,
            use_cache=use_cache, is_final_chunk=is_final_chunk,
        )
        # [B, 64, T_latent]

        mean = encoded.permute(0, 2, 1)  # [B, T_latent, 64]

        return VibeVoiceTokenizerEncoderOutput(
            mean=mean,
            std=self.config.fix_std,  # 0.5
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,  # [B, T_latent, 64]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        潜在変数から音声波形を復元。

        Args:
            latents: [B, T_latent, 64]

        Returns:
            audio: [B, 1, T_samples]

        入出力形状:
            [B, T_latent, 64]
            → permute: [B, 64, T_latent]
            → Decoder: [B, 1, T_samples]
        """
        latents = latents.permute(0, 2, 1)  # [B, 64, T_latent] → [B, vae_dim, T_latent]
        audio = self.decoder(
            latents, cache=cache, sample_indices=sample_indices,
            use_cache=use_cache,
        )
        return audio  # [B, 1, T_samples]

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完全な VAE パス: encode → sample → decode

        Args:
            audio: [B, 1, T_samples]

        Returns:
            (reconstructed_audio, sampled_latents)
            reconstructed_audio: [B, 1, T_samples]
            sampled_latents: [B, T_latent, 64]
        """
        encoder_output = self.encode(audio, cache, sample_indices, use_cache)
        sampled, _ = encoder_output.sample(
            dist_type=self.config.std_dist_type  # 'gaussian'
        )
        reconstructed = self.decode(sampled, cache, sample_indices, use_cache)
        return reconstructed, sampled


# ============================================================================
# Semantic Tokenizer Model（エンコーダのみ）
# ============================================================================

class VibeVoiceSemanticTokenizerModel(nn.Module):
    """
    Semantic Tokenizer: 決定論的な意味特徴抽出モデル。
    エンコーダのみで構成（デコーダなし）。

    Acoustic Tokenizer と同じエンコーダ構造だが:
    - vae_dim: 128（Acousticの2倍）
    - fix_std: 0（サンプリングなし、決定論的出力）
    - std_dist_type: 'none'
    - デコーダなし

    学習: ASR プロキシタスクで学習。エンコーダ出力を Transformer Decoder で
    テキスト転写を予測するように訓練。学習後デコーダは破棄。

    参照: modular_vibevoice_tokenizer.py の VibeVoiceSemanticTokenizerModel
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder_depths = [int(d) for d in config.encoder_depths.split('-')]

        self.encoder = TokenizerEncoder(
            channels=config.channels,           # 1
            n_filters=config.encoder_n_filters,  # 32
            ratios=config.encoder_ratios,        # [8, 5, 5, 4, 2, 2]
            depths=encoder_depths,
            vae_dim=config.vae_dim,             # 128
            causal=config.causal,                # True
        )

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,  # [B, 1, T_samples]
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> VibeVoiceTokenizerEncoderOutput:
        """
        音声波形をエンコードして決定論的な意味特徴を返す。

        Args:
            audio: [B, 1, T_samples]

        Returns:
            VibeVoiceTokenizerEncoderOutput
                mean: [B, T_latent, 128]
                std: 0 (サンプリングなし)

        入出力形状:
            [B, 1, T_samples] → Encoder → [B, 128, T_latent] → permute → [B, T_latent, 128]
        """
        encoded = self.encoder(
            audio, cache=cache, sample_indices=sample_indices,
            use_cache=use_cache, is_final_chunk=is_final_chunk,
        )
        mean = encoded.permute(0, 2, 1)  # [B, T_latent, 128]

        return VibeVoiceTokenizerEncoderOutput(
            mean=mean,
            std=0,  # fix_std = 0 → 決定論的
        )

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[None, torch.Tensor]:
        """
        Semantic 特徴の抽出。

        Args:
            audio: [B, 1, T_samples]

        Returns:
            (None, semantic_latents)
            semantic_latents: [B, T_latent, 128] (決定論的)
        """
        encoder_output = self.encode(audio, cache, sample_indices, use_cache)
        # dist_type='none' → mean をそのまま返す
        sampled, _ = encoder_output.sample(dist_type='none')
        return None, sampled  # デコーダがないため reconstructed は None
