"""
Swin-style Window Attention (FlexAttention backend)

- 入力形式: (B, T, D) の一般的な系列（画像パッチ・音声・テキストなど何でもよい）
- window内だけでattentionを計算する `WindowAttention` と、
  no-shift -> shift の2回分をまとめて1ブロックとして扱う `SwinWindowBlock` を提供する
- 実体は torch.nn.attention.flex_attention の block-sparse kernel。
  マスクを渡すだけの実装(F.scaled_dot_product_attention + attn_mask)と違い、
  window外のブロックは実際に計算がスキップされ、計算量削減の恩恵を受けられる
- 要件: PyTorch >= 2.5 (torch.nn.attention.flex_attention が必要)
- 計算量削減の恩恵は主にCUDA backendで有効。CPUでも動作はするが速度上のメリットは薄い
- 重要: use_compile=True (torch.compile(flex_attention)) にしないと、
  block-sparseなfused kernelではなくfull scoresを一度materializeするフォールバック実装が使われ、
  正しく動くが計算量削減の恩恵は得られない。速度目的で使う場合は必ず use_compile=True にすること

使い方:
    # 単体のwindow attention (shift_size=0: 通常window, shift_size>0: shifted window)
    attn = WindowAttention(dim=256, num_heads=8, window_size=8, shift_size=0)
    y = attn(x)  # x: (B, T, D) -> (B, T, D)

    # Swinのように W-MSA -> SW-MSA を1ブロックとして扱いたい場合
    block = SwinWindowBlock(dim=256, num_heads=8, window_size=8)
    y = block(x)
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_compiled_flex_attention = None


def _get_flex_attention(use_compile: bool):
    """torch.compile版flex_attentionをプロセス内で使い回すための取得関数。"""
    global _compiled_flex_attention
    if not use_compile:
        return flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def _make_window_mask_mod(window_size: int, shift: int, seq_len: int, valid_len: int):
    """
    shift付きwindow所属判定。

    (idx + shift) % seq_len を使うことで、実データをtorch.rollせずに
    「rollしてから通常window partitionした場合」と同じグルーピング(wrap-around込み)を再現する。
    valid_len は window_size に満たない端数をpaddingした際、padding分をkeyから除外するために使う。
    """

    def mask_mod(b, h, q_idx, kv_idx):
        q_bucket = ((q_idx + shift) % seq_len) // window_size
        kv_bucket = ((kv_idx + shift) % seq_len) // window_size
        same_window = q_bucket == kv_bucket
        kv_valid = kv_idx < valid_len
        return same_window & kv_valid

    return mask_mod


class WindowAttention(nn.Module):
    """
    Swin-styleのwindowed multi-head self-attention (1回分)。

    shift_size=0            -> 通常のwindow attention (W-MSA)
    shift_size=window_size//2 -> shifted window attention (SW-MSA)

    T が window_size で割り切れない場合は末尾をzero-paddingして計算し、
    padding部分はkeyとして参照されないようマスクした上で、出力時に切り詰めて返す。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        use_compile: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert 0 <= shift_size < window_size, "shift_size must satisfy 0 <= shift_size < window_size"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim**-0.5
        self.use_compile = use_compile

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._block_mask_cache = {}  # (T_pad, valid_len, device) -> BlockMask

    def _get_block_mask(self, seq_len_padded: int, valid_len: int, device: torch.device):
        key = (seq_len_padded, valid_len, device)
        if key not in self._block_mask_cache:
            mask_mod = _make_window_mask_mod(self.window_size, self.shift_size, seq_len_padded, valid_len)
            self._block_mask_cache[key] = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=seq_len_padded, KV_LEN=seq_len_padded, device=device
            )
        return self._block_mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        W = self.window_size

        pad_len = (W - T % W) % W
        x_padded = nn.functional.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
        T_pad = T + pad_len

        qkv = self.qkv(x_padded)  # (B, T_pad, 3*D)
        q, k, v = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.num_heads
        ).unbind(0)  # each: (B, num_heads, T_pad, head_dim)

        block_mask = self._get_block_mask(T_pad, valid_len=T, device=x.device)
        attn_fn = _get_flex_attention(self.use_compile)
        out = attn_fn(q, k, v, block_mask=block_mask, scale=self.scale)  # (B, num_heads, T_pad, head_dim)

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.proj_drop(self.proj(out))
        return out[:, :T, :] if pad_len > 0 else out


class SwinWindowBlock(nn.Module):
    """
    W-MSA -> SW-MSA の2段構成を1ブロック(1クラス)にまとめたTransformer block。
    Swinの「shift無し層 + shift有り層」のペアをそのまま1単位として積み重ねたい場合に使う。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        use_compile: bool = False,
    ):
        super().__init__()
        shift = window_size // 2

        self.norm1 = nn.LayerNorm(dim)
        self.w_msa = WindowAttention(
            dim, num_heads, window_size, shift_size=0, qkv_bias=qkv_bias, proj_drop=proj_drop, use_compile=use_compile
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = _mlp(dim, mlp_ratio)

        self.norm3 = nn.LayerNorm(dim)
        self.sw_msa = WindowAttention(
            dim, num_heads, window_size, shift_size=shift, qkv_bias=qkv_bias, proj_drop=proj_drop, use_compile=use_compile
        )
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = _mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        x = x + self.w_msa(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.sw_msa(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))
        return x


def _mlp(dim: int, mlp_ratio: float) -> nn.Sequential:
    hidden = int(dim * mlp_ratio)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- mask_modのロジック単体を、naiveなroll+partitionと突き合わせて検証 ---
    def naive_same_window(seq_len, window_size, shift):
        idx = torch.arange(seq_len)
        bucket = ((idx + shift) % seq_len) // window_size
        return bucket.unsqueeze(0) == bucket.unsqueeze(1)

    seq_len, window_size, shift = 16, 4, 2
    mask_mod = _make_window_mask_mod(window_size, shift, seq_len, valid_len=seq_len)
    q_idx, kv_idx = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len), indexing="ij")
    got = mask_mod(None, None, q_idx, kv_idx)
    expected = naive_same_window(seq_len, window_size, shift)
    assert torch.equal(got, expected), "shifted window mask logic mismatch"
    print("[OK] shifted window mask matches naive roll+partition grouping")

    # --- 動作確認: 単体のWindowAttention (shift無し/有り) ---
    B, T, D, heads = 2, 37, 64, 8  # Tはwindow_sizeで割り切れないケースも試す
    x = torch.randn(B, T, D)

    attn_no_shift = WindowAttention(dim=D, num_heads=heads, window_size=8, shift_size=0)
    y1 = attn_no_shift(x)
    print(f"WindowAttention (no shift) : in={tuple(x.shape)} -> out={tuple(y1.shape)}")

    attn_shift = WindowAttention(dim=D, num_heads=heads, window_size=8, shift_size=4)
    y2 = attn_shift(x)
    print(f"WindowAttention (shift=4)  : in={tuple(x.shape)} -> out={tuple(y2.shape)}")

    # --- 動作確認: W-MSA + SW-MSA をまとめたSwinWindowBlock ---
    block = SwinWindowBlock(dim=D, num_heads=heads, window_size=8)
    y3 = block(x)
    print(f"SwinWindowBlock             : in={tuple(x.shape)} -> out={tuple(y3.shape)}")

    total_params = sum(p.numel() for p in block.parameters())
    print(f"SwinWindowBlock params: {total_params:,}")
