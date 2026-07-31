"""Native Vision (MoonViT-V2) -- Kimi K3 論文 §2.4 (sec:native-vision) の実装。

Kimi K3 はテキスト・画像・動画を単一の共有バックボーンで処理するネイティブ
マルチモーダルモデルであり、事後的なモダリティ整合ステージを持たない (§2.4 冒頭)。
最大の特徴は、Vision Encoder (MoonViT-V2) を **contrastive事前学習 (SigLIP等) を
一切使わず、next-token predictionのみでゼロから学習する** 点である。これは
SigLIP初期化 (先代 MoonViT-3D) が LLM との共同最適化で不安定になる (勾配ノルムの
スパイクが頻発する, Fig.vitgradnorm) ことへの対処であり、結果として MoonViT-V2 は
SigLIP初期化ベースラインと同等の視覚性能を達成する (§2.4 "Architecture" 直前)。

アーキテクチャ (§2.4 "Architecture" 段落 + 公式実装 modeling_kimi_k3.py の
MoonViT3dPretrainedModel 系クラスに準拠):
    画像/動画 --(patchify)--> パッチ列 --(Conv2d patch embed + 2D位置埋め込み)-->
    ViT (27層, RMSNorm, bias無し, 2D RoPE) --(2x2 pixel-shuffle + 時間方向pooling)-->
    MLP Projector --> LLM 入力埋め込みに混合

このファイルは公式コード (modeling_kimi_k3.py) のクラス構成を単純化した
再実装であり、flash-attention 依存部分は標準の scaled_dot_product_attention に
置き換えている (計算内容は同一、カーネル選択のみの違い)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KimiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def navit_patchify(pixel_values: torch.Tensor, patch_size: int):
    """可変解像度画像をパッチ列に変換する (NaViT 方式、公式 media_utils.navit_patchify に対応)。

    Args:
        pixel_values: (T, H, W, 3)  1枚 (or 1動画) 分の正規化済み画素値
        patch_size: パッチ1辺のピクセル数 (K3 実値: 14)
    Returns:
        patches:  (T * H/P * W/P, 3, P, P)  パッチごとに切り出した画素ブロック
        grid_thw: (3,) = [T, H/P, W/P]  時間・高さ・幅のパッチグリッドサイズ
    """
    T, H, W, C = pixel_values.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H, W は patch_size の倍数である必要がある"
    x = pixel_values.view(T, H // P, P, W // P, P, C)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # (T, H/P, W/P, C, P, P)
    patches = x.view(-1, C, P, P)
    grid_thw = torch.tensor([T, H // P, W // P], dtype=torch.long)
    return patches, grid_thw


class PatchEmbed(nn.Module):
    """パッチ -> トークン埋め込み。1パッチ = (3, P, P) を Conv2d (kernel=stride=P) で
    1ベクトル (out_dim,) に写像する。これは Linear(3*P*P, out_dim) と等価だが、
    公式実装に合わせて Conv2d で実装する。
    """

    def __init__(self, out_dim: int, patch_size: int = 14, in_channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (N_patches, 3, P, P)
        Returns:
            (N_patches, out_dim)
        """
        return self.proj(patches).flatten(1)  # (N_patches, out_dim, 1, 1) -> (N_patches, out_dim)


class Rope2D(nn.Module):
    """2D 回転位置埋め込み (公式実装 Rope2DPosEmbRepeated の簡略版)。

    ヘッド次元を半分ずつ高さ軸・幅軸に割り当て、各パッチの (h, w) グリッド座標を
    回転角に変換する。動画の場合は各フレームに同じ2D位置を繰り返し適用する
    (時間方向は §2.4 の「時間的プーリング」で圧縮されるため、RoPE自体は空間のみ)。
    """

    def __init__(self, head_dim: int, max_grid: int = 256, theta_base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.max_grid = max_grid
        dim_range = torch.arange(0, head_dim, 4)[: head_dim // 4].float()
        self.register_buffer("inv_freq", 1.0 / (theta_base ** (dim_range / head_dim)), persistent=False)

    def get_freqs_cis(self, grid_thw: torch.Tensor, device) -> torch.Tensor:
        """
        Args:
            grid_thw: (3,) = [T, H, W]
        Returns:
            freqs_cis: (T*H*W, head_dim//2) complex64
        """
        T, H, W = grid_thw.tolist()
        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()
        h_freqs = torch.outer(h_pos, self.inv_freq.to(device))  # (H, head_dim//4)
        w_freqs = torch.outer(w_pos, self.inv_freq.to(device))  # (W, head_dim//4)
        h_cis = torch.polar(torch.ones_like(h_freqs), h_freqs)  # (H, head_dim//4)
        w_cis = torch.polar(torch.ones_like(w_freqs), w_freqs)  # (W, head_dim//4)
        grid_cis = torch.cat(
            [h_cis[:, None, :].expand(H, W, -1), w_cis[None, :, :].expand(H, W, -1)], dim=-1
        )  # (H, W, head_dim//2)
        grid_cis = grid_cis.reshape(H * W, -1)
        return grid_cis.unsqueeze(0).expand(T, -1, -1).reshape(-1, grid_cis.shape[-1])  # (T*H*W, head_dim//2)

    @staticmethod
    def apply(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """x: (N, num_heads, head_dim), freqs_cis: (N, head_dim//2) -> (N, num_heads, head_dim)"""
        x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(1)).flatten(-2)
        return out.type_as(x)


class MoonViTLayer(nn.Module):
    """MoonViT-V2 の1ブロック (Pre-RMSNorm + RoPE付きMHA + Pre-RMSNorm + GELU-MLP, bias無し)。"""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = KimiRMSNorm(hidden_dim)
        self.norm2 = KimiRMSNorm(hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_patches, hidden_dim)  1つの forward 内で複数画像/フレームをconcatして渡す
               (可変解像度対応のため、バッチ次元を持たずトークン軸に平坦化して扱う)
            freqs_cis: (N_patches, head_dim//2)
        Returns:
            (N_patches, hidden_dim)
        """
        residual = x
        h = self.norm1(x)
        qkv = self.wqkv(h).view(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (N, num_heads, head_dim)
        q = Rope2D.apply(q, freqs_cis)
        k = Rope2D.apply(k, freqs_cis)

        # (N, num_heads, head_dim) -> (1, num_heads, N, head_dim) : SDPAのバッチ次元に合わせる
        q_, k_, v_ = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q_, k_, v_)  # 画像内は全パッチ相互に attend (causalでない)
        attn = attn.squeeze(0).transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        x = residual + self.wo(attn)

        x = x + self.mlp(self.norm2(x))
        return x


def pixel_shuffle_temporal_pool(x: torch.Tensor, grid_thw: torch.Tensor, merge: tuple[int, int] = (2, 2)):
    """§2.4: 「2x2 pixel-shuffle による空間4倍ダウンサンプル + 時間方向プーリング」。

    Args:
        x: (T*H*W, d)  ViT出力トークン列 (1枚の画像/動画分)
        grid_thw: (3,) = [T, H, W]
        merge: 空間方向の圧縮カーネル (kh, kw)
    Returns:
        (H/kh * W/kw, kh*kw*d)  空間4倍圧縮 + 時間方向は平均プーリングで1枚に集約
        (この後 MLP Projector で d_llm 次元へ写像される)
    """
    T, H, W = grid_thw.tolist()
    kh, kw = merge
    d = x.shape[-1]
    seq = x.view(T, H // kh, kh, W // kw, kw, d)
    seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous()  # (T, H/kh, W/kw, kh, kw, d)
    seq = seq.mean(dim=0)  # 時間方向プーリング (動画の複数フレームを1トークン格子に集約): (H/kh, W/kw, kh, kw, d)
    seq = seq.view((H // kh) * (W // kw), kh * kw * d)
    return seq


class MLPProjector(nn.Module):
    """ViT特徴 (kh*kw*d_v 次元, pixel-shuffle済み) を LLM の隠れ次元 d_llm へ写像する。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, eps: float = 1e-5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, bias=False),
        )
        self.post_norm = KimiRMSNorm(out_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_norm(self.proj(x))


class MoonViTV2(nn.Module):
    """MoonViT-V2 本体 (ViT + Projector)。K3 実値: 27層, hidden=1024, heads=12, patch=14。

    形状の記法:
        N_patches   : 全画像/フレームのパッチ総数 (可変解像度対応のため単一トークン軸に平坦化)
        d_v         : Vision Encoder 隠れ次元 (K3実値 1024)
        d_llm       : LLM 隠れ次元 (K3実値 7168)
    """

    def __init__(
        self,
        patch_size: int = 14,
        hidden_dim: int = 1024,
        num_layers: int = 27,
        num_heads: int = 12,
        mlp_dim: int = 4096,
        merge: tuple[int, int] = (2, 2),
        llm_hidden_dim: int = 7168,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.merge = merge
        self.patch_embed = PatchEmbed(hidden_dim, patch_size)
        self.rope = Rope2D(hidden_dim // num_heads)
        self.layers = nn.ModuleList(
            [MoonViTLayer(hidden_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.final_norm = KimiRMSNorm(hidden_dim)
        self.projector = MLPProjector(
            in_dim=hidden_dim * merge[0] * merge[1], hidden_dim=hidden_dim, out_dim=llm_hidden_dim
        )

    def forward(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            images: 長さ num_images のリスト。各要素は正規化済み画素値 (T, H, W, 3)
                (T=1 なら静止画、T>1 なら動画のフレーム列)
        Returns:
            visual_tokens: 長さ num_images のリスト。各要素は (N_v_i, d_llm)
                (N_v_i = (H/2)*(W/2)、動画は時間方向プーリングにより1グリッドへ圧縮済み)
        """
        outputs = []
        for image in images:
            patches, grid_thw = navit_patchify(image, self.patch_size)  # (T*H'*W', 3, P, P), (3,)
            x = self.patch_embed(patches)  # (T*H'*W', d_v)
            freqs_cis = self.rope.get_freqs_cis(grid_thw, x.device)  # (T*H'*W', d_v/num_heads/2)
            for layer in self.layers:
                x = layer(x, freqs_cis)
            x = self.final_norm(x)  # (T*H'*W', d_v)

            merged = pixel_shuffle_temporal_pool(x, grid_thw, self.merge)  # (H'/2 * W'/2, 4*d_v)
            visual_tokens = self.projector(merged)  # (H'/2 * W'/2, d_llm)
            outputs.append(visual_tokens)
        return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    vit = MoonViTV2(
        patch_size=14, hidden_dim=32, num_layers=2, num_heads=4, mlp_dim=64, llm_hidden_dim=48
    )

    # 静止画 (T=1, H=W=28 = 2patch x 2patch) と、3フレームの動画 (T=3, H=W=28) を混在させる
    image = torch.randn(1, 28, 28, 3)
    video = torch.randn(3, 28, 28, 3)
    outs = vit([image, video])

    for i, o in enumerate(outs):
        print(f"visual_tokens[{i}] shape:", o.shape)
    # image: H'=W'=2 patches -> merge(2,2) で 1x1 グリッド -> (1, 48)
    # video: 3フレーム同一グリッドを時間方向プーリングで1枚に圧縮 -> (1, 48) (フレーム数に依存しない)
    assert outs[0].shape == (1, 48)
    assert outs[1].shape == (1, 48)
    print("MoonViT-V2 OK")
