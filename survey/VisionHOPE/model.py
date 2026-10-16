"""
VisionHOPE - 4方向Operator・残差Block・階層バックボーン (メインフロー)
================================================================

``srnl.py`` の SRNL を4方向スキャン (→←↓↑) に適用する VisionHOPEOperator、
それを組み込んだ VisionHOPEBlock、4ステージの階層バックボーン
HierarchicalVisionHOPE を実装する。

論文: VisionHOPE (arXiv:2609.33325) Sec 3.3 VisionHOPE Architecture
      - Eq.22-24: 4方向シリアライズ・チャンク長・方向融合
      - Eq.25: ブロックの構造 (Norm1→Proj_in→DW→Operator→Proj_out, 残差,
               Norm2→FFN, 残差)
公式コード: gh/VisionHOPE/models/{operator,block,hierarchical,layers,common}.py
      これらは全て素のPyTorch実装であり (CUDA拡張はSRNL内部の再帰計算のみ)、
      本ファイルはその構造をほぼそのまま移植している。唯一の違いは、公式の
      ``VisionHOPEOperator`` が内部で ``models.srnl.SRNL`` (CUDA拡張ラッパー) を
      使うのに対し、本ファイルは ``srnl.py`` の ``SRNLReference`` (純PyTorch) を
      使う点。高速推論専用の融合カーネル分岐 (``fast_inference`` / CUDA Graph 用の
      ``_forward_fast``) は、学習時と数学的に同じ計算を行う純粋な速度最適化で
      あり、CUDA拡張のビルドを要する実装の詳細であるため省略する
      (省略理由: CUDA拡張が無いと動作確認不能、かつ同じ計算の高速化パスに過ぎない)。

============================================================
Shape Convention
============================================================
B      : バッチサイズ
C_in   : 入力チャンネル数 (画像は3)
D      : ブロックの入出力チャンネル数 (embed_dim)
D_m    : Operator内部の作業チャンネル数 (mixer_dim, D以下が一般的)
H, W   : 空間の高さ・幅
N      : H*W (トークン数)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from srnl import SRNLReference


# ============================================================
# 1. 正規化 / FFN / Downsample 層 (公式 models/layers.py を移植)
# ============================================================

class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (ConvNeXt V2 と同じ定式化、公式 utils/grn.py)。

    ========================================
    式
    ========================================
    g_x = ||x||_2 (空間方向, チャンネルごと)      -> (B, C, 1, 1)
    n_x = g_x / (mean_c(g_x) + eps)
    out = x * (1 + gamma * n_x) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)       # (B, C, 1, 1)
        return x * (1.0 + gamma * nx) + beta                      # (B, C, H, W)


class LayerNorm2d(nn.Module):
    """チャンネル方向のLayerNorm (NCHW入力, 内部でNHWCへpermuteして計算)。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C) -> LN over C -> (B, C, H, W)
        x = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class FeedForwardNetwork(nn.Module):
    """ConvベースのFFN (1x1→GELU→[DWConv+GELU残差]→GRN→1x1), 公式 models/layers.py。"""

    def __init__(self, in_features: int, hidden_features: int, use_grn: bool = True, use_dwconv: bool = True):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.use_dwconv = use_dwconv
        self.dwconv = (
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
            if use_dwconv else nn.Identity()
        )
        self.dwconv_act = nn.GELU() if use_dwconv else nn.Identity()
        self.grn = GlobalResponseNorm(hidden_features) if use_grn else nn.Identity()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        x = self.act(self.fc1(x))                    # (B, hidden, H, W)
        if self.use_dwconv:
            x = self.dwconv_act(x + self.dwconv(x))   # (B, hidden, H, W)
        x = self.grn(x)
        return self.fc2(x)                             # (B, D, H, W)


class PatchEmbedding(nn.Module):
    """stride-4の畳み込みステム (3x3 conv x4, stride=[2,1,2,1])。"""

    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        widths = [embed_dim // 2, embed_dim // 2, embed_dim, embed_dim]
        strides = [2, 1, 2, 1]
        layers = []
        c = in_chans
        for i, (w, s) in enumerate(zip(widths, strides)):
            layers += [nn.Conv2d(c, w, 3, stride=s, padding=1), nn.BatchNorm2d(w)]
            if i < 3:
                layers.append(nn.GELU())
            c = w
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_chans, H, W) -> (B, embed_dim, H/4, W/4)
        return self.proj(x)


class Downsample(nn.Module):
    """ステージ間ダウンサンプル (stride-2 3x3conv + BN)。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim, H, W) -> (B, out_dim, H/2, W/2)
        return self.norm(self.proj(x))


# ============================================================
# 2. VisionHOPEOperator: 4方向スキャン + 空間復元 + チャンネル融合
# ============================================================

class VisionHOPEOperator(nn.Module):
    """4方向SRNLスキャンを適用し、空間特徴マップへ復元・融合する (論文 Sec 3.3 Eq.22-24)。

    ========================================
    Shape
    ========================================
    入力: x (B, D_m, H, W)
    出力: y (B, D_m, H, W)  (形状保存)
    """

    def __init__(self, dim: int, head_dim: int = 16, chunk_size: int | None = None):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.W_q = nn.Linear(dim, dim)  # 論文 Eq.13: q_t = W_q x_t (Operatorが外部から保持)
        self.direction_scale = nn.Parameter(torch.ones(4, dim))  # lambda_r (論文Eq.24)
        self.srnl = SRNLReference(dim, head_dim, chunk_size=chunk_size or 64)
        nn.init.trunc_normal_(self.W_q.weight, std=0.02)
        nn.init.zeros_(self.W_q.bias)

    @staticmethod
    def _serialize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """4方向(→,←,↓,↑)にシリアライズする (論文Eq.22 の P_r)。

        ========================================
        Shape
        ========================================
        入力:  x (B, D_m, H, W)
        出力: 4つの (B, H*W, D_m) 系列 (順方向行優先, 逆方向行優先, 順方向列優先, 逆方向列優先)
        """
        B, D, H, W = x.shape
        seq_hw = x.flatten(2).transpose(1, 2)                              # (B, H*W, D): 行優先(→)
        seq_wh = x.transpose(2, 3).flatten(2).transpose(1, 2)               # (B, H*W, D): 列優先(↓)
        return seq_hw, seq_hw.flip(1), seq_wh, seq_wh.flip(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        if D != self.dim:
            raise ValueError(f"VisionHOPEOperator expects {self.dim} channels, got {D}")

        chunk_w = self.chunk_size or W   # 行方向スキャンのチャンク長 = W (論文 Eq.23)
        chunk_h = self.chunk_size or H   # 列方向スキャンのチャンク長 = H

        seq_hw = x.flatten(2).transpose(1, 2)  # (B, N, D), N=H*W
        q_hw = self.W_q(seq_hw)                 # (B, N, D): 論文Eq.13, 全スキャン共通のクエリ

        xs = self._serialize(x)                 # 4 x (B, N, D)
        qs = (q_hw, q_hw.flip(1),
              q_hw.view(B, H, W, D).transpose(1, 2).reshape(B, H * W, D),
              q_hw.view(B, H, W, D).transpose(1, 2).reshape(B, H * W, D).flip(1))

        # 行優先(→,←)は chunk_w、列優先(↓,↑)は chunk_h で別々にSRNLを走らせる
        # (公式実装は1回のバッチ呼び出しだが、本参照実装では明確さのため
        #  srnl.chunk_size を一時的に切り替えて方向ごとに呼び出す)。
        outs = []
        for i, (xr, qr) in enumerate(zip(xs, qs)):
            self.srnl.chunk_size = chunk_w if i < 2 else chunk_h
            outs.append(self.srnl(xr, qr))      # (B, N, D)

        # 空間復元: 系列 -> (B, D, H, W)
        out_hw_fwd = outs[0].view(B, H, W, D).permute(0, 3, 1, 2)
        out_hw_bwd = outs[1].flip(1).view(B, H, W, D).permute(0, 3, 1, 2)
        out_wh_fwd = outs[2].view(B, W, H, D).permute(0, 3, 2, 1)
        out_wh_bwd = outs[3].flip(1).view(B, W, H, D).permute(0, 3, 2, 1)

        scales = self.direction_scale.view(4, 1, D, 1, 1)
        y = (scales[0, 0] * out_hw_fwd + scales[1, 0] * out_hw_bwd
             + scales[2, 0] * out_wh_fwd + scales[3, 0] * out_wh_bwd)  # (B, D, H, W), 論文Eq.24
        return y


# ============================================================
# 3. VisionHOPEBlock: 2つの残差ブランチ (論文 Sec 3.3, Eq.25)
# ============================================================

class VisionHOPEBlock(nn.Module):
    """VisionHOPEの残差ブロック (論文 Eq.25)。

    ========================================
    Shape
    ========================================
    入力/出力: x (B, D, H, W) (形状保存)
    """

    def __init__(self, dim: int, mixer_dim: int | None = None, head_dim: int = 16,
                 chunk_size: int | None = None, mlp_ratio: float = 4.0,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        mixer_dim = mixer_dim or dim
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)   # Conditional Positional Encoding
        self.norm1 = LayerNorm2d(dim)
        self.proj_in = nn.Conv2d(dim, mixer_dim, 1)
        self.local_conv = nn.Conv2d(mixer_dim, mixer_dim, 7, padding=3, groups=mixer_dim)
        self.local_gate = nn.Conv2d(mixer_dim, mixer_dim, 1)        # gated depthwise conv (論文中のDW)
        self.operator = VisionHOPEOperator(mixer_dim, head_dim, chunk_size)
        self.local_position = nn.Conv2d(mixer_dim, mixer_dim, 5, padding=2, groups=mixer_dim)  # LePE
        self.proj_out = nn.Linear(mixer_dim, dim)

        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForwardNetwork(dim, int(dim * mlp_ratio))

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))

        nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def _local_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W) -> (B, D_m, H, W): proj_in + gated depthwise conv
        x_inner = self.proj_in(x)
        return x_inner + torch.sigmoid(self.local_gate(x_inner)) * self.local_conv(x_inner)

    def spatial_branch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W) -> Z=DW(Proj_in(x)) -> F_VH(Z) + LePE -> Proj_out -> (B, D, H, W) (論文Eq.25)
        x_local = self._local_features(x)                 # (B, D_m, H, W)
        out = self.operator(x_local)                        # (B, D_m, H, W)
        out = out + self.local_position(x_local)             # (B, D_m, H, W): Locally-enhanced PE
        out = out.permute(0, 2, 3, 1)                         # (B, H, W, D_m)
        out = self.proj_out(out)                              # (B, H, W, D)
        return out.permute(0, 3, 1, 2)                         # (B, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cpe(x)                                   # (B, D, H, W)
        branch = self.spatial_branch(self.norm1(x))
        x = x + self.gamma1 * branch                           # 残差1: Operator分岐
        branch = self.ffn(self.norm2(x))
        return x + self.gamma2 * branch                        # 残差2: FFN分岐


# ============================================================
# 4. HierarchicalVisionHOPE: 4ステージの分類バックボーン
# ============================================================

class HierarchicalVisionHOPE(nn.Module):
    """4ステージ階層VisionHOPEバックボーン + 分類ヘッド (公式 models/hierarchical.py)。

    ========================================
    Shape
    ========================================
    入力:  x (B, 3, H_img, W_img)  (例: 224x224)
    出力: logits (B, num_classes)
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000,
                 embed_dims=(64, 128, 256, 512), mixer_dims=(32, 64, 128, 256),
                 depths=(3, 4, 18, 4), head_dim: int = 16, chunk_sizes=None,
                 mlp_ratios=(4.0, 4.0, 4.0, 4.0), projection_dim: int = 1024):
        super().__init__()
        chunk_sizes = chunk_sizes or [None] * 4
        self.downsample_layers = nn.ModuleList([PatchEmbedding(in_chans, embed_dims[0])])
        for i in range(3):
            self.downsample_layers.append(Downsample(embed_dims[i], embed_dims[i + 1]))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                VisionHOPEBlock(
                    dim=embed_dims[i], mixer_dim=mixer_dims[i], head_dim=head_dim,
                    chunk_size=chunk_sizes[i], mlp_ratio=mlp_ratios[i],
                ) for _ in range(depths[i])
            ]))

        self.norm = LayerNorm2d(embed_dims[-1])
        self.head_pre_logits = nn.Conv2d(embed_dims[-1], projection_dim, 1)
        self.head_norm = nn.BatchNorm2d(projection_dim)
        self.head_act = nn.SiLU()
        self.head = nn.Linear(projection_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))     # (B, embed_dims[i], H/2^(i+2), W/2^(i+2))
        x = self.norm(x)
        x = self.head_act(self.head_norm(self.head_pre_logits(x)))  # (B, projection_dim, H', W')
        return x.mean(dim=[2, 3])                                      # (B, projection_dim): GAP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)  # (B, projection_dim)
        return self.head(features)             # (B, num_classes)


# 公式 models/factory.py の3サイズ設定 (チャンクサイズ以外は実測値そのまま)。
MODEL_CONFIGS = {
    "visionhope_tiny": dict(embed_dims=(64, 128, 256, 512),
                             mixer_dims=(32, 64, 128, 256), depths=(3, 4, 18, 4)),
    "visionhope_small": dict(embed_dims=(64, 160, 320, 512),
                              mixer_dims=(32, 80, 160, 256), depths=(4, 8, 25, 8)),
    "visionhope_base": dict(embed_dims=(96, 192, 448, 640),
                             mixer_dims=(48, 96, 224, 320), depths=(4, 8, 25, 8)),
}


def create_model(name: str, num_classes: int = 1000, **kwargs) -> HierarchicalVisionHOPE:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}; choose from {tuple(MODEL_CONFIGS)}")
    config = dict(MODEL_CONFIGS[name])
    config.update(kwargs)
    return HierarchicalVisionHOPE(num_classes=num_classes, **config)


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== model.py Example Usage ===\n")
    torch.manual_seed(0)

    # 実モデルは224x224を4段階ダウンサンプルするため H,W が32で割り切れる必要がある。
    # ここではCPUで素早く確認できるよう小さなカスタム構成を使う。
    model = HierarchicalVisionHOPE(
        in_chans=3, num_classes=10,
        embed_dims=(16, 32, 64, 128), mixer_dims=(8, 16, 32, 64),
        depths=(1, 1, 1, 1), head_dim=4, chunk_sizes=[None, None, None, None],
        projection_dim=64,
    )
    # 4ステージ(stem/4 + 3回の/2ダウンサンプル = 合計/32)を経るため、最終ステージの
    # 空間解像度が1x1まで縮退しないよう64x64を使う(1x1だとSRNLの逐次更新結果が
    # どの出力にも使われず、M_k/M_v等の勾配がNoneになる退化ケースになるため)。
    x = torch.randn(2, 3, 64, 64)
    print(f"入力 shape: {tuple(x.shape)}")

    logits = model(x)
    print(f"出力 logits shape: {tuple(logits.shape)}  (B=2, num_classes=10)")

    loss = F.cross_entropy(logits, torch.tensor([3, 7]))
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
    n_total = sum(1 for _ in model.parameters())
    print(f"\nloss={loss.item():.4f}")
    print(f"パラメータ総数={n_params:,}  勾配が流れたパラメータ数={n_with_grad}/{n_total}")

    print("\n--- 公式サイズ設定 (visionhope_tiny) をCPUで小さな入力で確認 ---")
    tiny_cfg = dict(MODEL_CONFIGS["visionhope_tiny"])
    tiny_cfg["depths"] = (1, 1, 1, 1)  # CPUで素早く確認するため層数だけ削減
    tiny = HierarchicalVisionHOPE(num_classes=1000, head_dim=16, **tiny_cfg)
    x2 = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits2 = tiny(x2)
    print(f"visionhope_tiny(depths削減) 出力: {tuple(logits2.shape)}  (B=1, num_classes=1000)")


if __name__ == "__main__":
    example_usage()
