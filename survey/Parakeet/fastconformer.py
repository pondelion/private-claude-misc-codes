"""FastConformer encoderを構成要素ごとに追うためのPyTorch実装。

公式NeMoのproduction実装から、論文理解に重要な8倍subsampling、Macaron FFN、
self-attention、depthwise convolutionを抽出している。relative-position attentionや
streaming cacheは基盤コードが大きいため通常のMHAに置換しており、その境界はREADMEに記す。
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def length_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    """``lengths (B,)`` -> valid mask ``(B,T=maximum)``."""
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


class DepthwiseSubsampling(nn.Module):
    """3段のstride-2 depthwise separable convolution。

    入力 ``(B,F,T)`` を出力 ``(B,ceil(T/8),D)`` へ変える。各段は、channelごとの
    temporal convolution（depthwise）とchannelを混ぜる1x1 convolution（pointwise）からなる。
    通常convよりparameter/FLOPsを減らし、attention前に系列長も1/8へ削減する。
    """
    def __init__(self, feat_in: int = 128, channels: int = 256, d_model: int = 512):
        super().__init__()
        layers, cin = [], feat_in
        for _ in range(3):
            # groups=cinにより、各入力channelを独立に時間方向へ畳み込む。
            layers += [nn.Conv1d(cin, cin, 5, stride=2, padding=2, groups=cin), nn.Conv1d(cin, channels, 1), nn.SiLU()]
            cin = channels
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv1d(channels, d_model, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Conv1dは(B,C,T)、attentionは(B,T,D)を期待するため最後にtransposeする。
        x = self.out(self.net(x)).transpose(1, 2)  # (B,T8,D)
        for _ in range(3):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return x, lengths.clamp_max(x.shape[1])


class FeedForward(nn.Module):
    """Conformerのposition-wise FFN: ``(B,T,D)->(B,T,4D)->(B,T,D)``。

    各時刻を独立に変換し、時間方向の情報交換はattention/convへ任せる。
    block前後に1つずつ配置し、それぞれresidualへ0.5倍して加える。
    """
    def __init__(self, d_model: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, expansion*d_model), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(expansion*d_model, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class ConvolutionModule(nn.Module):
    """局所音響patternを捉えるConformer convolution module。

    ``(B,T,D) -> LN -> pointwise(2D) -> GLU(D) -> depthwise temporal conv
    -> BatchNorm -> SiLU -> pointwise(D) -> (B,T,D)``。GLUは2Dをvalue/gateへ
    二分し、学習されたgateで情報を選別する。
    """
    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = .1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.point_in = nn.Conv1d(d_model, 2*d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.point_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # LayerNormは最終D軸、Conv1dはchannel軸を第2軸に要求する。
        y = F.glu(self.point_in(self.norm(x).transpose(1, 2)), dim=1)
        y = self.point_out(F.silu(self.bn(self.depthwise(y))))
        return self.dropout(y.transpose(1, 2))


class FastConformerBlock(nn.Module):
    """Macaron型Conformer block。全中間shapeは``(B,T,D)``。

    padding_maskは``(B,T)``で、Trueが無効位置。attentionのkey/valueからpaddingを除き、
    最後にpadding出力を0へ戻すことで後段へ不要な値を伝播させない。
    """
    def __init__(self, d_model=512, n_heads=8, ff_expansion=4, kernel_size=9, dropout=.1):
        super().__init__()
        self.ff1, self.ff2 = (FeedForward(d_model, ff_expansion, dropout) for _ in range(2))
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.final_norm = nn.LayerNorm(d_model)
    def forward(self, x, padding_mask):
        # Macaron FFN: 2つのFFNを0.5ずつ加え、合計寄与を通常の1 FFN程度に保つ。
        x = x + .5*self.ff1(x)
        y = self.attn_norm(x)
        # Q=K=Vのbidirectional self-attention。weight自体は不要なので返さない。
        y = self.attn(y, y, y, key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + self.attn_drop(y)
        x = x + self.conv(x)
        return self.final_norm(x + .5*self.ff2(x)).masked_fill(padding_mask[:, :, None], 0)


class FastConformerEncoder(nn.Module):
    """Log-Mel ``(B,128,T)`` -> encoded ``(B,T_enc,D)``。

    ``T_enc≈ceil(T/8)``。論文の0.6B-v3は24層だが、constructorで層数を変えられるため、
    ``main_flow.py``ではCPU検証可能な小型構成を使う。
    """
    def __init__(self, feat_in=128, d_model=512, n_layers=17, n_heads=8, kernel_size=9, dropout=.1):
        super().__init__()
        self.subsampling = DepthwiseSubsampling(feat_in, min(256, d_model), d_model)
        self.layers = nn.ModuleList([FastConformerBlock(d_model, n_heads, 4, kernel_size, dropout) for _ in range(n_layers)])
    def forward(self, features, lengths):
        # 有効長もconvのstrideに合わせて更新し、padding maskを一度だけ構築する。
        x, lengths = self.subsampling(features, lengths)
        padding = ~length_mask(lengths, x.shape[1])
        for layer in self.layers: x = layer(x, padding)
        return x, lengths
