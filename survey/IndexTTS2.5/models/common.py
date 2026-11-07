"""有効長maskとWaveNetのtanh/sigmoid gateを基本演算で計算する。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

import torch


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor,
                                    input_b: torch.Tensor,
                                    n_channels: torch.Tensor) -> torch.Tensor:
    """入力2本[B,2C,T]を加算し、tanh側とsigmoid側に分けて積[B,C,T]を返す。

    Args:
        input_a: gate入力[B,2C,T]。Bは発話数、Cは出力チャネル数、Tは時間長。
        input_b: 加算条件[B,2C,T]。Bは発話数、Cは出力チャネル数、Tは時間長。
        n_channels: gate出力のチャネル数を先頭要素に持つ整数Tensor[1]。

    Returns:
        gate適用後の特徴[B,C,T]。Bは発話数、Cはn_channels[0]、Tは時間長。
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    # gateの値側と係数側をチャネル軸で等分する。[B,2C,T]→各[B,C,T]。
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts


def sequence_mask(
        length: torch.Tensor,
        max_length: int | torch.Tensor | None = None) -> torch.Tensor:
    """有効長[B]からbool mask[B,T]を作る。

    B=発話数、T=最大有効長。

    Args:
        length: 各発話の有効時間長[B]。Bは発話数。
        max_length: 生成するmaskの位置数。整数またはscalar Tensor[]。Noneでは入力長の最大値。
            []は軸を持たないscalarを表す。

    Returns:
        有効位置がTrueのmask[B,Tmax]。Bは発話数、Tmaxはmax_lengthまたは最大有効長。
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """各発話の有効長からpadding位置がTrueのmaskを作る。

    Args:
        lengths: 各発話の有効時間長[B]。Bは発話数。
        max_len: 位置表またはmaskの最大位置数。

    Returns:
        padding位置がTrueのmask[B,Tmax]。Bは発話数、Tmaxはmax_lenまたは最大有効長。
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
