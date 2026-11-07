"""CAMPPlusの話者embedding抽出器。畳み込み・統計pooling・投影をローカル実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=fbank frame数、F=80bin、C=中間特徴の次元数。入力[B,T,F]→話者embedding[B,192]。
"""
from __future__ import annotations

from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch
import torch.utils.checkpoint as cp


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """文字列の指定順にBatchNorm/ReLU等の非線形・正規化層を組み立てる。

    Args:
        config_str: BatchNormやReLU等を並べる順序の指定。
        channels: 畳み込みの特徴チャネル数。

    Returns:
        指定順に正規化・活性化を接続したnn.Sequential。
    """
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x: torch.Tensor,
                       dim: int = -1,
                       keepdim: bool = False,
                       unbiased: bool = True,
                       eps: float = 0.01) -> torch.Tensor:
    """時間特徴[B,C,T]の平均・標準偏差を連結する。

    T=時刻数、C=channel数。

    Args:
        x: 時間特徴[B,C,T]。Bは発話数、Cはチャネル数、Tはフレーム数。
        dim: 統計を計算する時間軸。既定は末尾軸。
        keepdim: 統計pooling後も集約した長さ1の軸を残すか。
        unbiased: 標準偏差に不偏分散の補正を使用するか。
        eps: 元の署名との互換用引数。現行関数の計算には使用しない。

    Returns:
        平均と標準偏差を連結した特徴[B,2C]。keepdim=Trueでは[B,2C,1]。Bは発話数、Cは入力チャネル数。
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    # 時間平均[B,C]と標準偏差[B,C]を特徴軸で結合する。[B,2C]。Bは発話数、Cは入力の特徴次元数。
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        # 時間統計の集約軸を長さ1で復元する。[B,2C]→[B,2C,1]。
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """時間方向の平均と標準偏差を特徴として連結する。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """時間方向の平均と標準偏差を特徴として連結する。

        特徴[B,C,T]→[B,2C]またはkeepdim時[B,2C,1]。C=channel数、T=時刻数。

        Args:
            x: 時間特徴[B,C,T]。Bは発話数、Cはチャネル数、Tはフレーム数。

        Returns:
            平均と標準偏差を連結した特徴[B,2C]。keepdim=Trueでは[B,2C,1]。Bは発話数、Cは入力チャネル数。
        """
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    """時間Convと非線形/正規化で音声局所文脈を抽出する。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 config_str: str = 'batchnorm-relu') -> None:
        """時間Convと非線形/正規化で音声局所文脈を抽出するための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            padding: 左右に追加する長さ。フィルタではpaddingの有効・無効。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            bias: 射影・畳み込みにバイアスを付けるか。
            config_str: BatchNormやReLU等を並べる順序の指定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, (
                'Expect equal paddings, but '
                'got even kernel size ({})').format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """時間Convと非線形/正規化で音声局所文脈を抽出する。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは入力時間長。

        Returns:
            特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、Toutは畳み込み後の時間長。
        """
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """局所Conv出力を大域/segment文脈のchannel gateで重み付けする。
    """

    def __init__(self,
                 bn_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int,
                 bias: bool,
                 reduction: int = 2) -> None:
        """局所Conv出力を大域/segment文脈のchannel gateで重み付けするための層と設定を構築する。

        Args:
            bn_channels: ボトルネック射影後のチャネル数。
            out_channels: 出力チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            padding: 左右に追加する長さ。フィルタではpaddingの有効・無効。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            bias: 射影・畳み込みにバイアスを付けるか。
            reduction: チャネルgateのボトルネック圧縮率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """局所Conv出力を大域/segment文脈のchannel gateで重み付けする。

        入力[B,Cin,T]→出力[B,Cout,T]。 B=発話数、Cin/Cout=特徴次元数、T=時間長。
        channel gateで局所出力を重み付けする。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。

        Returns:
            文脈gateで重み付けした特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、Toutは畳み込み後の時間長。
        """
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self,
                    x: torch.Tensor,
                    seg_len: int = 100,
                    stype: str = 'avg') -> torch.Tensor:
        """時間特徴[B,C,T]を固定長segmentでpoolし、同じsegment内へ再展開する。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。
            seg_len: segment poolingの1区間に含むフレーム数。
            stype: segment poolingの種類。avgまたはmax。

        Returns:
            各区間の統計を元の時刻へ展開した特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは入力時間長。
        """
        if stype == 'avg':
            seg = F.avg_pool1d(x,
                               kernel_size=seg_len,
                               stride=seg_len,
                               ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x,
                               kernel_size=seg_len,
                               stride=seg_len,
                               ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        # 区間ごとの統計を区間内の各時刻へ戻す。[B,C,Nseg,seg_len]→[B,C,Nseg*seg_len]。Nsegは区間数。
        seg = rearrange(
            seg.unsqueeze(-1).expand(*shape, seg_len), "b c t s -> b c (t s)")
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """bottleneck射影と文脈依存TDNNを適用する。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bn_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 config_str: str = 'batchnorm-relu',
                 memory_efficient: bool = False) -> None:
        """bottleneck射影と文脈依存TDNNを適用するための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            bn_channels: ボトルネック射影後のチャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            bias: 射影・畳み込みにバイアスを付けるか。
            config_str: BatchNormやReLU等を並べる順序の指定。
            memory_efficient: 中間活性を再計算するgradient checkpointingを使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, (
            'Expect equal paddings, but '
            'got even kernel size ({})').format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        """入力のBatchNorm/非線形とbottleneck Convを計算する。

        時間軸Tは維持する。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。

        Returns:
            ボトルネック特徴[B,Cbn,T]。Bは発話数、Cbnはボトルネックの特徴次元数、Tは時間長。
        """
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """bottleneck射影と文脈依存TDNNを適用する。

        入力[B,Cin,T]→bottleneck→出力[B,Cgrowth,T]。B=発話数、Cgrowth=新たに抽出するchannel数、
        T=時間長。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。

        Returns:
            新規特徴[B,Cgrowth,T]。Bは発話数、Cgrowthは追加チャネル数、Tは時間長。
        """
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """TDNN層の新規特徴をchannel方向へ逐次連結する。
    """

    def __init__(self,
                 num_layers: int,
                 in_channels: int,
                 out_channels: int,
                 bn_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 config_str: str = 'batchnorm-relu',
                 memory_efficient: bool = False) -> None:
        """TDNN層の新規特徴をchannel方向へ逐次連結するための層と設定を構築する。

        Args:
            num_layers: 積み重ねる層数。
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            bn_channels: ボトルネック射影後のチャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            bias: 射影・畳み込みにバイアスを付けるか。
            config_str: BatchNormやReLU等を並べる順序の指定。
            memory_efficient: 中間活性を再計算するgradient checkpointingを使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels +
                                      i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TDNN層の新規特徴をchannel方向へ逐次連結する。

        入力[B,Cin,T]→出力[B,Cin+layers*Cgrowth,T]。 各層の新規channelを連結し、時間長Tは維持する。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。

        Returns:
            連結特徴[B,Cin+N*Cgrowth,T]。Bは発話数、Nは層数、Cgrowthは各層の追加チャネル数、Tは時間長。
        """
        for layer in self:
            # 既存特徴へ新規特徴を追加する。[B,Ccurrent,T]と[B,Cgrowth,T]→
            # [B,Ccurrent+Cgrowth,T]。
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """蓄積した特徴channelを1x1射影で圧縮する。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True,
                 config_str: str = 'batchnorm-relu') -> None:
        """蓄積した特徴channelを1x1射影で圧縮するための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            bias: 射影・畳み込みにバイアスを付けるか。
            config_str: BatchNormやReLU等を並べる順序の指定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """蓄積した特徴channelを1x1射影で圧縮する。

        Args:
            x: 特徴[B,Cin,T]。Bは発話数、Cinは入力の特徴次元数、Tは時間長。

        Returns:
            圧縮特徴[B,Cout,T]。Bは発話数、Coutは出力の特徴次元数、Tは時間長。
        """
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """特徴の全結合射影と非線形/正規化を計算する。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = False,
                 config_str: str = 'batchnorm-relu') -> None:
        """特徴の全結合射影と非線形/正規化を計算するための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            bias: 射影・畳み込みにバイアスを付けるか。
            config_str: BatchNormやReLU等を並べる順序の指定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴の全結合射影と非線形/正規化を計算する。

        入力[B,Din]または[B,Din,T]を出力の特徴次元数Doutへ射影する。 B=発話数、T=時間長。
        channel-first入力では1x1 Convを使用する。

        Args:
            x: 特徴[B,Din]または[B,Din,T]。Bは発話数、Dinは入力の特徴次元数、Tは時間長。

        Returns:
            特徴[B,Dout]または[B,Dout,T]。Doutは射影後の特徴次元数。BとTは入力から維持する。
        """
        if len(x.shape) == 2:
            # 2次元の発話特徴へ長さ1の時間軸を足し、射影後に戻す。[B,Din]→[B,Din,1]→[B,Dout]。
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    """周波数/時間Convを残差接続して局所特徴を抽出する。
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """周波数/時間Convを残差接続して局所特徴を抽出するための層と設定を構築する。

        Args:
            in_planes: 入力の畳み込みチャネル数。
            planes: 残差ブロックの出力チャネル数。
            stride: 畳み込み窓を移動する間隔。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """周波数/時間Convを残差接続して局所特徴を抽出する。

        Args:
            x: 特徴[B,Cin,F,T]。Bは発話数、Cinは入力チャネル数、Fは周波数長、Tは時間長。

        Returns:
            残差特徴[B,Cout,Fout,T]。Bは発話数、Coutは出力チャネル数、Foutは圧縮後周波数長、Tは時間長。
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
