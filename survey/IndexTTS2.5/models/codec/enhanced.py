"""semantic codec全体の局所実装。外部EnhancedCodecを呼ばない。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from typing import Any
import logging
import os

from torch import nn
from torch.nn import functional as F
import torch

from models.codec.blocks import VocosBackbone
from models.codec.quantization import ResidualVQ

logger = logging.getLogger(__name__)


def init_weights(m: Any) -> None:
    """Conv/Linearの重みとbiasをcodecの初期化規則で設定する。

    Args:
        m: 初期化対象のPyTorchモジュール。

    Returns:
        None。対象モジュールの重み・バイアスを更新する。
    """
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class EnhancedCodec(nn.Module):
    """時間圧縮、Vocos、FVQ、Vocos復元、時間拡大を順に計算する。
    """

    def __init__(self,
                 codebook_size: int = 8192,
                 hidden_size: int = 1024,
                 codebook_dim: int = 8,
                 vocos_dim: int = 384,
                 vocos_intermediate_dim: int = 2048,
                 vocos_num_layers: int = 12,
                 num_quantizers: int = 1,
                 downsample_scale: int | None = 2,
                 cfg: Any = None) -> None:
        """時間圧縮、Vocos、FVQ、Vocos復元、時間拡大を順に計算するための層と設定を構築する。

        codebook_size=codebookの語彙数、codebook_dim=量子化検索空間の次元数。

        Args:
            codebook_size: 量子化codebookに登録する離散IDの個数。
            hidden_size: 内部表現の特徴次元数。
            codebook_dim: 最近傍検索とcodebook埋め込みの次元数。
            vocos_dim: Vocos内部の特徴次元数。
            vocos_intermediate_dim: Vocos内FFNの中間次元数。
            vocos_num_layers: VocosのConvNeXt層数。
            num_quantizers: 残差量子化器の個数。
            downsample_scale: codecの時間圧縮設定。1以下またはNoneでは圧縮しない。
            cfg: 元の署名との互換用の追加設定。現行計算には使用しない。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        codebook_size = (cfg.codebook_size if cfg is not None
                         and hasattr(cfg, "codebook_size") else codebook_size)
        codebook_dim = (cfg.codebook_dim if cfg is not None
                        and hasattr(cfg, "codebook_dim") else codebook_dim)
        hidden_size = (cfg.hidden_size if cfg is not None
                       and hasattr(cfg, "hidden_size") else hidden_size)
        vocos_dim = (cfg.vocos_dim if cfg is not None
                     and hasattr(cfg, "vocos_dim") else vocos_dim)
        vocos_intermediate_dim = (cfg.vocos_intermediate_dim if cfg is not None
                                  and hasattr(cfg, "vocos_intermediate_dim")
                                  else vocos_intermediate_dim)
        vocos_num_layers = (cfg.vocos_num_layers if cfg is not None
                            and hasattr(cfg, "vocos_num_layers") else
                            vocos_num_layers)
        num_quantizers = (cfg.num_quantizers if cfg is not None and hasattr(
            cfg, "num_quantizers") else num_quantizers)
        downsample_scale = (cfg.downsample_scale if cfg is not None
                            and hasattr(cfg, "downsample_scale") else
                            downsample_scale)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale

        if self.downsample_scale != None and self.downsample_scale > 1:
            self.down = nn.Conv1d(self.hidden_size,
                                  self.hidden_size,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
            self.up = nn.Conv1d(self.hidden_size,
                                self.hidden_size,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # 時間軸を圧縮する。[B,T50,D]→[B,D,U]。Uは圧縮後の長さ。
        """時間圧縮、Vocos、FVQ、Vocos復元、時間拡大を順に計算する。

        Args:
            x: 正規化音声特徴[B,T50,D]。Bは発話数、T50は約50Hzのフレーム数、Dは入力の特徴次元数。

        Returns:
            復元特徴[B,Teven,D]、量子化損失[]、ID[Nq,B,U]、再構成損失[]の組。Bは発話数、
            Tevenは必要に応じて奇数末尾を除いた長さ、Dは特徴次元数、Nqは量子化器数、Uは圧縮長。[]はscalar。
        """
        feat = x
        length = x.size(1)
        if length % 2 != 0:
            # 圧縮前の長さが奇数なら最後の1フレームを除く。
            x = x[:, :-1, :]
            feat = feat[:, :-1, :]  # 再構成GTも同じ長さへ切り詰め、予測との対応を保つ。
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            # 時間をstrideで圧縮する。[B,D,T50]→[B,D,U]。Bは発話数、Dは特徴次元数、Uは圧縮後長。
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        # Vocosへ[B,D,U]で渡し、出力[B,U,D]を量子化用の[B,D,U]へ戻す。圧縮なしではUが入力長。
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        # 量子化特徴[B,D,U]をVocosで復元特徴[B,U,D]へ変換する。
        # 量子化特徴[B,D,U]→復元用特徴[B,U,D]。Bは発話数、Dは特徴次元数、Uはsemantic長。
        x = self.decoder(quantized_out)
        x_rec = x

        # 時間軸を2倍へ拡大し、出力畳み込みで元の特徴次元数へ戻す。
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            # 最近傍補間で時間を2倍にする。[B,D,U]→[B,D,2U]。この順序は対象の公式codecと同じ。
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            # 出力畳み込み後に時間軸を中央へ戻す。[B,D,2U]→[B,2U,D]。
            x_rec = self.up(x).transpose(1, 2)

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices
        reconstruction_loss = F.mse_loss(x_rec, feat)

        return x_rec, codebook_loss, all_indices, reconstruction_loss

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """特徴[B,T50,D]を時間圧縮・Vocos・FVQに通し、IDと量子化特徴を返す。

        Args:
            x: 正規化音声特徴[B,T50,D]。Bは発話数、T50は約50Hzのフレーム数、Dは入力の特徴次元数。

        Returns:
            ID[B,U]または[Nq,B,U]と量子化特徴[B,U,D]の組。Bは発話数、Nqは量子化器数、Uは圧縮長、Dは特徴次元数。
        """
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            # 時間をstrideで圧縮する。[B,D,T50]→[B,D,U]。Bは発話数、Dは特徴次元数、Uは圧縮後長。
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        # Vocosへ[B,D,U]で渡し、出力[B,U,D]を量子化用の[B,D,U]へ戻す。圧縮なしではUが入力長。
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self) -> None:
        """登録済みの各層へcodecの重み初期化を再適用する。

        Returns:
            None。登録した層の重みを再初期化する。
        """
        self.apply(init_weights)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """semantic IDから連続特徴を復元する。

        Args:
            codes: semantic ID[Nq,B,U]または[B,U]。Nqは量子化器数、Bは発話数、Uはsemantic長。

        Returns:
            復元特徴[B,Trec,D]。Bは発話数、Trecは時間拡大後の長さ、Dは元の特徴次元数。対象の2倍圧縮構成ではTrec=2U。
        """
        # 量子化器が1個のID列には量子化器軸を追加する。
        if codes.dim() == 2:
            # ID[B,U]→[1,B,U]。先頭1は量子化器数、Bは発話数、Uはsemantic長。
            codes = codes.unsqueeze(0)

        # 各codebookのlookup値を合計し、量子化特徴[B,D,U]を復元する。
        quantized_out = self.quantizer.vq2emb(codes)
        # 量子化特徴[B,D,U]→復元用特徴[B,U,D]。Bは発話数、Dは特徴次元数、Uはsemantic長。
        x = self.decoder(quantized_out)

        # 時間圧縮を使った構成では、復元特徴の時間軸を拡大する。
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            # 最近傍補間で時間を2倍にする。[B,D,U]→[B,D,2U]。この順序は対象の公式codecと同じ。
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            # 出力畳み込み後に時間軸を中央へ戻す。[B,D,2U]→[B,2U,D]。
            x_rec = self.up(x).transpose(1, 2)

        return x_rec

    def load_checkpoint(self, checkpoint_path: Any) -> None:
        """保存されたcodec重みを名前とshapeが一致する層へ読み込む。

        Args:
            checkpoint_path: 読み込むcheckpointファイルのパス。

        Returns:
            None。インスタンスの重みを更新する。
        """
        assert os.path.isfile(
            checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        saved_state_dict = checkpoint_dict['model']
        state_dict = self.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in saved_state_dict and saved_state_dict[k].shape == v.shape:
                new_state_dict[k] = saved_state_dict[k]
            else:
                logger.warning("%s is not in the checkpoint or shape mismatch",
                               k)
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
        logger.info("Loaded codec checkpoint '%s'", checkpoint_path)
