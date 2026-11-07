"""S2MelのDiT本体。時刻・話者・semantic条件とMelを融合する。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=参照+生成Mel長、F=Mel bin数、C=semantic条件の特徴次元数、D=隠れ状態の次元数。Mel[B,F,T]、
条件[B,T,C]。
"""
from __future__ import annotations

from typing import Any
import math

from torch import nn
from torch.nn.utils import weight_norm
import torch

from models.common import sequence_mask
from models.s2mel.transformer import ModelArgs, Transformer
from models.s2mel.wavenet import WN


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> Any:
    """条件のscaleとshiftで各時刻の特徴を変調する。

    Args:
        x: 入力特徴[B,T,D]。Bは発話数、Tは位置数、Dは特徴次元数。
        shift: 条件の加算項[B,D]。Bは発話数、Dは特徴次元数。
        scale: 条件の乗算係数[B,D]。Bは発話数、Dは特徴次元数。

    Returns:
        変調した特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """flow時刻をsin/cos特徴とMLPで埋め込む。"""

    def __init__(self,
                 hidden_size: int,
                 frequency_embedding_size: int = 256) -> None:
        """flow時刻をsin/cos特徴とMLPで埋め込むための層と設定を構築する。

        Args:
            hidden_size: 内部表現の特徴次元数。
            frequency_embedding_size: 時刻のsin/cos特徴の次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """flow時刻からsin/cosの周波数特徴を計算する。

        Args:
            t: flow時刻[B]。Bは発話数。連続値の時刻を受け取る。

        Returns:
            時刻sin/cos特徴[B,Q]。Bは発話数、Qはfrequency_embedding_size。
            特徴次元数が奇数の場合では末尾をゼロで補う。
        """
        # 参照資料: https://github.com/openai/glide-
        # text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        # 周波数ごとのcosとsinを特徴軸で結合する。[B,Q/2]×2→[B,Q]。Bは発話数、Qは時刻特徴次元数。
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """flow時刻をsin/cos特徴とMLPで埋め込む。

        Args:
            t: flow時刻[B]。Bは発話数。連続値の時刻を受け取る。

        Returns:
            時刻埋め込み[B,D]。Bは発話数、Dはhidden_size。
        """
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class StyleEmbedder(nn.Module):
    """話者特徴を射影し、必要に応じて条件をdropoutする。"""

    def __init__(self, input_size: int, hidden_size: int,
                 dropout_prob: float) -> None:
        """話者特徴を射影し、必要に応じて条件をdropoutするための層と設定を構築する。

        Args:
            input_size: 入力特徴の次元数。
            hidden_size: 内部表現の特徴次元数。
            dropout_prob: 話者条件を無効化する確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(int(use_cfg_embedding),
                                            hidden_size)
        self.style_in = weight_norm(
            nn.Linear(input_size, hidden_size, bias=True))
        self.input_size = input_size
        self.dropout_prob = dropout_prob

    def forward(self,
                labels: torch.Tensor,
                train: Any,
                force_drop_ids: Any = None) -> torch.Tensor:
        """話者特徴を射影し、必要に応じて条件をdropoutする。

        Args:
            labels: 話者特徴[B,Din]。Bは発話数、Dinは話者特徴次元数。
            train: 話者条件dropoutを適用する学習時の処理か。
            force_drop_ids: 条件を無効にする発話のmask[B]。Bは発話数。Noneでは確率で決める。

        Returns:
            射影した話者特徴[B,D]。Bは発話数、Dはhidden_size。
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        else:
            labels = self.style_in(labels)
        embeddings = labels
        return embeddings


class FinalLayer(nn.Module):
    """時刻条件で変調した特徴をMelの出力次元へ射影する。"""

    def __init__(self, hidden_size: int, patch_size: Any,
                 out_channels: int) -> None:
        """時刻条件で変調した特徴をMelの出力次元へ射影するための層と設定を構築する。

        Args:
            hidden_size: 内部表現の特徴次元数。
            patch_size: 最終射影のpatchの大きさ。出力特徴数へ乗算される。
            out_channels: 出力チャネル数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = weight_norm(
            nn.Linear(hidden_size,
                      patch_size * patch_size * out_channels,
                      bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """時刻条件で変調した特徴をMelの出力次元へ射影する。

        特徴[B,T,D]と時刻[B,D]→出力[B,T,Dout]。T=Mel長、Dout=出力の特徴次元数。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            c: 時刻条件[B,D]または[B,1,D]。Bは発話数、Dは隠れ状態の次元数。受け付ける形は各射影の定義に従う。

        Returns:
            射影特徴[B,T,P*P*Cout]。Bは発話数、Tは位置数、Pはpatch_size、Coutはout_channels。
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(torch.nn.Module):
    """時刻・話者・semantic条件からMelのflow速度を予測する。"""

    def __init__(self, args: Any) -> None:
        """時刻・話者・semantic条件からMelのflow速度を予測するための層と設定を構築する。

        args=公式configと同じ階層の構成値。

        Args:
            args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(DiT, self).__init__()
        self.time_as_token = args.DiT.time_as_token if hasattr(
            args.DiT, 'time_as_token') else False
        self.style_as_token = args.DiT.style_as_token if hasattr(
            args.DiT, 'style_as_token') else False
        self.uvit_skip_connection = args.DiT.uvit_skip_connection if hasattr(
            args.DiT, 'uvit_skip_connection') else False
        model_args = ModelArgs(
            block_size=16384,
            n_layer=args.DiT.depth,
            n_head=args.DiT.num_heads,
            dim=args.DiT.hidden_dim,
            head_dim=args.DiT.hidden_dim // args.DiT.num_heads,
            vocab_size=1024,
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = args.DiT.in_channels
        self.out_channels = args.DiT.in_channels
        self.num_heads = args.DiT.num_heads

        self.x_embedder = weight_norm(
            nn.Linear(args.DiT.in_channels, args.DiT.hidden_dim, bias=True))

        self.content_type = args.DiT.content_type  # 条件特徴が離散IDか連続値かを設定する。
        # 離散ID条件の入力次元を設定する。
        self.content_codebook_size = args.DiT.content_codebook_size
        self.content_dim = args.DiT.content_dim  # 連続値条件の入力次元を設定する。
        self.cond_embedder = nn.Embedding(
            args.DiT.content_codebook_size,
            args.DiT.hidden_dim)  # 離散IDを特徴へ変換する埋め込み表を用意する。
        self.cond_projection = nn.Linear(args.DiT.content_dim,
                                         args.DiT.hidden_dim,
                                         # 連続値のsemantic特徴を内部特徴の次元数へ射影する。
                                         bias=True)

        self.is_causal = args.DiT.is_causal

        self.t_embedder = TimestepEmbedder(args.DiT.hidden_dim)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        # Mel速度の出力層をMLP方式またはWaveNet方式から選ぶ。
        self.final_layer_type = args.DiT.final_layer_type
        if self.final_layer_type == 'wavenet':
            self.t_embedder2 = TimestepEmbedder(args.wavenet.hidden_dim)
            self.conv1 = nn.Linear(args.DiT.hidden_dim,
                                   args.wavenet.hidden_dim)
            self.conv2 = nn.Conv1d(args.wavenet.hidden_dim,
                                   args.DiT.in_channels, 1)
            self.wavenet = WN(hidden_channels=args.wavenet.hidden_dim,
                              kernel_size=args.wavenet.kernel_size,
                              dilation_rate=args.wavenet.dilation_rate,
                              n_layers=args.wavenet.num_layers,
                              gin_channels=args.wavenet.hidden_dim,
                              p_dropout=args.wavenet.p_dropout,
                              causal=False)
            self.final_layer = FinalLayer(args.wavenet.hidden_dim, 1,
                                          args.wavenet.hidden_dim)
            self.res_projection = nn.Linear(
                args.DiT.hidden_dim,
                args.wavenet.hidden_dim)  # Transformer出力をWaveNet出力へ足すための残差射影。
            self.wavenet_style_condition = args.wavenet.style_condition
            assert args.DiT.style_condition == args.wavenet.style_condition
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(args.DiT.hidden_dim, args.DiT.hidden_dim),
                nn.SiLU(),
                nn.Linear(args.DiT.hidden_dim, args.DiT.in_channels),
            )
        self.transformer_style_condition = args.DiT.style_condition

        self.class_dropout_prob = args.DiT.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, args.DiT.hidden_dim)

        self.long_skip_connection = args.DiT.long_skip_connection
        self.skip_linear = nn.Linear(
            args.DiT.hidden_dim + args.DiT.in_channels, args.DiT.hidden_dim)

        self.cond_x_merge_linear = nn.Linear(
            args.DiT.hidden_dim + args.DiT.in_channels * 2 +
            args.style_encoder.dim * self.transformer_style_condition *
            (not self.style_as_token), args.DiT.hidden_dim)
        if self.style_as_token:
            self.style_in = nn.Linear(args.style_encoder.dim,
                                      args.DiT.hidden_dim)

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        """位置回転係数[T,d/2,2]とcausal mask[T,T]、必要なKV保存領域を準備する。

        Args:
            max_batch_size: キャッシュが受け付ける最大発話数。
            max_seq_length: キャッシュが保持できる最大位置数。

        Returns:
            None。位置表、mask、必要なKV保存領域をインスタンスへ登録する。
        """
        self.transformer.setup_caches(max_batch_size,
                                      max_seq_length,
                                      use_kv_cache=False)

    def forward(self,
                x: torch.Tensor,
                prompt_x: torch.Tensor,
                x_lens: torch.Tensor,
                t: torch.Tensor,
                style: torch.Tensor,
                cond: torch.Tensor,
                mask_content: bool = False) -> torch.Tensor:
        """時刻・話者・semantic条件からMelのflow速度を予測する。

        Args:
            x: 現在時刻のMel状態[B,F,T]。Bは発話数、FはMel bin数、Tは参照を含む全Mel長。
            prompt_x: 生成区間をゼロにした参照Mel条件[B,F,T]。Bは発話数、FはMel bin数、
                Tは参照と生成を含む時間長。
            x_lens: 各発話のMel有効フレーム数[B]。Bは発話数。
            t: flow時刻[B]。Bは発話数。連続値の時刻を受け取る。
            style: 話者条件[B,192]。Bは発話数。
            cond: 整列済みsemantic特徴[B,T,C]。Bは発話数、Tは全Mel長、Cは条件の特徴次元数。
            mask_content: semantic条件を無効化するか。真なら全条件をゼロにする。

        Returns:
            予測速度[B,F,T]。Bは発話数、FはMel bin数、Tは入力と同じ全Mel長。
        """
        class_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
        if not self.training and mask_content:
            class_dropout = True

        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # 時刻特徴[B,D]。Bは発話数、DはDiT内部特徴の次元数。
        # semantic条件[B,T,C]をDiT条件の特徴次元数へ射影する。Bは発話数、Tは全Mel長、Cは入力条件の特徴次元数。
        cond = cond_in_module(cond)

        # Melの特徴軸を末尾へ移す。[B,F,T]→[B,T,F]。Bは発話数、FはMel bin数、Tは全Mel長。
        x = x.transpose(1, 2)
        # Melの特徴軸を末尾へ移す。[B,F,T]→[B,T,F]。Bは発話数、FはMel bin数、Tは全Mel長。
        prompt_x = prompt_x.transpose(1, 2)

        x_in = torch.cat(
            [x, prompt_x, cond],
            # 現在Mel・参照Mel・semantic特徴を連結する。[B,T,F]×2と[B,T,C]→
            # [B,T,2F+C]。
            dim=-1)

        # 話者条件を専用tokenにしない設定では、各Mel位置の特徴へ連結する。
        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat(
                [x_in, style[:, None, :].repeat(1, T, 1)],
                # 話者特徴を全時刻へ展開して加える。[B,T,2F+C]→[B,T,2F+C+Ds]。Dsは話者埋め込みの次元数
                # 。
                dim=-1)

        if class_dropout:
            x_in[..., self.in_channels:] = x_in[
                ...,
                # 現在のMel状態のF次元だけを残し、参照・semantic・話者の条
                # 件部分を無効化する。
                self.in_channels:] * 0

        # 融合した隠れ状態[B,T,D]。Bは発話数、Tは全Mel長、DはDiT内部特徴の次元数。
        x_in = self.cond_x_merge_linear(x_in)

        if self.style_as_token:
            style = self.style_in(style)
            style = torch.zeros_like(style) if class_dropout else style
            # 話者条件を1個のtokenとして先頭に追加する。[B,T,D]→[B,T+1,D]。
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)

        if self.time_as_token:
            # 時刻条件を1個のtokenとして先頭に追加する。[B,Tcurrent,D]→[B,Tcurrent+1,D]。
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)

        x_mask = sequence_mask(
            x_lens + self.style_as_token + self.time_as_token,
            max_length=x_in.size(1)).to(x.device).unsqueeze(
                # 有効位置mask[B,1,Ttokens]。Ttokensは追加条件tokenを含む
                # 全長。
                1)
        # 位置index[Ttokens]。Ttokensは追加条件tokenを含む全長。
        input_pos = self.input_pos[:x_in.size(1)]
        x_mask_expanded = x_mask[:, None, :].repeat(
            1,
            1,
            x_in.size(1),
            # 参照可能なkey位置mask[B,1,Ttokens,Ttokens]を全queryへ展開する。
            1) if not self.is_causal else None
        x_res = self.transformer(
            x_in,
            t1.unsqueeze(1),
            input_pos,
            # 文脈化した特徴[B,Ttokens,D]。Ttokensは時刻・話者tokenを含む長さ
            # 、DはDiT内部特徴の次元数。
            x_mask_expanded)
        # 追加した時刻tokenだけを除き、Mel位置の特徴へ戻す。
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        # 追加した話者tokenだけを除く。両条件tokenの除去後は[B,T,D]。
        x_res = x_res[:, 1:] if self.style_as_token else x_res

        if self.long_skip_connection:
            # Transformer特徴と元のMelを結合して射影する。[B,T,D+F]→[B,T,D]。FはMel bin数。
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        if self.final_layer_type == 'wavenet':
            # WaveNetの内部特徴の次元数Cwへ射影する。[B,T,D]→[B,T,Cw]。Cwはwavenet.hidden_dim。
            x = self.conv1(x_res)
            x = x.transpose(1, 2)
            t2 = self.t_embedder2(t)
            # WaveNet特徴[B,Cw,T]を時間優先[B,T,Cw]へ戻し、同shapeの残差射影を加える。
            x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(
                # Transformer入力側の特徴を出力へ残差接続する。
                1,
                2) + self.res_projection(x_res)
            # 時刻変調後にMel bin数へ射影し、軸を戻す。[B,T,Cw]→[B,T,F]→[B,F,T]。
            x = self.final_layer(x, t1).transpose(1, 2)
            x = self.conv2(x)
        else:
            # MLP出力経路では各時刻を直接Mel bin数へ射影する。[B,T,D]→[B,T,F]。
            x = self.final_mlp(x_res)
            x = x.transpose(1, 2)
        # 最終出力はMel速度[B,F,T]。Bは発話数、FはMel bin数、Tは全Mel長。
        return x
