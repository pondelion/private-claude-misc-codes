"""
F5-TTS メインフロー - 簡略化疑似コード
==========================================

F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
公式実装: https://github.com/SWivid/F5-TTS
対応ファイル: src/f5_tts/model/cfm.py + src/f5_tts/model/backbones/dit.py

処理の流れ:
1. 音声 → Mel-Spectrogram抽出
2. テキスト → 文字トークン化 + フィラーパディング
3. ランダムマスク → Infilling条件生成
4. Flow Matching: φ_t = (1-t)*x0 + t*x1
5. DiTでフロー予測
6. 推論: ODE求解 (Euler + Sway Sampling + CFG)

============================================================
Shape Convention
============================================================
B: バッチサイズ
nw: 波形サンプル数 (24kHz × 秒数)
N: mel フレーム数 (= nw / hop_length, hop=256)
F: mel 周波数ビン数 (= 100, n_mel_channels)
nt: テキストトークン長 (文字数)
dim: DiT隠れ次元 (= 1024)
text_dim: テキスト埋め込み次元 (= 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import math


# ============================================================
# 設定
# ============================================================

DEFAULT_CONFIG = {
    # Mel-Spectrogram
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'n_mel_channels': 100,       # F: mel周波数ビン数
    'target_sample_rate': 24000,  # 24kHz

    # DiT Architecture
    'dim': 1024,                 # 隠れ次元
    'depth': 22,                 # DiTブロック数
    'heads': 16,                 # Attention ヘッド数
    'dim_head': 64,              # ヘッドあたりの次元
    'ff_mult': 2,                # FFN倍率 (1024 * 2 = 2048)
    'text_dim': 512,             # テキスト埋め込み次元
    'text_num_embeds': 2546,     # 語彙サイズ (英字+ピンイン+記号+フィラー+その他)
    'conv_layers': 4,            # ConvNeXt V2ブロック数 (テキスト前処理)

    # Flow Matching
    'sigma': 0.0,                # ノイズレベル (OTでは0)
    'frac_lengths_mask': (0.7, 1.0),  # マスク比率の範囲

    # CFG (Classifier-Free Guidance)
    'audio_drop_prob': 0.3,      # 音声条件ドロップ確率
    'cond_drop_prob': 0.2,       # 全条件ドロップ確率 (音声+テキスト)

    # Inference
    'nfe_steps': 32,             # ODE求解ステップ数 (NFE)
    'cfg_strength': 2.0,         # CFG強度
    'sway_sampling_coef': -1.0,  # Sway Sampling係数 (s < 0: 左寄り)
    'ode_method': 'euler',       # ODE求解法
}


# ============================================================
# Mel-Spectrogram 抽出
# ============================================================

class MelSpec(nn.Module):
    """
    波形 → メルスペクトログラム変換

    ========================================
    Shape
    ========================================
    入力:
        wav: (B, nw)
            - B: バッチサイズ
            - nw: 波形サンプル数

    出力:
        mel: (B, F, N)
            - F: mel周波数ビン数 (100)
            - N: 時間フレーム数 (≈ nw / hop_length)

    ========================================
    処理詳細
    ========================================
    Vocos backend:
        torchaudio.MelSpectrogram(power=1) → clamp(min=1e-5) → log()

    BigVGAN backend:
        STFT → |magnitude| → mel_basis @ mag → clamp → log()
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
        target_sample_rate: int = 24000,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length

        # Vocos backend (デフォルト)
        # torchaudio.transforms.MelSpectrogram(
        #     sample_rate=24000, n_fft=1024, win_length=1024,
        #     hop_length=256, n_mels=100, power=1,
        #     center=True, normalized=False, norm=None
        # )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        入力: wav (B, nw) - 24kHz waveform
        出力: mel (B, F=100, N) - log mel-spectrogram
        """
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, win_length=1024,
            hop_length=self.hop_length, n_mels=self.n_mel_channels,
            power=1, center=True, normalized=False, norm=None,
        ).to(wav.device)
        mel = mel_transform(wav)              # (B, 100, N) power=1 (振幅スペクトログラム)
        mel = mel.clamp(min=1e-5).log()       # (B, 100, N) 対数変換
        return mel


# ============================================================
# Conditional Flow Matching (CFM) ラッパー
# ============================================================

class CFM(nn.Module):
    """
    Conditional Flow Matching ラッパー

    Flow Matchingの学習・推論を管理:
    - 学習: ランダムマスク → ノイズ混合 → フロー予測 → MSE損失
    - 推論: ODE求解 (Euler + Sway Sampling + CFG)

    ========================================
    コンポーネント
    ========================================
    - mel_spec: MelSpec         - 波形→mel変換
    - transformer: DiT          - フロー予測モデル (22層)
    """

    def __init__(
        self,
        transformer: nn.Module,  # DiTバックボーン
        sigma: float = 0.0,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.0),
    ):
        super().__init__()

        self.transformer = transformer
        self.mel_spec = MelSpec()

        # CFG設定
        self.audio_drop_prob = audio_drop_prob  # 音声条件ドロップ確率
        self.cond_drop_prob = cond_drop_prob    # 全条件ドロップ確率

        # マスク比率の範囲 (70%-100%をマスク)
        self.frac_lengths_mask = frac_lengths_mask

        self.sigma = sigma
        self.num_channels = 100  # F: mel次元

    # ============================================================
    # 学習時のforward
    # ============================================================

    def forward(
        self,
        inp: torch.Tensor,       # (B, N, F) mel or (B, nw) raw wave
        text: torch.Tensor,      # (B, nt) トークン or list[str]
        lens: Optional[torch.Tensor] = None,  # (B,) 有効長
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        学習時のフォワードパス

        ========================================
        Shape
        ========================================
        入力:
            inp: (B, N, F) or (B, nw)
                - N: mel フレーム数
                - F: mel周波数ビン数 (100)
                - nw: 波形サンプル数 (rawの場合)

            text: (B, nt) or list[str]
                - nt: テキストトークン長

            lens: (B,) 各サンプルの有効mel長

        出力:
            loss: スカラー - MSE損失 (マスク領域のみ)
            cond: (B, N, F) - 条件音声 (非マスク部分)
            pred: (B, N, F) - 予測フローベクトル

        ========================================
        処理詳細
        ========================================
        """
        # --- Step 1: mel抽出 (rawの場合) ---
        if inp.ndim == 2:
            inp = self.mel_spec(inp)       # (B, nw) → (B, F, N)
            inp = inp.permute(0, 2, 1)     # (B, F, N) → (B, N, F)

        batch, seq_len = inp.shape[:2]

        # --- Step 2: マスク生成 ---
        # lens: 各サンプルの有効長
        if lens is None:
            lens = torch.full((batch,), seq_len, device=inp.device)
        mask = lens_to_mask(lens, length=seq_len)  # (B, N) True=有効位置

        # ランダムスパンマスク (70%-100%の連続区間をマスク)
        frac_lengths = torch.zeros(batch).uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)  # (B, N) True=マスク対象
        rand_span_mask = rand_span_mask & mask  # 有効範囲内のみ

        # --- Step 3: 条件音声 (マスク外を保持) ---
        x1 = inp                                          # (B, N, F) 元のmel
        cond = torch.where(
            rand_span_mask[..., None],  # (B, N, 1)
            torch.zeros_like(x1),       # マスク部分: ゼロ
            x1                          # 非マスク部分: 元のmel
        )                                                 # (B, N, F) 条件音声

        # --- Step 4: ノイズサンプリング & フロー混合 ---
        x0 = torch.randn_like(x1)                        # (B, N, F) ガウスノイズ
        time = torch.rand(batch, device=inp.device)       # (B,) 時刻 t ~ U[0,1]

        t = time.unsqueeze(-1).unsqueeze(-1)              # (B, 1, 1) ブロードキャスト用
        phi_t = (1 - t) * x0 + t * x1                    # (B, N, F) φ_t: OT補間
        flow = x1 - x0                                    # (B, N, F) フロー目標

        # --- Step 5: CFGドロップ (学習時) ---
        # 音声条件ドロップ: p=0.3
        import random as _random
        drop_audio_cond = _random.random() < self.audio_drop_prob

        # 全条件ドロップ: p=0.2 (音声+テキスト両方)
        if _random.random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # --- Step 6: DiTでフロー予測 ---
        pred = self.transformer(
            x=phi_t,                    # (B, N, F) ノイズ混合mel
            cond=cond,                  # (B, N, F) 条件音声
            text=text,                  # (B, nt) テキストトークン
            time=time,                  # (B,) フローステップ
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,                  # (B, N) 有効位置マスク
        )                               # → (B, N, F) 予測フロー

        # --- Step 7: 損失計算 (マスク領域のみ) ---
        loss = F.mse_loss(pred, flow, reduction='none')   # (B, N, F)
        loss = loss[rand_span_mask]                        # (マスク要素数, F)
        loss = loss.mean()                                 # スカラー

        return loss, cond, pred

    # ============================================================
    # 推論時のsample
    # ============================================================

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,            # (B, N_ref, F) 参照mel or (B, nw) raw
        text: torch.Tensor,            # (B, nt) トークン or list[str]
        duration: int,                 # 生成mel長
        *,
        lens: Optional[torch.Tensor] = None,
        steps: int = 32,               # NFE (ステップ数)
        cfg_strength: float = 2.0,     # CFG強度
        sway_sampling_coef: Optional[float] = -1.0,  # Sway Sampling係数
        seed: Optional[int] = None,
        vocoder: Optional[Callable] = None,  # mel → waveform
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推論: ODE求解でmel生成

        ========================================
        Shape
        ========================================
        入力:
            cond: (B, N_ref, F) or (B, nw)
                - N_ref: 参照melのフレーム数
                - F: mel次元 (100)

            text: (B, nt) or list[str]
                - nt: テキストトークン長

            duration: int
                - 生成するmel全体の長さ (N)

        出力:
            out: (B, N, F) or (B, nw)
                - 生成mel (vocoder指定時は波形)
            trajectory: (steps+1, B, N, F)
                - ODE軌跡

        ========================================
        処理詳細
        ========================================
        1. 条件mel準備 (パディング)
        2. ノイズ初期化 y0 ~ N(0,I)
        3. Sway Samplingタイムステップ生成
        4. ODE求解 (Euler + CFG)
        5. 条件部分置換 + Vocoder
        """
        self.eval()

        # --- Step 1: mel抽出 (rawの場合) ---
        if cond.ndim == 2:
            cond = self.mel_spec(cond)         # (B, nw) → (B, F, N_ref)
            cond = cond.permute(0, 2, 1)       # → (B, N_ref, F)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if lens is None:
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # --- Step 2: 条件マスク (参照部分=True) ---
        cond_mask = lens_to_mask(lens)                             # (B, N_ref) 参照部分

        max_duration = duration
        # 条件melをduration長にゼロパディング
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)  # (B, N, F)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)  # (B, N)
        cond_mask = cond_mask.unsqueeze(-1)                        # (B, N, 1) ブロードキャスト用

        # 条件音声: 参照部分のみ保持
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))  # (B, N, F)

        # --- Step 3: ノイズ初期化 ---
        if seed is not None:
            torch.manual_seed(seed)
        y0 = torch.randn(batch, max_duration, self.num_channels,
                          device=device, dtype=cond.dtype)         # (B, N, F)

        # --- Step 4: Sway Samplingタイムステップ生成 ---
        t = torch.linspace(0, 1, steps + 1, device=device)        # (steps+1,)
        # 例: steps=32 → [0.000, 0.031, 0.063, ..., 1.000]

        if sway_sampling_coef is not None:
            # Sway Sampling: f(u; s) = u + s * (cos(π/2 * u) - 1 + u)
            # s=-1: 左寄り (初期ステップ密、後期疎)
            # s=0: 均一 (通常サンプリングと同一)
            # s>0: 右寄り (初期疎、後期密)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            # s=-1の場合の変換例:
            # u=0.1 → t≈0.012 (初期圧縮)
            # u=0.5 → t≈0.293 (中間圧縮)
            # u=0.9 → t≈0.956 (後期拡張)

        # --- Step 5: ODE求解 (Euler + CFG) ---
        def ode_fn(t_step, x):
            """
            ODE右辺: フロー予測関数

            入力:
                t_step: スカラー - 現在のフローステップ
                x: (B, N, F) - 現在の状態

            出力:
                v: (B, N, F) - 予測フローベクトル (CFG適用済み)
            """
            if cfg_strength < 1e-5:
                # CFGなし: 条件付き予測のみ
                pred = self.transformer(
                    x=x, cond=step_cond, text=text,
                    time=t_step, mask=None,
                    drop_audio_cond=False, drop_text=False,
                )                                          # (B, N, F)
                return pred

            # CFGあり: 条件付き + 無条件を同時計算
            # cfg_infer=True: バッチを2倍にして効率的に処理
            pred_cfg = self.transformer(
                x=x, cond=step_cond, text=text,
                time=t_step, mask=None,
                cfg_infer=True,  # 条件/無条件をバッチ結合
            )                                              # (2B, N, F)

            pred_cond, pred_uncond = torch.chunk(pred_cfg, 2, dim=0)
            # pred_cond:   (B, N, F) - 条件付き予測
            # pred_uncond: (B, N, F) - 無条件予測

            # CFG線形外挿
            v = pred_cond + cfg_strength * (pred_cond - pred_uncond)  # (B, N, F)
            return v

        # ODE積分 (torchdiffeq.odeint)
        # trajectory[0] = y0, trajectory[-1] = 生成結果
        trajectory = odeint(ode_fn, y0, t)  # (steps+1, B, N, F)

        # --- Step 6: 後処理 ---
        sampled = trajectory[-1]                                   # (B, N, F)

        # 条件部分を元のmelに置換 (参照部分は変更しない)
        out = torch.where(cond_mask, cond, sampled)                # (B, N, F)

        # Vocoder適用 (指定時)
        if vocoder is not None:
            out = out.permute(0, 2, 1)     # (B, N, F) → (B, F, N)
            out = vocoder(out)             # (B, F, N) → (B, nw) waveform

        return out, trajectory


# ============================================================
# ODE求解 (簡略化)
# ============================================================

def odeint(fn, y0, t, method='euler'):
    """
    ODE求解 (Euler法)

    ========================================
    Shape
    ========================================
    入力:
        fn: Callable - ODE右辺 f(t, y) → dy/dt
        y0: (B, N, F) - 初期値
        t: (steps+1,) - タイムステップ列

    出力:
        trajectory: (steps+1, B, N, F) - ODE軌跡

    ========================================
    処理詳細 (Euler法)
    ========================================
    y_{k+1} = y_k + (t_{k+1} - t_k) * f(t_k, y_k)

    各ステップでDiTのforward 1回 (CFGありなら実質2回)
    """
    trajectory = [y0]
    y = y0

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]              # ステップ幅
        v = fn(t[i], y)                    # (B, N, F) フロー予測
        y = y + dt * v                     # (B, N, F) Eulerステップ
        trajectory.append(y)

    return torch.stack(trajectory, dim=0)  # (steps+1, B, N, F)


# ============================================================
# ユーティリティ関数
# ============================================================

def lens_to_mask(
    lens: torch.Tensor,
    length: Optional[int] = None,
) -> torch.Tensor:
    """
    有効長 → boolマスク変換

    ========================================
    Shape
    ========================================
    入力:
        lens: (B,) 各サンプルの有効長
        length: int マスクの全体長 (Noneならlens.max())

    出力:
        mask: (B, N) True=有効位置
            例: lens=[3, 5], length=6
            → [[T, T, T, F, F, F],
               [T, T, T, T, T, F]]
    """
    if length is None:
        length = lens.max().item()
    arange = torch.arange(length, device=lens.device)
    return arange < lens.unsqueeze(1)


def mask_from_frac_lengths(
    lens: torch.Tensor,         # (B,) 有効長
    frac_lengths: torch.Tensor, # (B,) マスク比率 [0.7, 1.0]
) -> torch.Tensor:
    """
    ランダムスパンマスク生成

    ========================================
    Shape
    ========================================
    入力:
        lens: (B,) 各サンプルの有効長
        frac_lengths: (B,) マスクする比率 (0.7-1.0)

    出力:
        mask: (B, N) True=マスク対象
            連続したスパンをランダムな開始位置からマスク

    ========================================
    処理詳細
    ========================================
    1. マスク長 = lens * frac_lengths (例: 938 * 0.85 ≈ 797)
    2. 開始位置 = rand() * (lens - mask_len) (例: rand() * 141)
    3. [開始, 開始+マスク長) の範囲をTrue

    例: lens=10, frac=0.7 → mask_len=7, start=2
    → [F, F, T, T, T, T, T, T, T, F]
    """
    mask_lengths = (lens * frac_lengths).long()
    max_start = lens - mask_lengths
    start = (torch.rand_like(frac_lengths) * max_start).long()

    length = lens.max().item()
    arange = torch.arange(length, device=lens.device)
    mask = (arange >= start.unsqueeze(1)) & (arange < (start + mask_lengths).unsqueeze(1))
    return mask


def list_str_to_idx(
    texts: List[str],
    vocab_char_map: Dict[str, int],
) -> torch.Tensor:
    """
    文字列リスト → トークンインデックス

    ========================================
    Shape
    ========================================
    入力:
        texts: list[str] 長さB - テキスト文字列
        vocab_char_map: dict - 文字→インデックス辞書

    出力:
        tokens: (B, max_len) - トークンインデックス (パディング=-1)

    ========================================
    処理詳細
    ========================================
    - 各文字を語彙辞書でインデックス化
    - 未知文字は無視 or 特殊トークン
    - バッチ内で最長に合わせて-1パディング
    - フィラートークン: index 0 (Embedding側で+1して使用)

    語彙構成:
    - 英字: a-z, A-Z
    - 数字: 0-9
    - 記号: .,!?;:-'"()...
    - 中国語ピンイン: a1, a2, ... (pypinyin変換後)
    - その他言語文字
    - 合計: 2546トークン
    """
    text_token_lists = []
    for text in texts:
        char_list = []
        for char in text:
            if char in vocab_char_map:
                char_list.append(vocab_char_map[char])
        text_token_lists.append(torch.tensor(char_list))

    # バッチパディング (-1でパディング)
    tokens = torch.nn.utils.rnn.pad_sequence(
        text_token_lists, padding_value=-1, batch_first=True
    )
    return tokens  # (B, max_len)


# ============================================================
# メインの使用例
# ============================================================

def example_training():
    """
    学習時の使用例

    ========================================
    データフロー
    ========================================
    音声 (B, nw) → MelSpec → (B, N, 100)
    テキスト list[str] → tokenize → (B, nt) → pad → (B, N)
    ランダムマスク → (B, N) True=生成対象
    cond = (1-mask) ⊙ mel → (B, N, 100)
    φ_t = (1-t)*noise + t*mel → (B, N, 100)
    DiT(φ_t, cond, text, t) → pred_flow (B, N, 100)
    loss = MSE(pred_flow, mel-noise) [mask only]
    """
    # モデル初期化
    from dit_model import DiT
    transformer = DiT(
        dim=1024, depth=22, heads=16, dim_head=64,
        ff_mult=2, mel_dim=100, text_num_embeds=2546,
        text_dim=512, conv_layers=4,
    )
    model = CFM(
        transformer=transformer,
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        frac_lengths_mask=(0.7, 1.0),
    )

    # ダミーデータ
    batch_size = 4
    audio = torch.randn(batch_size, 240000)  # (4, 240000) 10秒 @ 24kHz
    text = ["Hello, how are you?",
            "I am fine, thank you.",
            "What is your name?",
            "My name is F5-TTS."]

    # フォワードパス
    loss, cond, pred = model(audio, text)
    # loss: スカラー (MSE損失)
    # cond: (4, ~938, 100) 条件音声
    # pred: (4, ~938, 100) 予測フロー

    loss.backward()


def example_inference():
    """
    推論時の使用例

    ========================================
    データフロー
    ========================================
    参照音声 (1, nw_ref) → MelSpec → (1, N_ref, 100)
    Duration推定 → N = N_ref + N_ref * len(gen)/len(ref)
    cond = [ref_mel, zeros] → (1, N, 100)
    text = [ref_text, gen_text] → tokenize → (1, N)
    y0 ~ N(0,I) → (1, N, 100)
    Sway Sampling → t: (steps+1,)
    ODE求解 (Euler + CFG) → (1, N, 100)
    条件部分置換 → Vocoder → waveform (1, nw)
    """
    # 入力
    ref_audio = torch.randn(1, 72000)       # (1, 72000) 3秒参照音声
    ref_text = "Are you OK?"
    gen_text = "I'm fine, thank you very much!"

    # Duration推定
    ref_mel_len = 72000 // 256               # ≈ 281 フレーム
    duration = ref_mel_len + int(ref_mel_len * len(gen_text) / len(ref_text))
    # ≈ 281 + 281 * 30/11 ≈ 281 + 766 = 1047 フレーム

    # テキスト結合
    combined_text = ref_text + " " + gen_text  # 参照 + 生成テキスト

    # 推論 (model は CFM クラスのインスタンス)
    # generated, trajectory = model.sample(
    #     cond=ref_audio,          # (1, 72000) raw wave
    #     text=[combined_text],    # list[str]
    #     duration=duration,       # 1047
    #     steps=32,                # 32 NFE
    #     cfg_strength=2.0,        # CFG強度
    #     sway_sampling_coef=-1.0, # 左寄りサンプリング
    #     vocoder=vocos,           # mel → waveform
    # )
    # generated: (1, nw_total) waveform
    # trajectory: (33, 1, 1047, 100) ODE軌跡

    # 参照部分を除去して生成波形のみ取得
    # ref_nw = ref_mel_len * 256
    # gen_waveform = generated[0, ref_nw:]  # (nw_gen,)
    # return gen_waveform, 24000


if __name__ == "__main__":
    print("=== F5-TTS Main Flow ===")
    print()
    print("学習時:")
    print("  入力: audio (B, nw), text list[str]")
    print("  処理: MelSpec → マスク → Flow Mixing → DiT → MSE Loss")
    print("  出力: loss (scalar)")
    print()
    print("推論時:")
    print("  入力: ref_audio (B, nw), text list[str], duration int")
    print("  処理: Sway Sampling → ODE (Euler+CFG) → Vocoder")
    print("  出力: waveform (B, nw)")
    print()
    print("Base Model設定:")
    print("  DiT: dim=1024, depth=22, heads=16, head_dim=64")
    print("  テキスト: vocab=2546, text_dim=512, ConvNeXt V2 ×4")
    print("  Mel: 24kHz, hop=256, n_mel=100")
    print("  学習: AdamW, lr=7.5e-5, 1.2M updates, 8×A100")
    print("  推論: 32 NFE, CFG=2.0, Sway s=-1, RTF=0.15")
