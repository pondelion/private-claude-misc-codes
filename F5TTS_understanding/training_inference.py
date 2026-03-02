"""
F5-TTS 学習・推論パイプライン - 簡略化疑似コード
==================================================

学習ループ、データ処理、推論パイプラインの詳細。
対応ファイル:
  - src/f5_tts/model/trainer.py     (学習)
  - src/f5_tts/model/dataset.py     (データ)
  - src/f5_tts/infer/utils_infer.py (推論)
  - src/f5_tts/api.py               (高レベルAPI)

============================================================
Shape Convention
============================================================
B: バッチサイズ (動的、フレーム数ベース)
nw: 波形サンプル数 (24kHz)
N: melフレーム数 (= nw / 256)
F: mel周波数ビン数 (= 100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 学習設定
# ============================================================

@dataclass
class TrainingConfig:
    """
    F5-TTS Base Model の学習設定

    対応: src/f5_tts/configs/F5TTS_Base.yaml
    """
    # データ
    dataset_name: str = "Emilia_ZH_EN"     # Emilia多言語データセット
    total_hours: float = 95000              # ≈95K時間 (英語+中国語)
    sample_rate: int = 24000                # 24kHz

    # バッチ
    batch_size_per_gpu: int = 38400         # フレーム数ベース (≈ 409秒/GPU)
    max_samples: int = 64                   # 最大サンプル数/バッチ
    num_gpus: int = 8                       # 8 × NVIDIA A100 80G
    total_batch_frames: int = 307200        # 38400 × 8 = 307,200 フレーム
    # ≈ 3,276秒 ≈ 54.6分/バッチ

    # 最適化
    optimizer: str = "AdamW"
    learning_rate: float = 7.5e-5
    warmup_updates: int = 20000             # 線形ウォームアップ
    max_grad_norm: float = 1.0
    epochs: int = 11
    total_updates: int = 1200000            # ≈1.2Mアップデート

    # EMA
    ema_decay: float = 0.9999              # Exponential Moving Average

    # チェックポイント
    save_per_updates: int = 50000           # 50Kアップデートごと保存
    last_checkpoint_per_updates: int = 5000 # 5Kアップデートごと最新保存

    # Mel-Spectrogram
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 100

    # マスク
    frac_lengths_mask: Tuple[float, float] = (0.7, 1.0)  # 70-100%マスク

    # CFG
    audio_drop_prob: float = 0.3
    cond_drop_prob: float = 0.2


# ============================================================
# データセット
# ============================================================

class F5TTSDataset:
    """
    F5-TTS 学習データセット

    対応: src/f5_tts/model/dataset.py

    ========================================
    データ形式
    ========================================
    各サンプル:
        audio: (nw,) 波形 (24kHz, mono)
        text: str テキスト (英語 or 中国語)
        duration: float 秒数 (0.3s - 30s でフィルタ)

    ========================================
    テキスト処理
    ========================================
    英語: アルファベット + 記号をそのまま使用
        "Hello, world!" → ['H','e','l','l','o',',',' ','w','o','r','l','d','!']

    中国語: ピンイン変換 (pypinyin)
        "你好世界" → "ni3 hao3 shi4 jie4"

    語彙: 2546トークン
        - 英字: a-z, A-Z (52)
        - 数字: 0-9 (10)
        - 記号: .,!?;:-'"()... (30+)
        - ピンイン: a1-a4, b1-b4, ... (数百)
        - その他言語文字 (Emiliaの多言語)
        - フィラートークン (1)

    ========================================
    Mel抽出
    ========================================
    audio (nw,) → MelSpectrogram → mel (F, N)
    F=100, N = nw/256

    例: 10秒音声
        nw = 240,000
        N = 240,000 / 256 = 937.5 → 938フレーム
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        # データ読み込み (Arrow形式 or HuggingFace)
        # self.data = load_dataset(...)

    def __getitem__(self, idx):
        """
        返値:
            mel: (N, F=100) melスペクトログラム (transpose済み)
            text: str テキスト文字列
            duration: int melフレーム数
        """
        import torchaudio
        sample = self.data[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]

        audio, sr = torchaudio.load(audio_path)
        # audio: (channels, nw)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.config.sample_rate)
        # audio: (1, nw)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, n_mels=self.config.n_mel_channels,
            power=1, center=True,
        )
        mel = mel_transform(audio).squeeze(0)       # (F, N)
        mel = mel.clamp(min=1e-5).log()              # (F, N) 対数メル
        mel = mel.permute(1, 0)                       # (N, F) sequence-first
        duration = mel.shape[0]

        return mel, text, duration


class DynamicBatchSampler:
    """
    動的バッチサンプラー (フレーム数ベース)

    対応: src/f5_tts/model/dataset.py

    ========================================
    処理詳細
    ========================================
    1. 全サンプルをDurationでソート
    2. 類似長のサンプルをグループ化 (パディング削減)
    3. バッチのフレーム合計が target_frames 以下になるよう調整
    4. max_samples制約で1バッチの最大サンプル数を制限

    例:
        target_frames = 38,400 (≈ 409秒)
        max_samples = 64
        → 5秒音声: ~468フレーム → バッチ内82サンプル (max_samplesで64に)
        → 10秒音声: ~938フレーム → バッチ内40サンプル
        → 30秒音声: ~2813フレーム → バッチ内13サンプル

    ========================================
    利点
    ========================================
    - パディング最小化: 類似長をバッチ化
    - GPU利用率最大化: フレーム数で制御
    - 長いシーケンスも学習可能: バッチサイズ自動調整
    """

    def __init__(
        self,
        durations: List[int],
        target_frames: int = 38400,
        max_samples: int = 64,
    ):
        self.durations = durations
        self.target_frames = target_frames
        self.max_samples = max_samples

    def __iter__(self):
        # ソート → グループ化 → フレーム数でバッチ分割
        sorted_indices = sorted(range(len(self.durations)),
                                key=lambda i: self.durations[i])

        batch = []
        batch_frames = 0
        for idx in sorted_indices:
            dur = self.durations[idx]
            if (batch_frames + dur > self.target_frames or
                    len(batch) >= self.max_samples):
                yield batch
                batch = []
                batch_frames = 0
            batch.append(idx)
            batch_frames += dur

        if batch:
            yield batch


def collate_fn(batch):
    """
    バッチ照合関数

    ========================================
    Shape
    ========================================
    入力: list of (mel, text, duration)
        mel: (N_i, F) 各サンプル可変長

    出力:
        mel_padded: (B, N_max, F) ゼロパディング
        mel_lengths: (B,) 各サンプルの有効長
        text: list[str] テキスト文字列リスト

    ========================================
    処理
    ========================================
    1. バッチ内最長のmel長に合わせてゼロパディング
    2. テキストはリストのまま (tokenizeはモデル内で実行)
    """
    mels = [item[0] for item in batch]     # list of (N_i, F)
    texts = [item[1] for item in batch]
    durations = [item[2] for item in batch]

    # パディング
    mel_padded = torch.nn.utils.rnn.pad_sequence(
        mels, batch_first=True, padding_value=0.0
    )  # (B, N_max, F)

    mel_lengths = torch.tensor(durations)   # (B,)

    return mel_padded, mel_lengths, texts


# ============================================================
# 学習ループ
# ============================================================

class Trainer:
    """
    F5-TTS 学習ループ

    対応: src/f5_tts/model/trainer.py

    ========================================
    学習フロー概要
    ========================================
    for epoch in range(11):
        for batch in dataloader:
            mel, lens, text = batch
            loss, _, _ = model(mel, text, lens=lens)
            loss.backward()
            optimizer.step()
            ema.update()
            if update % 50K == 0: save_checkpoint()
    """

    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config

        # Accelerateによる分散学習
        # self.accelerator = Accelerator(...)

        # オプティマイザ
        # self.optimizer = AdamW(model.parameters(), lr=7.5e-5)

        # LRスケジューラ: 線形ウォームアップ + 線形減衰
        # self.scheduler = get_linear_schedule_with_warmup(
        #     optimizer, warmup_steps=20000, training_steps=1200000
        # )

        # EMA (Exponential Moving Average)
        # self.ema = EMA(model, decay=0.9999)

    def train(self, dataset):
        """
        メイン学習ループ

        ========================================
        1バッチの処理フロー
        ========================================
        1. データ取得: mel (B, N, 100), lens (B,), text list[str]
        2. forward: loss, cond, pred = model(mel, text, lens=lens)
           内部:
             a. mel抽出 (rawの場合)
             b. ランダムマスク生成 (70-100%)
             c. ノイズ混合: φ_t = (1-t)*noise + t*mel
             d. DiT予測: pred_flow = DiT(φ_t, cond, text, t)
             e. MSE loss (マスク領域のみ)
        3. backward: loss.backward()
        4. gradient clipping: max_norm=1.0
        5. optimizer.step()
        6. ema.update()
        7. logging & checkpointing

        ========================================
        学習統計 (Base Model)
        ========================================
        - データ: Emilia ~95K時間 (英語+中国語)
        - バッチ: 307,200 フレーム (8 GPU × 38,400)
        - アップデート: ~1.2M
        - 所要時間: > 1週間 (8×A100 80G)
        - 1アップデート: ≈0.91時間のデータ
        """
        for epoch in range(self.config.epochs):
            for batch_idx, batch in enumerate(self._get_dataloader(dataset)):
                mel, lens, text = batch
                # mel:  (B, N_max, 100) パディング済み
                # lens: (B,) 有効長
                # text: list[str] テキスト

                # --- Forward ---
                loss, cond, pred = self.model(
                    inp=mel,        # (B, N, 100)
                    text=text,      # list[str]
                    lens=lens,      # (B,)
                )
                # loss: スカラー (マスク領域のMSE)

                # --- Backward ---
                loss.backward()

                # --- Gradient Clipping ---
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm  # 1.0
                )

                # --- Optimizer Step ---
                # self.optimizer.step()
                # self.scheduler.step()
                # self.optimizer.zero_grad()

                # --- EMA Update ---
                # self.ema.update()

                # --- Logging ---
                # if update % 100 == 0:
                #     log({"loss": loss.item(), "lr": scheduler.get_lr()})

                # --- Checkpointing ---
                # if update % 50000 == 0:
                #     save_checkpoint(update)

    def _get_dataloader(self, dataset):
        """動的バッチサンプラー付きDataLoader"""
        from torch.utils.data import DataLoader

        durations = [dataset[i][2] for i in range(len(dataset))]
        sampler = DynamicBatchSampler(
            durations=durations,
            target_frames=self.config.batch_size_per_gpu,
            max_samples=self.config.max_samples,
        )
        return DataLoader(
            dataset, batch_sampler=sampler,
            collate_fn=collate_fn, num_workers=16,
        )


# ============================================================
# 推論パイプライン
# ============================================================

class InferencePipeline:
    """
    F5-TTS 推論パイプライン

    対応: src/f5_tts/infer/utils_infer.py

    ========================================
    推論フロー
    ========================================
    1. 参照音声前処理 (リサンプル、無音除去)
    2. Duration推定
    3. テキスト準備 (参照+生成テキスト結合)
    4. ODE求解 (Sway Sampling + CFG)
    5. mel → waveform (Vocoder)
    6. チャンク結合 (長文の場合)
    """

    def __init__(
        self,
        model: nn.Module,
        vocoder,          # Vocos or BigVGAN
        sample_rate: int = 24000,
        hop_length: int = 256,
    ):
        self.model = model
        self.vocoder = vocoder
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def preprocess_ref_audio(
        self,
        ref_audio_path: str,
        ref_text: str = "",
        max_duration: float = 12.0,  # 最大12秒
    ) -> Tuple[torch.Tensor, str]:
        """
        参照音声の前処理

        ========================================
        Shape
        ========================================
        入力:
            ref_audio_path: str - 音声ファイルパス
            ref_text: str - 参照テキスト (空ならWhisperで自動認識)

        出力:
            ref_audio: (1, nw) - 前処理済み波形
            ref_text: str - テキスト (ASR結果含む)

        ========================================
        処理
        ========================================
        1. 音声読み込み (任意形式)
        2. モノラル変換
        3. 24kHzリサンプル
        4. 無音除去 (pydub: 先頭/末尾の-50dB以下)
        5. 最大12秒にカット
        6. ref_text空なら Whisper (faster_whisper) で認識
        """
        import torchaudio
        audio, sr = torchaudio.load(ref_audio_path)
        # audio: (channels, nw)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        # audio: (1, nw)

        # 最大長にカット
        max_samples = int(max_duration * self.sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]

        # ref_text空ならASRで自動認識 (Whisper等)
        if ref_text == "":
            # faster_whisper or whisper で自動認識
            # ref_text = whisper_transcribe(audio, sr=self.sample_rate)
            raise ValueError("ref_textが空です。参照テキストを指定するかWhisperを設定してください。")

        return audio, ref_text

    def estimate_duration(
        self,
        ref_mel_len: int,
        ref_text: str,
        gen_text: str,
        speed: float = 1.0,
    ) -> int:
        """
        生成音声のDuration (mel長) を推定

        ========================================
        計算式
        ========================================
        ref_text_len = len(ref_text.encode("utf-8"))   # バイト長
        gen_text_len = len(gen_text.encode("utf-8"))

        duration = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        例:
            ref: 3秒音声 → ref_mel_len = 281
            ref_text: "Are you OK?" → 11バイト
            gen_text: "I'm fine, thank you!" → 21バイト

            duration = 281 + 281 * 21 / 11 / 1.0
                     = 281 + 536 = 817 フレーム
                     ≈ 8.7秒

        ========================================
        speed制御
        ========================================
        speed > 1.0: 速く話す (duration短縮)
        speed < 1.0: ゆっくり話す (duration延長)
        speed = 1.0: 参照音声と同じ話速 (デフォルト)
        """
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_mel_len + int(
            ref_mel_len / ref_text_len * gen_text_len / speed
        )
        return duration

    def generate(
        self,
        ref_audio: torch.Tensor,       # (1, nw) 参照音声
        ref_text: str,                   # 参照テキスト
        gen_text: str,                   # 生成テキスト
        *,
        nfe_steps: int = 32,             # ODE求解ステップ
        cfg_strength: float = 2.0,       # CFG強度
        sway_sampling_coef: float = -1.0, # Sway Sampling係数
        speed: float = 1.0,              # 話速制御
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        音声生成メイン関数

        ========================================
        Shape
        ========================================
        入力:
            ref_audio: (1, nw)  参照音声波形
            ref_text:  str      参照テキスト
            gen_text:  str      生成テキスト

        出力:
            waveform: (nw_gen,) 生成音声波形 (参照部分除外)
            sample_rate: int (24000)

        ========================================
        処理フロー
        ========================================
        """
        device = ref_audio.device

        # --- Step 1: Mel抽出 ---
        ref_mel = self.model.mel_spec(ref_audio)          # (1, nw) → (1, F, N_ref)
        ref_mel = ref_mel.permute(0, 2, 1)                 # (1, N_ref, F)
        ref_mel_len = ref_mel.shape[1]

        # --- Step 2: Duration推定 ---
        duration = self.estimate_duration(
            ref_mel_len, ref_text, gen_text, speed
        )

        # --- Step 3: テキスト準備 ---
        # 参照テキスト + 生成テキストを結合
        combined_text = ref_text + " " + gen_text

        # --- Step 4: ODE求解 ---
        generated, trajectory = self.model.sample(
            cond=ref_mel,                      # (1, N_ref, F)
            text=[combined_text],              # list[str]
            duration=duration,                 # int mel長
            steps=nfe_steps,                   # 32 NFE
            cfg_strength=cfg_strength,          # 2.0
            sway_sampling_coef=sway_sampling_coef,  # -1.0
            seed=seed,
            vocoder=self.vocoder,              # mel → waveform
        )
        # generated: (1, nw_total) 全体波形 (参照+生成)
        # trajectory: (33, 1, N, F) ODE軌跡

        # --- Step 5: 参照部分を除去 ---
        ref_nw = ref_mel_len * self.hop_length              # 参照部分のサンプル数
        gen_waveform = generated[0, ref_nw:]                 # (nw_gen,) 生成部分のみ

        return gen_waveform, self.sample_rate

    def generate_long_text(
        self,
        ref_audio: torch.Tensor,
        ref_text: str,
        gen_text: str,
        max_chars: int = 135,           # 1チャンクの最大文字数
        cross_fade_duration: float = 0.15,  # クロスフェード秒数
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        長文テキストのチャンク分割生成

        ========================================
        処理フロー
        ========================================
        1. gen_textを句点・読点で分割 (max_chars文字以下)
        2. 各チャンクを独立に generate()
        3. チャンク間をクロスフェードで結合

        例:
            gen_text (300文字) → 3チャンク (100文字ずつ)
            → generate() ×3
            → cross_fade_join() → 最終波形

        ========================================
        クロスフェード
        ========================================
        overlap_samples = cross_fade_duration * sample_rate
                        = 0.15 * 24000 = 3600 samples

        結合部分:
            fade_out = chunk_1[-overlap:] * (1 - t)
            fade_in  = chunk_2[:overlap] * t
            overlap  = fade_out + fade_in

        ========================================
        チャンク分割ルール
        ========================================
        1. 句点 (。.!?) で分割
        2. 読点 (、,;) で分割 (句点で分割しきれない場合)
        3. max_charsで強制分割 (それでも長い場合)
        """
        # テキスト分割
        chunks = split_text(gen_text, max_chars)

        # 各チャンク生成
        waveforms = []
        for chunk in chunks:
            wav, sr = self.generate(
                ref_audio, ref_text, chunk, **kwargs
            )
            waveforms.append(wav)

        # クロスフェード結合
        if len(waveforms) == 1:
            return waveforms[0], self.sample_rate

        result = waveforms[0]
        overlap = int(cross_fade_duration * self.sample_rate)
        for wav in waveforms[1:]:
            result = cross_fade_join(result, wav, overlap)

        return result, self.sample_rate


def split_text(text: str, max_chars: int = 135) -> List[str]:
    """
    テキストをチャンク分割

    優先順位:
    1. 句点 (。.!?) で分割
    2. 読点 (、,;:) で分割
    3. max_charsで強制分割
    """
    # 句読点での分割ロジック
    chunks = []
    current = ""
    for char in text:
        current += char
        if char in "。.!?！？" and len(current) >= max_chars // 3:
            chunks.append(current.strip())
            current = ""
        elif len(current) >= max_chars:
            # 最後の読点位置で分割
            last_punct = max(
                current.rfind("、"), current.rfind(","),
                current.rfind(";"), current.rfind(":")
            )
            if last_punct > max_chars // 3:
                chunks.append(current[:last_punct + 1].strip())
                current = current[last_punct + 1:]
            else:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def cross_fade_join(
    wav1: torch.Tensor,    # (n1,) 前の波形
    wav2: torch.Tensor,    # (n2,) 後の波形
    overlap: int,          # オーバーラップサンプル数
) -> torch.Tensor:
    """
    2つの波形をクロスフェードで結合

    ========================================
    Shape
    ========================================
    入力:
        wav1: (n1,) 前の波形
        wav2: (n2,) 後の波形
        overlap: int オーバーラップサンプル数 (通常3600 ≈ 0.15s)

    出力:
        result: (n1 + n2 - overlap,) 結合波形
    """
    t = torch.linspace(0, 1, overlap, device=wav1.device)  # (overlap,)

    # フェードアウト/イン
    fade_out = wav1[-overlap:] * (1 - t)     # (overlap,)
    fade_in = wav2[:overlap] * t             # (overlap,)

    # 結合
    result = torch.cat([
        wav1[:-overlap],                     # 前の波形 (非オーバーラップ)
        fade_out + fade_in,                  # クロスフェード部分
        wav2[overlap:],                      # 後の波形 (非オーバーラップ)
    ])
    return result


# ============================================================
# Sway Sampling 詳細
# ============================================================

def sway_sampling_timesteps(
    steps: int = 32,
    s: float = -1.0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Sway Sampling タイムステップ生成

    ========================================
    Shape
    ========================================
    入力:
        steps: int ODE求解ステップ数 (NFE)
        s: float Sway係数
            s < 0: 左寄り (初期密) ← F5-TTSデフォルト (s=-1)
            s = 0: 均一 (通常サンプリング)
            s > 0: 右寄り (後期密)

    出力:
        t: (steps+1,) タイムステップ列 [0, ..., 1]

    ========================================
    数式
    ========================================
    f_sway(u; s) = u + s * (cos(π/2 * u) - 1 + u)

    この関数は単調増加で [0,1] → [0,1] の写像。
    s の有効範囲: [-1, 2/(π-2)] ≈ [-1, 1.75]

    ========================================
    直感
    ========================================
    s=-1 の場合:
    - cos(π/2 * u) は u=0 で 1, u=1 で 0
    - f(u) = u + (-1)*(cos(π/2*u) - 1 + u)
           = u - cos(π/2*u) + 1 - u
           = 1 - cos(π/2*u)

    u=0.0 → t=0.000  (変化なし)
    u=0.1 → t=0.012  (大幅圧縮: 初期密度↑)
    u=0.2 → t=0.049
    u=0.3 → t=0.109
    u=0.5 → t=0.293  (中間で約半分)
    u=0.7 → t=0.541
    u=0.9 → t=0.844
    u=1.0 → t=1.000  (変化なし)

    → 初期ステップ (t≈0) が密に、後期ステップ (t≈1) が疎に
    → テキスト-音声アラインメント (初期で決定) の精度向上
    """
    u = torch.linspace(0, 1, steps + 1, device=device)  # 均一サンプリング
    t = u + s * (torch.cos(torch.pi / 2 * u) - 1 + u)    # Sway変換
    return t


# ============================================================
# CFG (Classifier-Free Guidance) 詳細
# ============================================================

def cfg_inference_detail():
    """
    CFG推論の詳細説明

    ========================================
    学習時のCFGドロップ (2段階)
    ========================================

    Stage 1: 音声条件ドロップ (p=0.3)
        if random() < 0.3:
            cond = zeros  # 参照音声をゼロに
            # テキストは保持
        → モデルは「テキストだけから生成する能力」を学習

    Stage 2: 全条件ドロップ (p=0.2) [Stage 1を上書き]
        if random() < 0.2:
            cond = zeros  # 音声ゼロ
            text = zeros  # テキストもゼロ
        → モデルは「完全に無条件で生成する能力」を学習

    確率テーブル:
        P(cond=mel, text=text) = (1-0.3) * (1-0.2) = 0.56  (通常学習)
        P(cond=0, text=text)   = 0.3 * (1-0.2)     = 0.24  (音声のみドロップ)
        P(cond=0, text=0)      = 0.2                = 0.20  (全ドロップ)

    ========================================
    推論時のCFG外挿
    ========================================

    1. バッチ2倍化:
        x_input = cat([x, x])                    # (2B, N, F)
        cond_input = cat([cond, zeros])           # (2B, N, F)
        text_input = cat([text, zeros])           # (2B, nt)

    2. DiT予測:
        pred = DiT(x_input, cond_input, text_input, t)  # (2B, N, F)
        v_cond, v_uncond = pred.chunk(2)                  # 各 (B, N, F)

    3. CFG外挿:
        v = v_cond + α * (v_cond - v_uncond)
        # α=2.0: 条件方向を3倍に増幅
        # v_cond - v_uncond: 「条件ありとなしの差」= 条件の効果

    ========================================
    CFG強度αの効果
    ========================================
    α=0:   v = v_uncond            (無条件生成、多様だが不忠実)
    α=1:   v = 2*v_cond - v_uncond (標準CFG)
    α=2:   v = 3*v_cond - 2*v_uncond (F5-TTSデフォルト、忠実度高)
    α>3:   過剰CFG (アーティファクト・ロボット声リスク)
    """
    # この関数はドキュメント専用 (docstringに処理の詳細説明を含む)


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMADetail:
    """
    EMA (Exponential Moving Average) の詳細

    ========================================
    数式
    ========================================
    θ_ema ← decay * θ_ema + (1 - decay) * θ_model

    decay = 0.9999 (F5-TTSデフォルト)
    → θ_ema は θ_model の指数移動平均

    ========================================
    効果
    ========================================
    - 学習中のパラメータ振動を平滑化
    - 汎化性能の向上 (特にTTSで重要)
    - 推論時はEMAパラメータを使用

    ========================================
    保存/読込
    ========================================
    チェックポイント:
        - model_state_dict: 学習中のパラメータ
        - ema_state_dict: EMAパラメータ ← 推論に使用
        - optimizer_state_dict: オプティマイザ状態
        - scheduler_state_dict: スケジューラ状態
        - update: 現在のアップデート数
    """
    # この関数はドキュメント専用 (docstringにEMAの詳細説明を含む)


# ============================================================
# 高レベルAPI
# ============================================================

class F5TTSApi:
    """
    F5-TTS 高レベルAPI

    対応: src/f5_tts/api.py

    ========================================
    使用例
    ========================================
    tts = F5TTSApi(model="F5TTS_v1_Base")  # 自動ダウンロード
    waveform, sr, mel = tts.infer(
        ref_file="prompt.wav",
        ref_text="Are you OK?",
        gen_text="I'm fine, thank you very much!",
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
    )
    # waveform: numpy array (nw,)
    # sr: 24000
    """

    def __init__(
        self,
        model: str = "F5TTS_v1_Base",
        ckpt_file: str = "",
        device: Optional[str] = None,
    ):
        # 自動デバイス検出 (CUDA > XPU > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = "xpu"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # モデル読み込み (HuggingFace自動ダウンロード or ローカルチェックポイント)
        # from f5_tts.model import CFM, DiT
        # self.model = CFM(DiT(**F5TTS_v1_Base_config), mel_spec_kwargs=...)
        # ckpt = torch.load(ckpt_file or download_from_hf(model))
        # self.model.load_state_dict(ckpt["ema_model_state_dict"])  # EMAパラメータ使用
        # self.model.to(self.device).eval()

        # Vocoder (Vocos) 読み込み
        # from vocos import Vocos
        # self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        self.pipeline = InferencePipeline(
            model=None,  # 上記で初期化したモデル
            vocoder=None,  # 上記で初期化したVocoder
        )

    def infer(
        self,
        ref_file: str,          # 参照音声パス
        ref_text: str,          # 参照テキスト
        gen_text: str,          # 生成テキスト
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        seed: int = -1,
    ):
        """
        高レベル推論インターフェース

        ========================================
        処理フロー
        ========================================
        1. ref_audio前処理 (リサンプル、無音除去、最大12秒)
        2. ref_text空ならWhisper認識
        3. gen_textが長い場合チャンク分割 (~135文字)
        4. 各チャンク生成 (ODE + Sway + CFG)
        5. クロスフェード結合
        6. RMS正規化 (target_rms=0.1)
        7. (waveform, sample_rate, mel) 返却
        """
        # Step 1-2: 参照音声前処理
        ref_audio, ref_text = self.pipeline.preprocess_ref_audio(ref_file, ref_text)
        # ref_audio: (1, nw_ref), ref_text: str

        # Step 3-5: 長文ならチャンク分割、短文ならそのまま生成
        actual_seed = seed if seed >= 0 else None
        if len(gen_text) > 135:
            waveform, sr = self.pipeline.generate_long_text(
                ref_audio, ref_text, gen_text,
                nfe_steps=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, speed=speed,
                seed=actual_seed,
            )
        else:
            waveform, sr = self.pipeline.generate(
                ref_audio, ref_text, gen_text,
                nfe_steps=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, speed=speed,
                seed=actual_seed,
            )
        # waveform: (nw_gen,)

        # Step 6: RMS正規化
        target_rms = 0.1
        current_rms = waveform.square().mean().sqrt()
        if current_rms > 0:
            waveform = waveform * (target_rms / current_rms)

        # Step 7: numpy変換して返却
        return waveform.cpu().numpy(), sr, None


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=== F5-TTS Training & Inference Pipeline ===")
    print()
    print("学習設定 (Base Model):")
    print("  データ: Emilia ~95K時間 (英語+中国語)")
    print("  バッチ: 307,200 フレーム (8 GPU × 38,400)")
    print("  最適化: AdamW, lr=7.5e-5, warmup=20K")
    print("  更新: ~1.2M updates, 11 epochs")
    print("  GPU: 8 × NVIDIA A100 80G, > 1週間")
    print()
    print("推論設定:")
    print("  NFE: 32ステップ (16でRTF=0.15)")
    print("  CFG: α=2.0 (条件強調)")
    print("  Sway: s=-1 (初期ステップ重視)")
    print("  Vocoder: Vocos (24kHz mel → waveform)")
    print("  速度: RTF ≈ 0.15 (10秒音声を1.5秒で生成)")
    print()
    print("Sway Sampling (s=-1):")
    t = sway_sampling_timesteps(steps=8, s=-1.0)
    for i, ti in enumerate(t):
        u = i / 8
        print(f"  u={u:.3f} → t={ti.item():.4f}")
    print()
    print("推論パラメータ比較:")
    print("  | 設定 | NFE | CFG | Sway | RTF | WER | SIM-o |")
    print("  |------|-----|-----|------|-----|-----|-------|")
    print("  | 高品質| 32  | 2.0 | -1   | 0.31| 2.42| 0.66  |")
    print("  | 高速  | 16  | 2.0 | -1   | 0.15| 2.53| 0.66  |")
    print("  | 最高品| 32  | 2.0 | -0.8 | 0.31| 2.42| 0.66  |")
    print("  | CFGなし| 32 | 0.0 | -1   | 0.16| ↑   | ↓     |")
