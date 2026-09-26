"""Parakeet TDT-CTC (ja) の最小ファインチューニング・スケッチ (holdout評価つき)。

方針: NeMoに依存するのは「モデルインスタンス化」と、その構成部品
(preprocessor / tokenizer / decoder / joint / loss(TDT) / ctc_decoder / ctc_loss)
を関数として呼ぶことだけ。Dataset / DataLoader / 学習ループ / optimizer は
全て素のPyTorchで書く。NeMoの Trainer / PyTorch Lightning / Hydra 設定は使わない。

学習lossは、NeMo公式の EncDecHybridRNNTCTCModel.training_step と同じ式:
    total_loss = (1 - ctc_loss_weight) * tdt_loss + ctc_loss_weight * ctc_loss
(ctc_loss_weight はモデル自身の cfg から読む。このモデルでは 0.3)
TDT lossもCTC lossと同様、NeMoが既に構築済みのコンポーネント(model.loss)を
呼ぶだけで済み、自前で再実装する必要はない。CTC lossだけで学習すると、
共有エンコーダーがTDTデコーダー(凍結されたまま)の想定する分布からずれていき、
TDT側(=実運用のtranscribe())の精度がむしろ悪化する現象が実際に観測されたため、
両lossを合成する公式の方式に修正した。

学習に使わない holdout(test)セットを分けておき、エポックごとに
(1) model.transcribe() によるTDTデコード経路のreading CER
(2) CTC分岐のgreedyデコードによるreading CER
の両方を計測し、trainのlossと合わせて推移を確認する。

環境メモ: numba-cuda 0.30.4 が numpy>=2.0 で削除された np.row_stack に依存しているため
importエラーになる場合がある。冒頭で np.row_stack = np.vstack の互換シムを当てて回避している
(挙動はvstackと同一なので数値的な影響はない)。これによりTDT lossはNeMo本来の高速な
numba CUDAカーネル実装のまま使える(精度・速度とも公式実装と同一)。

使い方:
    uv run python scripts/finetune_parakeet_ctc.py --dataset jsut \\
        --num-train 32 --num-holdout 16 --epochs 20 --lr 1e-5
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

# numba-cuda 0.30.4 が numpy>=2.0 で削除された np.row_stack (vstackの旧エイリアス) に
# まだ依存しているための互換シム。挙動はvstackと完全に同一なので数値的な影響はない。
# numba-cudaのimport(=nemoのimport)より前に当てる必要がある。
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import argparse
import logging
import random
from functools import partial

import jiwer
import librosa
import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from compute_cer import normalize, to_reading
from datasets_ja import DATASETS

_nemo_logger = logging.getLogger("nemo_logger")
_nemo_logger.addFilter(lambda record: record.levelno >= logging.ERROR)

MODEL_NAME = "nvidia/parakeet-tdt_ctc-0.6b-ja"
TARGET_SR = 16000


class AudioTextDataset(Dataset):
    """(波形, テキスト) のペアを返すだけの素のPyTorch Dataset。"""

    def __init__(self, samples: list[tuple[str, "os.PathLike", str]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, audio_path, text = self.samples[idx]
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        return waveform, text


def collate_fn(batch, tokenizer):
    waveforms, texts = zip(*batch)

    signal_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    max_signal_len = int(signal_lengths.max())
    signal_batch = torch.zeros(len(waveforms), max_signal_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        signal_batch[i, : len(w)] = torch.from_numpy(w)

    token_ids = [tokenizer.text_to_ids(t) for t in texts]
    target_lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long)
    max_target_len = int(target_lengths.max())
    target_batch = torch.zeros(len(token_ids), max_target_len, dtype=torch.long)
    for i, ids in enumerate(token_ids):
        target_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return signal_batch, signal_lengths, target_batch, target_lengths


def ctc_greedy_decode(log_probs, lengths, tokenizer) -> list[str]:
    """CTC分岐のgreedyデコード(argmax→連続重複除去→blank除去)。blankは最終インデックス。"""
    blank_id = log_probs.shape[-1] - 1
    pred_ids = log_probs.argmax(dim=-1)

    texts = []
    for i in range(pred_ids.size(0)):
        ids = pred_ids[i, : lengths[i]].tolist()
        collapsed = []
        prev = None
        for tok_id in ids:
            if tok_id != prev and tok_id != blank_id:
                collapsed.append(tok_id)
            prev = tok_id
        texts.append(tokenizer.ids_to_text(collapsed))
    return texts


def compute_reading_cer(gts: list[str], preds: list[str]) -> float:
    norm_gt = [to_reading(normalize(t)) for t in gts]
    norm_pred = [to_reading(normalize(t)) for t in preds]
    return jiwer.cer(norm_gt, norm_pred)


def evaluate(model, holdout_samples, device, eval_batch_size: int) -> tuple[float, float]:
    """holdoutセットを (TDTデコード経路, CTC greedyデコード経路) の両方で評価する。"""
    model.eval()
    audio_paths = [str(audio_path) for _, audio_path, _ in holdout_samples]
    gts = [text for _, _, text in holdout_samples]

    with torch.no_grad():
        # TDT側: 実運用のtranscribe()と同じ経路(バッチ単位で回して進捗を見えるようにする)
        tdt_preds = []
        for i in tqdm(range(0, len(audio_paths), eval_batch_size), desc="eval(TDT)", leave=False):
            batch_paths = audio_paths[i : i + eval_batch_size]
            outputs = model.transcribe(batch_paths, batch_size=len(batch_paths), verbose=False)
            tdt_preds.extend(o.text if hasattr(o, "text") else str(o) for o in outputs)

        # CTC側: エンコーダー forward + ctc_decoder + greedy decode
        ctc_preds = []
        for i in tqdm(range(0, len(holdout_samples), eval_batch_size), desc="eval(CTC)", leave=False):
            batch = holdout_samples[i : i + eval_batch_size]
            waveforms = []
            for _, audio_path, _ in batch:
                w, sr = sf.read(str(audio_path), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if sr != TARGET_SR:
                    w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
                waveforms.append(w)

            signal_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
            signal_batch = torch.zeros(len(waveforms), int(signal_lengths.max()), dtype=torch.float32)
            for j, w in enumerate(waveforms):
                signal_batch[j, : len(w)] = torch.from_numpy(w)

            signal_batch, signal_lengths = signal_batch.to(device), signal_lengths.to(device)
            processed_signal, processed_signal_len = model.preprocessor(
                input_signal=signal_batch, length=signal_lengths
            )
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, length=processed_signal_len
            )
            log_probs = model.ctc_decoder(encoder_output=encoded)
            ctc_preds.extend(ctc_greedy_decode(log_probs, encoded_len, model.tokenizer))

    tdt_cer = compute_reading_cer(gts, tdt_preds)
    ctc_cer = compute_reading_cer(gts, ctc_preds)

    model.train()
    return tdt_cer, ctc_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="jsut")
    parser.add_argument("--num-train", type=int, default=32, help="学習に使うサンプル数")
    parser.add_argument("--num-holdout", type=int, default=16, help="評価専用(未学習)のサンプル数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1, help="何エポックごとにholdout評価するか")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size", type=int, default=16, help="holdout評価時のバッチサイズ"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="warmup後に到達するピークLR")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="全ステップ数に対するwarmup期間の割合"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_samples = DATASETS[args.dataset]()
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    train_samples = all_samples[: args.num_train]
    holdout_samples = all_samples[args.num_train : args.num_train + args.num_holdout]
    assert not set(s[0] for s in train_samples) & set(s[0] for s in holdout_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ここだけNeMo依存: モデルインスタンス化と、その構成部品を関数として呼ぶ。
    # Trainer/Lightning/Hydraは使わない。
    print(f"loading {MODEL_NAME} on {device} ...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    model = model.to(device)
    model.train()

    ctc_loss_weight = model.ctc_loss_weight
    print(f"ctc_loss_weight (モデル自身のcfgより): {ctc_loss_weight}")

    # dither(学習時ノイズ付加)を無効化し、lossの変化を素直に観測できるようにする。
    model.preprocessor.featurizer.dither = 0.0

    train_dataset = AudioTextDataset(train_samples)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=model.tokenizer),
    )

    # encoder + TDT側(decoder/joint) + CTC側(ctc_decoder) を全て更新する。
    params = (
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.joint.parameters())
        + list(model.ctc_decoder.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # warmup(線形) + cosine decay。標準PyTorchの SequentialLR で連結するだけで、
    # NeMo/Trainer側のスケジューラー機構には一切依存しない。ステップ単位で更新する。
    total_steps = args.epochs * len(loader)
    warmup_steps = min(max(1, int(total_steps * args.warmup_ratio)), total_steps - 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    print(f"train: {len(train_samples)} samples / holdout: {len(holdout_samples)} samples")
    print(f"total steps: {total_steps} (warmup: {warmup_steps})")

    init_tdt_cer, init_ctc_cer = evaluate(model, holdout_samples, device, args.eval_batch_size)
    print(
        f"epoch  -  train_loss   -  holdout reading CER"
        f"  TDT {init_tdt_cer * 100:.2f}% / CTC {init_ctc_cer * 100:.2f}% (finetune前)"
    )

    progress = tqdm(total=total_steps, desc="finetuning")
    for epoch in range(args.epochs):
        epoch_losses = []
        for signal, signal_len, targets, target_len in loader:
            signal, signal_len = signal.to(device), signal_len.to(device)
            targets, target_len = targets.to(device), target_len.to(device)

            processed_signal, processed_signal_len = model.preprocessor(
                input_signal=signal, length=signal_len
            )
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, length=processed_signal_len
            )

            # TDT側: 予測ネットワーク(decoder) → joint → TDT loss(numba CUDAカーネル、高速)
            # joint.fuse_loss_wer が有効な場合、joint呼び出し内でlossまで計算される
            # (メモリ効率化のための融合カーネル)。公式のtraining_stepと同じ分岐で対応する。
            decoder_out, target_length, _ = model.decoder(targets=targets, target_length=target_len)
            if not model.joint.fuse_loss_wer:
                joint_out = model.joint(encoder_outputs=encoded, decoder_outputs=decoder_out)
                tdt_loss = model.loss(
                    log_probs=joint_out,
                    targets=targets,
                    input_lengths=encoded_len,
                    target_lengths=target_length,
                )
            else:
                tdt_loss, _, _, _ = model.joint(
                    encoder_outputs=encoded,
                    decoder_outputs=decoder_out,
                    encoder_lengths=encoded_len,
                    transcripts=targets,
                    transcript_lengths=target_length,
                    compute_wer=False,
                )

            # CTC側
            ctc_log_probs = model.ctc_decoder(encoder_output=encoded)
            ctc_loss = model.ctc_loss(
                log_probs=ctc_log_probs,
                targets=targets,
                input_lengths=encoded_len,
                target_lengths=target_len,
            )

            # NeMo公式のHybrid学習と同じ重み付き合成
            loss = (1 - ctc_loss_weight) * tdt_loss + ctc_loss_weight * ctc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

            progress.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            progress.update(1)

        mean_loss = sum(epoch_losses) / len(epoch_losses)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            tdt_cer, ctc_cer = evaluate(model, holdout_samples, device, args.eval_batch_size)
            progress.write(
                f"epoch {epoch:3d}  train_loss {mean_loss:.4f}"
                f"  holdout reading CER  TDT {tdt_cer * 100:.2f}% / CTC {ctc_cer * 100:.2f}%"
            )
        else:
            progress.write(f"epoch {epoch:3d}  train_loss {mean_loss:.4f}")
    progress.close()


if __name__ == "__main__":
    main()
