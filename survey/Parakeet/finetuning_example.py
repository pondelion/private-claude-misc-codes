"""DataFrame から Parakeet Hybrid TDT-CTC を fine-tune する実用例。

NeMo Trainer/Lightning/Hydra は使わず、公式 checkpoint 内の preprocessor,
FastConformer encoder, prediction network, joint, TDT loss, CTC head/lossだけを呼ぶ。

DataFrame columns:
  audio (str, required): audio path
  text  (str, required): transcript

CLI: python finetuning_example.py --train train.csv --valid valid.csv --output parakeet_ft.pt
JSON/JSONL/Parquetにも対応。既定モデルは nvidia/parakeet-tdt_ctc-0.6b-ja。
"""
from __future__ import annotations
import argparse
import os
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if not hasattr(np, "row_stack"):  # numba-cuda 0.30.x + NumPy 2 compatibility
    np.row_stack = np.vstack


def read_dataframe(path: str) -> pd.DataFrame:
    """拡張子からreaderを選び、共通してDataFrameを返す。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv": return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}: return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".parquet", ".pq"}: return pd.read_parquet(path)
    raise ValueError(f"unsupported metadata format: {suffix}")


class AudioTextDataFrameDataset(Dataset):
    """metadataだけを保持する軽量Dataset。

    worker processで並列化できるよう、音声I/Oは``__getitem__``ではなくcollate時に行う。
    DataFrameをcopy/resetし、呼出側のindexや後続変更に依存しないようにする。
    """
    def __init__(self, df: pd.DataFrame):
        missing = {"audio", "text"} - set(df.columns)
        if missing: raise ValueError(f"missing DataFrame columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"audio": str(row.audio), "text": str(row.text)}


def load_audio(path: str, target_sr=16000) -> torch.Tensor:
    """任意channel/rateの音声をmono float32 ``(S,)``へ統一する。

    ``always_2d``により入力を``(S,C)``へ揃え、channel平均でmono化する。sample rateが
    checkpointと異なる場合はPyTorch線形補間を用いる。高品質resamplerより簡潔だが、
    Dataset処理を外部frameworkへ隠さず理解できる構成を優先している。
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(data).mean(1)
    if sr != target_sr:
        size = round(x.numel() * target_sr / sr)
        x = torch.nn.functional.interpolate(x[None,None], size=size, mode="linear", align_corners=False)[0,0]
    return x


def collate_batch(rows, tokenizer, sample_rate=16000):
    """可変長音声とtokenをzero paddingしてdense batchへまとめる。

    出力はsignals ``(B,Smax)``、signal_lengths ``(B,)``、targets ``(B,Umax)``、
    target_lengths ``(B,)``、評価用raw text list。lossはlengthを使うのでpadding値自体は無視される。
    """
    waves = [load_audio(row["audio"], sample_rate) for row in rows]
    signal_lengths = torch.tensor([x.numel() for x in waves], dtype=torch.long)
    # waveform padding。lengthがあるため、0を実際の無音教師として誤解しない。
    signals = torch.zeros(len(rows), int(signal_lengths.max()))
    for i, x in enumerate(waves): signals[i,:x.numel()] = x
    # checkpoint自身のtokenizerを使い、語彙IDとdecoder headを必ず一致させる。
    ids = [tokenizer.text_to_ids(row["text"]) for row in rows]
    target_lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    targets = torch.zeros(len(rows), int(target_lengths.max()), dtype=torch.long)
    for i, x in enumerate(ids): targets[i,:len(x)] = torch.tensor(x)
    return signals, signal_lengths, targets, target_lengths, [r["text"] for r in rows]


def compute_batch_losses(model, batch, device):
    """Exact official Hybrid training path and shapes.

    waveform (B,S) -> mel (B,128,Tm) -> encoder (B,D,Te), Te≈Tm/8;
    targets (B,U) -> decoder (B,H,U+1); joint internally represents (B,Te,U+1,V+1+K).
    """
    signal, signal_len, targets, target_len, _ = batch
    signal, signal_len = signal.to(device), signal_len.to(device)
    targets, target_len = targets.to(device), target_len.to(device)
    # (B,S) -> (B,F,Tm)。NeMo encoderはchannel-first featureを受け取る。
    processed, processed_len = model.preprocessor(input_signal=signal, length=signal_len)
    if hasattr(model, "spec_augmentation") and model.spec_augmentation is not None:
        processed = model.spec_augmentation(input_spec=processed, length=processed_len)
    # FastConformer: (B,F,Tm) -> (B,D,Te)。教材版encoderとは軸順が異なる点に注意。
    encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
    decoder_out, decoder_target_len, _ = model.decoder(targets=targets, target_length=target_len)
    # TDTのfull jointはB*Te*(U+1)*(V+1+K)で非常に大きい。fused modeでは
    # model内部がsub-batchごとにjoint→lossを計算し、巨大tensorの常駐を避ける。
    if model.joint.fuse_loss_wer:
        tdt, _, _, _ = model.joint(encoder_outputs=encoded, decoder_outputs=decoder_out,
                                   encoder_lengths=encoded_len, transcripts=targets,
                                   transcript_lengths=decoder_target_len, compute_wer=False)
    else:
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder_out)
        tdt = model.loss(log_probs=joint, targets=targets, input_lengths=encoded_len,
                         target_lengths=decoder_target_len)
    # CTC branchは同じencodedから直接frame分布を作り、prediction networkを通らない。
    ctc_lp = model.ctc_decoder(encoder_output=encoded)
    ctc = model.ctc_loss(log_probs=ctc_lp, targets=targets, input_lengths=encoded_len,
                         target_lengths=target_len)
    total = (1-model.ctc_loss_weight)*tdt + model.ctc_loss_weight*ctc
    return total, tdt.detach(), ctc.detach()


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """dropout/augmentationを止め、勾配なしでvalidation mean lossを計算する。"""
    model.eval(); totals = []
    for batch in loader: totals.append(float(compute_batch_losses(model, batch, device)[0]))
    model.train()
    return sum(totals)/max(1,len(totals))


def train(train_df, valid_df=None, model_name="nvidia/parakeet-tdt_ctc-0.6b-ja",
          output="parakeet_finetuned.pt", epochs=3, batch_size=4, grad_acc=1,
          lr=2e-5, weight_decay=1e-3, max_grad_norm=1.0, num_workers=0, device=None):
    """encoder、TDT decoder/joint、CTC headをすべてfine-tuneする。

    公式modelの部品は利用するが、Trainer/Lightningに制御を委譲せず、optimizer updateの
    順序を明示する。戻り値は学習済みmodel、保存物はmodel名とstate_dictを含むcheckpoint。
    """
    import nemo.collections.asr as nemo_asr
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint configからtokenizer、loss、duration集合、CTC weightも復元される。
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).to(device)
    # 小規模fine-tuneで追加noiseが再現性を損なわないようditherを無効化する。
    model.preprocessor.featurizer.dither = 0.0
    model.train()
    collate = partial(collate_batch, tokenizer=model.tokenizer, sample_rate=int(model.cfg.sample_rate))
    train_loader = DataLoader(AudioTextDataFrameDataset(train_df), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate, pin_memory=device.startswith("cuda"))
    valid_loader = None if valid_df is None else DataLoader(AudioTextDataFrameDataset(valid_df),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    # preprocessorには通常学習parameterがない。両decode経路に関係する4 moduleを更新する。
    params = (list(model.encoder.parameters()) + list(model.decoder.parameters())
              + list(model.joint.parameters()) + list(model.ctc_decoder.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(train_loader) // grad_acc)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/20)
    optimizer.zero_grad(set_to_none=True)
    update = 0
    for epoch in range(epochs):
        sums = [0.0,0.0,0.0]
        for i, batch in enumerate(train_loader):
            total, tdt, ctc = compute_batch_losses(model, batch, device)
            # gradient accumulation時はlossを分割し、update当たりの勾配scaleを一定に保つ。
            (total/grad_acc).backward()
            for j,x in enumerate((total,tdt,ctc)): sums[j] += float(x.detach())
            boundary = (i+1) % grad_acc == 0 or i+1 == len(train_loader)
            if boundary:
                # clipping後に1 updateし、その単位でschedulerも1 step進める。
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); update += 1
        n = max(1,len(train_loader)); message = f"epoch={epoch+1} total={sums[0]/n:.4f} tdt={sums[1]/n:.4f} ctc={sums[2]/n:.4f}"
        if valid_loader is not None: message += f" valid={evaluate_loss(model, valid_loader, device):.4f}"
        print(message)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "optimizer_updates": update}, output)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--valid")
    p.add_argument("--model-name", default="nvidia/parakeet-tdt_ctc-0.6b-ja")
    p.add_argument("--output", default="parakeet_finetuned.pt")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-acc", type=int, default=1); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()
    train(read_dataframe(a.train), None if not a.valid else read_dataframe(a.valid), a.model_name,
          a.output, a.epochs, a.batch_size, a.grad_acc, a.lr, num_workers=a.num_workers)


if __name__ == "__main__": main()
