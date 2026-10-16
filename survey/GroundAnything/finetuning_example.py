"""DataFrameから理解用GroundAnything decoderを実際に学習する素のPyTorch例。

DataFrame columns:
  input_ids: prompt+serialized responseのtoken ID list
  response_start: response先頭index（それ以前のpromptはloss対象外）

実checkpointではprocessorが画像をvisual tokenへ変換する。本例はAR-to-Diffusion objectiveの
学習loopを独立に検証する目的でpre-tokenized sequenceを受け、Trainer等を使わない。
"""
from __future__ import annotations
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model_blocks import TinyGroundAnythingDecoder
from block_diffusion import complementary_corruption, conversion_attention_mask, conversion_loss


class GroundingDataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        missing = {"input_ids", "response_start"} - set(df.columns)
        if missing: raise ValueError(f"missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        ids = ast.literal_eval(row.input_ids) if isinstance(row.input_ids, str) else list(row.input_ids)
        start = int(row.response_start)
        if not 0 < start < len(ids): raise ValueError("response_start must split prompt and response")
        return torch.tensor(ids, dtype=torch.long), start


def collate_rows(rows, pad_id: int):
    """可変長sequenceを``tokens/response_mask/valid_mask (B,Tmax)``へpaddingする。"""
    maximum = max(len(ids) for ids,_ in rows)
    tokens = torch.full((len(rows), maximum), pad_id, dtype=torch.long)
    response = torch.zeros_like(tokens, dtype=torch.bool)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    for b,(ids,start) in enumerate(rows):
        tokens[b,:len(ids)] = ids; valid[b,:len(ids)] = True; response[b,start:len(ids)] = True
    return tokens, response, valid


def train(train_df, vocab_size, mask_id, eos_id, pad_id=0, epochs=3, batch_size=4, lr=2e-4, output="groundanything_tiny.pt"):
    """各sampleのcomplementary corruptionを毎step再sampleしてjoint objectiveを最適化する。"""
    model = TinyGroundAnythingDecoder(vocab_size)
    loader = DataLoader(GroundingDataFrameDataset(train_df), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda rows: collate_rows(rows, pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for tokens,response,valid in loader:
            sample_losses = []
            for b in range(tokens.shape[0]):
                length = int(valid[b].sum()); clean=tokens[b,:length]; supervised=response[b,:length]
                views = complementary_corruption(clean, supervised, mask_id, eos_id)
                noisy_mask = conversion_attention_mask(views.block_ids, supervised)
                causal = torch.ones(length,length,dtype=torch.bool).tril()
                clean_logits = model(token_ids=clean[None], allowed_attention=causal)[0]
                logits1 = model(token_ids=views.view1[None], allowed_attention=noisy_mask)[0]
                logits2 = model(token_ids=views.view2[None], allowed_attention=noisy_mask)[0]
                sample_losses.append(conversion_loss(clean_logits, logits1, logits2, clean, supervised,
                                                     views.masked1, views.masked2)["loss"])
            loss = torch.stack(sample_losses).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach())
        print(f"epoch={epoch+1} loss={total/max(1,len(loader)):.4f}")
    torch.save(model.state_dict(), output)
    return model


if __name__ == "__main__":
    # 実際の利用ではtokenizerで作ったID列をDataFrameへ格納する。
    demo = pd.DataFrame({"input_ids": [[1,2,10,11,12,63],[1,3,20,21,63]], "response_start": [2,2]})
    train(demo, vocab_size=65, mask_id=64, eos_id=63, epochs=1, batch_size=2)
