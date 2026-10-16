"""MDLM/BD3-LMの学習・推論を一巡する例。実行: ``python main_flow.py``。"""

from __future__ import annotations

import torch

from bd3lm import bd3lm_forward_and_loss
from mdlm import corrupt_tokens, mdlm_loss
from model import TinyDiffusionLM
from noise_schedule import LinearSchedule
from sampling import sample_bd3lm, sample_mdlm


def main() -> None:
    torch.manual_seed(7)
    vocab_size, mask_id = 31, 30
    model = TinyDiffusionLM(vocab_size, max_length=64, width=48, layers=2, heads=4)
    schedule = LinearSchedule()

    # 0..2はprompt、3..11はresponseというSFT例。lossはresponseだけに課す。
    clean = torch.randint(0, mask_id, (2, 12))                 # (B=2,L=12)
    eligible = torch.zeros_like(clean, dtype=torch.bool)       # (2,12)
    eligible[:, 3:] = True
    corruption = corrupt_tokens(clean, eligible, mask_id, schedule)

    # MDLM: noisyな同じ長さの系列を双方向attentionでclean-token分類する。
    logits = model(corruption.noisy_ids)                       # (2,12,V=31)
    md_loss, md_stats = mdlm_loss(logits, clean, eligible, corruption, schedule)
    md_loss.backward()
    assert torch.isfinite(md_loss)

    # BD3-LM: [noisy, clean]という2L入力と専用maskで、過去clean blockだけを条件にする。
    model.zero_grad(set_to_none=True)
    bd_loss, bd_stats = bd3lm_forward_and_loss(
        model, clean, eligible, corruption, schedule, block_size=4
    )
    bd_loss.backward()
    assert torch.isfinite(bd_loss)

    prompt = clean[:1, :3]                                    # (1,P=3)
    md_out = sample_mdlm(model, prompt, mask_id, generation_length=8, steps=4)
    bd_out = sample_bd3lm(model, prompt, mask_id, generation_length=8,
                          block_size=4, steps_per_block=2)
    assert not md_out.sequences.eq(mask_id).any()
    assert not bd_out.sequences.eq(mask_id).any()
    print({"mdlm_loss": float(md_loss.detach()), "bd3lm_loss": float(bd_loss.detach()),
           "masked_fraction": float(md_stats["masked_fraction"]),
           "bd3_concat_length": int(bd_stats["concat_length"]),
           "mdlm_shape": tuple(md_out.sequences.shape),
           "bd3lm_shape": tuple(bd_out.sequences.shape)})


if __name__ == "__main__":
    main()
