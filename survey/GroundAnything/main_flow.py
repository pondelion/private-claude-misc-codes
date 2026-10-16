"""corruption→shared decoder→AR/MDM loss→backward→diffusion decodeを一度に実行する。"""
import torch
from model_blocks import TinyGroundAnythingDecoder
from block_diffusion import complementary_corruption, conversion_attention_mask, conversion_loss
from decoding import entropy_guided_decode


def example_usage():
    torch.manual_seed(7)
    vocab, mask_id, eos_id = 128, 127, 126
    # [prompt 4 tokens | response 8 tokens]。最後のresponse tokenがEOS。
    clean = torch.tensor([4,5,6,7, 20,21,22,23,24,25,26,eos_id])
    response = torch.arange(clean.numel()) >= 4
    views = complementary_corruption(clean, response, mask_id, eos_id, block_size=4)
    allowed = conversion_attention_mask(views.block_ids, response)
    model = TinyGroundAnythingDecoder(vocab, d_model=64, n_heads=4, n_layers=2)
    causal = torch.ones(len(clean), len(clean), dtype=torch.bool).tril()
    clean_logits = model(token_ids=clean[None], allowed_attention=causal)[0]
    noisy1 = model(token_ids=views.view1[None], allowed_attention=allowed)[0]
    noisy2 = model(token_ids=views.view2[None], allowed_attention=allowed)[0]
    losses = conversion_loss(clean_logits, noisy1, noisy2, clean, response, views.masked1, views.masked2)
    losses["loss"].backward()
    assert model.token_embedding.weight.grad is not None
    generated = entropy_guided_decode(model, clean[:4][None], mask_id, eos_id,
                                      block_size=4, entropy_threshold=4.0, max_blocks=2)
    print({k: float(v.detach()) for k,v in losses.items()})
    print("generated token ids:", generated.tolist())


if __name__ == "__main__": example_usage()
