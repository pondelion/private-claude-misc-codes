"""画像token、prompt、responseを処理する小型理解用decoder。

実モデルはMoonViT-V2 + projector + 36層Qwen3-4B。ここでは既にproject済みのvisual embeddingsを
受け、同一Transformer weightをcausal/blocked-bidirectional両modeで使う核心だけを実装する。
"""
from __future__ import annotations
import torch
from torch import nn


class TinyGroundAnythingDecoder(nn.Module):
    """token ids ``(B,T)`` またはembedding ``(B,T,D)`` -> logits ``(B,T,V)``。"""
    def __init__(self, vocab_size: int, d_model=128, n_heads=4, n_layers=3, max_length=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers)
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids=None, inputs_embeds=None, allowed_attention=None):
        """allowed_attention ``(T,T)``はTrue=可視。未指定なら通常causal maskを使う。"""
        x = self.token_embedding(token_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = x.shape
        x = x + self.position(torch.arange(T, device=x.device))[None]
        if allowed_attention is None:
            allowed_attention = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        # PyTorch Transformerのbool maskはTrue=遮断なので反転する。
        hidden = self.layers(x, mask=~allowed_attention)
        return self.head(self.norm(hidden))

    def embed_multimodal(self, token_ids, visual_embeddings, image_positions):
        """text embeddingのimage placeholder位置をvisual embeddingで置換する。

        token_ids ``(B,T)``, visual_embeddings ``(B,Nv,D)``, image_positions ``(B,Nv)``。
        実モデルではMoonViT patchを2x2 aggregateし、最大1024 visual tokenをここへ挿入する。
        """
        x = self.token_embedding(token_ids)
        for b in range(x.shape[0]):
            x[b, image_positions[b]] = visual_embeddings[b]
        return x
