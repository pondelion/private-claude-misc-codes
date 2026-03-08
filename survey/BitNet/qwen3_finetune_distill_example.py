"""
Qwen3-0.6B に対する BitDistill 適用サンプル

Qwen3-0.6B (FP16) を 1.58-bit BitNet に変換し、
小規模データで蒸留ファインチューニングする完全な例。

BitDistill 3段階パイプライン:
  Stage 1: SubLN挿入 + Linear→BitLinear変換
  Stage 2: 継続事前学習 (重み分布変換)
  Stage 3: 蒸留ファインチューニング (Teacher→Student)

必要パッケージ:
  pip install torch transformers datasets accelerate

参考: BitNet Distillation (arXiv:2510.13998)

注意: このサンプルは教育・デモ用です。
      本番環境では公式リポジトリの実装を使用してください。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import copy
import math


# ==============================================================================
# BitLinear: 1.58-bit線形層 (再掲・自己完結版)
# ==============================================================================

def ste_ternary_quantize(weight):
    """
    Straight-Through Estimator (STE) による三値量子化

    順伝播: w_q = RoundClip(w / Δ, -1, 1) * Δ  (量子化適用)
    逆伝播: ∂L/∂w ≈ ∂L/∂w_q                     (量子化を無視してそのまま通す)

    実装トリック:
      w_q = w + (quantize(w) - w).detach()
      → 値は quantize(w) だが、勾配は w に対してのみ流れる
      → .detach() で (quantize(w) - w) の勾配を切断しているため
         backwardでは w_q の勾配がそのまま w に伝わる
    """
    scale = weight.abs().mean()
    w_quant = scale * torch.clamp(torch.round(weight / (scale + 1e-8)), -1, 1)
    # STE: 値は量子化済み、勾配はweight方向にそのまま通す
    return weight + (w_quant - weight).detach()


class BitLinear(nn.Module):
    """
    1.58-bit 線形層

    学習時: FP16マスター重み → STE → 三値量子化 → 行列積
    推論時: 三値重み × INT8活性化 → 加減算のみ
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 重み量子化 (STE)
        if self.training:
            w = ste_ternary_quantize(self.weight)
        else:
            scale = self.weight.abs().mean()
            w = scale * torch.clamp(torch.round(self.weight / (scale + 1e-8)), -1, 1)

        # 活性化量子化 (INT8シミュレーション)
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_scale = gamma / 127.0
        x_q = torch.clamp(torch.round(x / x_scale), -128, 127) * x_scale

        return F.linear(x_q, w, self.bias)


# ==============================================================================
# モデル変換: Qwen3 → BitNet
# ==============================================================================

def convert_to_bitnet(model):
    """
    Qwen3モデルの全Linear層をBitLinearに変換

    変換対象:
      - q_proj, k_proj, v_proj, o_proj (Attention)
      - gate_proj, up_proj, down_proj (FFN)

    変換除外:
      - embed_tokens (Embedding層)
      - lm_head (出力層)

    SubLN挿入:
      - MHSA出力投影前にLayerNorm追加
      - FFN出力投影前にLayerNorm追加

    Args:
        model: HuggingFace Qwen3モデル

    Returns:
        model: BitLinear変換済みモデル
    """
    # Embedding と lm_head は除外するモジュール名のセット
    skip_modules = {'embed_tokens', 'lm_head', 'norm', 'input_layernorm',
                    'post_attention_layernorm', 'rotary_emb'}

    def _convert_module(module, name=""):
        for child_name, child in module.named_children():
            if child_name in skip_modules:
                continue

            if isinstance(child, nn.Linear):
                # BitLinearに置換
                bit_linear = BitLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                bit_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bit_linear.bias.data.copy_(child.bias.data)
                setattr(module, child_name, bit_linear)
            else:
                _convert_module(child, f"{name}.{child_name}")

    _convert_module(model)

    # SubLN挿入 (各Transformerレイヤーに)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        print("Warning: Could not find transformer layers for SubLN insertion")
        return model

    for layer in layers:
        hidden_dim = layer.input_layernorm.weight.shape[0]

        # MHSA SubLN
        if not hasattr(layer, 'sub_ln_attn'):
            layer.sub_ln_attn = nn.LayerNorm(hidden_dim)

        # FFN SubLN
        if hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'up_proj'):
                ffn_dim = layer.mlp.up_proj.out_features
            else:
                ffn_dim = hidden_dim * 4
        else:
            ffn_dim = hidden_dim * 4

        if not hasattr(layer, 'sub_ln_ffn'):
            layer.sub_ln_ffn = nn.LayerNorm(ffn_dim)

    print(f"BitNet変換完了: {sum(1 for n, _ in model.named_modules() if 'BitLinear' in type(_).__name__)} BitLinear layers")
    return model


# ==============================================================================
# 蒸留損失
# ==============================================================================

class DistillationLoss(nn.Module):
    """
    BitDistill 統合損失

    L = L_CE + λ × L_LD + γ × L_AD

    ハイパーパラメータ (分類タスク):
      τ = 5.0, λ = 10, γ = 1e5
    """
    def __init__(self, temperature=5.0, lambda_ld=10.0, gamma_ad=1e5):
        super().__init__()
        self.temperature = temperature
        self.lambda_ld = lambda_ld
        self.gamma_ad = gamma_ad

    def forward(self, s_logits, t_logits, labels, s_attns=None, t_attns=None):
        # Cross-Entropy
        loss_ce = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Logits蒸留
        tau = self.temperature
        t_probs = F.softmax(t_logits / tau, dim=-1)
        s_log_probs = F.log_softmax(s_logits / tau, dim=-1)
        loss_ld = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (tau ** 2)

        # Attention蒸留 (簡略版: attention weightsのKL div)
        loss_ad = torch.tensor(0.0, device=s_logits.device)
        if s_attns is not None and t_attns is not None:
            # 最終レイヤーのattentionのみ蒸留
            t_attn = t_attns[-1]  # (B, H, T, T)
            s_attn = s_attns[-1]
            loss_ad = F.kl_div(
                (s_attn + 1e-8).log(),
                t_attn,
                reduction='batchmean',
            )

        total = loss_ce + self.lambda_ld * loss_ld + self.gamma_ad * loss_ad
        return total, {'ce': loss_ce.item(), 'ld': loss_ld.item(), 'ad': loss_ad.item()}


# ==============================================================================
# サンプルデータセット
# ==============================================================================

class SimpleClassificationDataset(Dataset):
    """
    テキスト分類用サンプルデータセット

    実際のBitDistill論文ではGLUEベンチマーク (MNLI, QNLI, SST-2) を使用。
    ここでは動作確認用の簡易データセットを生成。
    """
    def __init__(self, tokenizer, num_samples=100, max_length=128, num_labels=3):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length

        # サンプルデータ生成
        templates = [
            "This is a positive example of text classification.",
            "This sentence is neutral and does not express any sentiment.",
            "This is clearly a negative example with bad implications.",
        ]

        self.texts = []
        self.labels = []
        for i in range(num_samples):
            label = i % num_labels
            self.texts.append(templates[label])
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==============================================================================
# メイン: Qwen3-0.6B BitDistill ファインチューニング
# ==============================================================================

def main():
    """
    BitDistill 完全パイプライン

    Stage 1: Qwen3-0.6B をBitNet化 (SubLN + BitLinear)
    Stage 2: 継続事前学習 (小規模デモ)
    Stage 3: 蒸留ファインチューニング (Teacher: FP16, Student: 1.58-bit)
    """

    print("=" * 70)
    print("BitDistill: Qwen3-0.6B → 1.58-bit ファインチューニングサンプル")
    print("=" * 70)

    # --- ハイパーパラメータ ---
    MODEL_NAME = "Qwen/Qwen3-0.6B"  # Qwen3-0.6B
    MAX_LENGTH = 128
    BATCH_SIZE = 4
    NUM_SAMPLES = 100       # デモ用小規模データ
    NUM_LABELS = 3          # 3クラス分類
    PRETRAIN_STEPS = 50     # 継続事前学習ステップ (デモ: 50, 論文: ~10B tokens分)
    FINETUNE_EPOCHS = 3     # ファインチューニングエポック
    TEMPERATURE = 5.0       # 蒸留温度
    LAMBDA_LD = 10.0        # Logits蒸留重み
    GAMMA_AD = 1e3          # Attention蒸留重み (デモ用に小さめ)
    LEARNING_RATE = 2e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # =============================================
    # モデル・トークナイザーのロード
    # =============================================
    print("\n--- モデルロード ---")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Teacher: FP16 分類モデル
        teacher = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        teacher.config.pad_token_id = tokenizer.pad_token_id
        teacher = teacher.to(device)

        print(f"Teacher loaded: {MODEL_NAME}")
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"Teacher parameters: {teacher_params / 1e6:.1f}M")

    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        print("transformers と accelerate がインストールされているか確認してください。")
        print("pip install transformers accelerate")
        print("\n--- ダミーモデルでデモを続行 ---")
        _run_dummy_demo()
        return

    # =============================================
    # データセット準備
    # =============================================
    print("\n--- データセット準備 ---")
    dataset = SimpleClassificationDataset(
        tokenizer, num_samples=NUM_SAMPLES, max_length=MAX_LENGTH, num_labels=NUM_LABELS
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # =============================================
    # Stage 1: Teacher ファインチューニング
    # =============================================
    print("\n" + "=" * 70)
    print("Stage 0: Teacher (FP16) のファインチューニング")
    print("=" * 70)

    teacher.train()
    optimizer_teacher = torch.optim.AdamW(teacher.parameters(), lr=LEARNING_RATE)

    for epoch in range(2):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer_teacher.zero_grad()
            loss.backward()
            optimizer_teacher.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    teacher.eval()
    print("Teacher ファインチューニング完了")

    # =============================================
    # Stage 1: BitNet変換 (SubLN + BitLinear)
    # =============================================
    print("\n" + "=" * 70)
    print("Stage 1: SubLN挿入 + BitLinear変換")
    print("=" * 70)

    # Teacherをコピーして Student を作成
    student = copy.deepcopy(teacher)
    student = student.float()  # FP32に変換 (QAT用)

    # BitLinear変換
    student = convert_to_bitnet(student)
    student = student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {student_params / 1e6:.1f}M")

    # 量子化前後の重みの統計
    for name, param in student.named_parameters():
        if 'BitLinear' in str(type(param)):
            break
        if 'weight' in name and param.dim() == 2:
            scale = param.abs().mean()
            w_ternary = torch.clamp(torch.round(param / (scale + 1e-8)), -1, 1)
            unique, counts = torch.unique(w_ternary, return_counts=True)
            print(f"  {name}: scale={scale:.4f}, 三値分布: ", end="")
            for u, c in zip(unique.tolist(), counts.tolist()):
                print(f"{int(u):+d}={c/w_ternary.numel()*100:.1f}% ", end="")
            print()
            break

    # =============================================
    # Stage 2: 継続事前学習 (簡略版)
    # =============================================
    print("\n" + "=" * 70)
    print("Stage 2: 継続事前学習 (重み分布変換)")
    print("=" * 70)
    print(f"  ステップ数: {PRETRAIN_STEPS} (デモ用)")
    print("  論文では10B tokens (FALCON corpus) を使用")

    student.train()
    optimizer_ct = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)

    # 継続事前学習 (分類タスクのデータで簡略化)
    step = 0
    for batch in dataloader:
        if step >= PRETRAIN_STEPS:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer_ct.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer_ct.step()

        step += 1
        if step % 10 == 0:
            print(f"  Step {step}/{PRETRAIN_STEPS}, Loss: {loss.item():.4f}")

    print("継続事前学習完了")

    # =============================================
    # Stage 3: 蒸留ファインチューニング
    # =============================================
    print("\n" + "=" * 70)
    print("Stage 3: 蒸留ベースファインチューニング")
    print("=" * 70)
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  λ (Logits): {LAMBDA_LD}, γ (Attention): {GAMMA_AD}")

    teacher.eval()
    student.train()

    optimizer_ft = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = DistillationLoss(
        temperature=TEMPERATURE,
        lambda_ld=LAMBDA_LD,
        gamma_ad=GAMMA_AD,
    )

    for epoch in range(FINETUNE_EPOCHS):
        total_losses = {'total': 0, 'ce': 0, 'ld': 0, 'ad': 0}
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Teacher forward (frozen)
            with torch.no_grad():
                t_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            # Student forward
            s_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            # 蒸留損失
            loss, loss_dict = criterion(
                s_outputs.logits, t_outputs.logits, labels,
                s_outputs.attentions, t_outputs.attentions,
            )

            optimizer_ft.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer_ft.step()

            total_losses['total'] += loss.item()
            for k in ['ce', 'ld', 'ad']:
                total_losses[k] += loss_dict[k]

            # 精度計算
            preds = s_outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        n_batches = len(dataloader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        accuracy = correct / total * 100

        print(
            f"  Epoch {epoch+1}/{FINETUNE_EPOCHS}: "
            f"Loss={avg_losses['total']:.4f} "
            f"(CE={avg_losses['ce']:.4f}, LD={avg_losses['ld']:.4f}, AD={avg_losses['ad']:.6f}) "
            f"Acc={accuracy:.1f}%"
        )

    print("\n蒸留ファインチューニング完了!")

    # =============================================
    # 評価
    # =============================================
    print("\n" + "=" * 70)
    print("評価")
    print("=" * 70)

    student.eval()
    teacher.eval()

    correct_teacher = 0
    correct_student = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            s_out = student(input_ids=input_ids, attention_mask=attention_mask)

            correct_teacher += (t_out.logits.argmax(-1) == labels).sum().item()
            correct_student += (s_out.logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

    teacher_acc = correct_teacher / total * 100
    student_acc = correct_student / total * 100

    print(f"  Teacher (FP16):      {teacher_acc:.1f}%")
    print(f"  Student (1.58-bit):  {student_acc:.1f}%")
    print(f"  差分:                {student_acc - teacher_acc:+.1f}pt")

    # メモリ比較
    teacher_mem = sum(p.nelement() * p.element_size() for p in teacher.parameters()) / 1024 / 1024
    student_mem = sum(p.nelement() * 2 / 8 for p in student.parameters()) / 1024 / 1024  # 1.58-bit想定
    print(f"\n  Teacher メモリ: {teacher_mem:.1f} MB")
    print(f"  Student メモリ (1.58-bit推定): {student_mem:.1f} MB")
    print(f"  圧縮率: {teacher_mem / student_mem:.1f}x")

    print("\n" + "=" * 70)
    print("完了! このサンプルではデモ用の小規模データを使用しています。")
    print("本番環境では以下を推奨:")
    print("  - Stage 2: FALCON corpus (10B tokens)")
    print("  - Stage 3: GLUE/CNN-DailyMail等の標準ベンチマーク")
    print("  - ハードウェア: 8x GPU (AMD Mi300X)")
    print("=" * 70)


# ==============================================================================
# ダミーデモ (transformersが利用不可の場合)
# ==============================================================================

def _run_dummy_demo():
    """transformersが利用不可の場合のダミーデモ"""
    print("\n--- ダミーモデルでBitLinearの動作確認 ---")

    # 簡単なモデル
    class DummyModel(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=128, num_labels=3):
            super().__init__()
            self.layer1 = BitLinear(input_dim, hidden_dim)
            self.act = nn.ReLU()
            self.layer2 = BitLinear(hidden_dim, num_labels)

        def forward(self, x):
            return self.layer2(self.act(self.layer1(x)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel().to(device)

    # ダミーデータ
    x = torch.randn(8, 64, device=device)
    labels = torch.randint(0, 3, (8,), device=device)

    # 学習ステップ
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(20):
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item() * 100
            print(f"  Step {step+1}, Loss: {loss.item():.4f}, Acc: {acc:.1f}%")

    # 重みの三値分布を確認
    print("\n--- 学習後の重みの三値分布 ---")
    for name, param in model.named_parameters():
        if 'weight' in name:
            scale = param.abs().mean()
            w_ternary = torch.clamp(torch.round(param / (scale + 1e-8)), -1, 1)
            unique, counts = torch.unique(w_ternary, return_counts=True)
            print(f"  {name}: ", end="")
            for u, c in zip(unique.tolist(), counts.tolist()):
                print(f"{int(u):+d}={c/w_ternary.numel()*100:.1f}% ", end="")
            print()


if __name__ == "__main__":
    main()
