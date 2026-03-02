"""
DINOv3 Main Flow - 簡略化疑似コード
====================================

DINOv3の全体フロー (Teacher-Student SSL Framework)
論文: https://arxiv.org/abs/2508.10104

このファイルではDINOv3の学習フレームワーク全体を簡略化して示します。
主要コンポーネント:
  1. Student Backbone (ViT-7B)
  2. Teacher Backbone (EMA of Student)
  3. Gram Teacher (frozen snapshot)
  4. DINO Head (画像レベル)
  5. iBOT Head (パッチレベル)
  6. 4種の損失関数

公式実装参照: dinov3/train/ssl_meta_arch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
# DINOv3 Head (DINO / iBOT 共通)
# ============================================================
class DINOHead(nn.Module):
    """
    DINO/iBOT共通のプロジェクションヘッド

    MLP + L2正規化 + プロトタイプ線形層

    DINOv3 ViT-7B 構成:
      DINO Head: D(4096) → 8192 → 8192 → 512 → 256K prototypes
      iBOT Head: D(4096) → 8192 → 8192 → 384 → 96K prototypes
    """

    def __init__(
        self,
        in_dim: int,          # バックボーン出力次元 (D=4096)
        hidden_dim: int,      # MLP隠れ層次元 (8192)
        bottleneck_dim: int,  # ボトルネック次元 (DINO: 512, iBOT: 384)
        n_prototypes: int,    # プロトタイプ数 (DINO: 256K, iBOT: 96K)
        n_layers: int = 3,    # MLP層数
    ):
        super().__init__()

        # MLP: in_dim → hidden_dim → ... → bottleneck_dim
        layers = []
        for i in range(n_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            dim_out = bottleneck_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # プロトタイプ (バイアスなし)
        self.last_layer = nn.Linear(bottleneck_dim, n_prototypes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) or (n_tokens, D) - バックボーン出力

        Returns:
            logits: (B, K) or (n_tokens, K) - プロトタイプロジット
        """
        x = self.mlp(x)                    # (B, bottleneck_dim)
        x = F.normalize(x, dim=-1)         # L2正規化
        logits = self.last_layer(x)         # (B, K)
        return logits


# ============================================================
# DINOv3 SSL Meta Architecture
# ============================================================
class DINOv3SSLFramework(nn.Module):
    """
    DINOv3の学習フレームワーク全体

    コンポーネント:
      - Student: backbone + dino_head + ibot_head (学習対象)
      - Teacher: backbone + dino_head + ibot_head (EMA, 勾配なし)
      - Gram Teacher: backbone のみ (frozen snapshot, オプション)

    学習フロー:
      1. Teacher: Global Crops のみ処理 → ターゲット生成
      2. Student: Global + Local Crops 処理 → 予測生成
      3. 損失計算: DINO + iBOT + KoLeo + Gram
      4. Student パラメータ更新
      5. Teacher EMA 更新
    """

    def __init__(
        self,
        # Backbone設定
        embed_dim: int = 4096,
        depth: int = 40,
        num_heads: int = 32,
        patch_size: int = 16,
        n_storage_tokens: int = 4,
        # Head設定
        dino_n_prototypes: int = 256_000,
        dino_bottleneck_dim: int = 512,
        ibot_n_prototypes: int = 96_000,
        ibot_bottleneck_dim: int = 384,
        head_hidden_dim: int = 8192,
        head_n_layers: int = 3,
        # Crop設定
        n_global_crops: int = 2,
        n_local_crops: int = 8,
        global_crop_size: int = 256,
        local_crop_size: int = 112,
        # EMA設定
        ema_momentum: float = 0.999,
        # 損失重み
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        gram_loss_weight: float = 2.0,
        use_gram: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.ema_momentum = ema_momentum
        self.use_gram = use_gram

        # 損失重み
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.gram_loss_weight = gram_loss_weight

        # === Student (学習対象) ===
        self.student_backbone = self._build_backbone(
            embed_dim, depth, num_heads, patch_size, n_storage_tokens
        )
        self.student_dino_head = DINOHead(
            embed_dim, head_hidden_dim, dino_bottleneck_dim,
            dino_n_prototypes, head_n_layers
        )
        self.student_ibot_head = DINOHead(
            embed_dim, head_hidden_dim, ibot_bottleneck_dim,
            ibot_n_prototypes, head_n_layers
        )

        # === Teacher (EMA, 勾配なし) ===
        self.teacher_backbone = self._build_backbone(
            embed_dim, depth, num_heads, patch_size, n_storage_tokens
        )
        self.teacher_dino_head = DINOHead(
            embed_dim, head_hidden_dim, dino_bottleneck_dim,
            dino_n_prototypes, head_n_layers
        )
        self.teacher_ibot_head = DINOHead(
            embed_dim, head_hidden_dim, ibot_bottleneck_dim,
            ibot_n_prototypes, head_n_layers
        )
        # Teacher の勾配を無効化
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False
        for p in self.teacher_ibot_head.parameters():
            p.requires_grad = False

        # === Gram Teacher (frozen snapshot, オプション) ===
        if use_gram:
            self.gram_teacher_backbone = self._build_backbone(
                embed_dim, depth, num_heads, patch_size, n_storage_tokens
            )
            for p in self.gram_teacher_backbone.parameters():
                p.requires_grad = False

    def _build_backbone(self, embed_dim, depth, num_heads, patch_size, n_storage_tokens):
        """バックボーン構築 (簡略化)"""
        # 実際は DinoVisionTransformer を使用
        # ここでは概念的な構造のみ示す
        return nn.Identity()  # 実際のViT実装で置き換え

    # ============================================================
    # Teacher 出力取得
    # ============================================================
    @torch.no_grad()
    def get_teacher_output(
        self,
        global_crops: torch.Tensor,
        masks: torch.Tensor,
        teacher_temp: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher は Global Crops のみ処理

        Args:
            global_crops: (n_global*B, 3, 256, 256) - 2枚のグローバルクロップ
            masks: (n_global*B, P) - マスク (P=256 for 256x256/16)
            teacher_temp: float - Teacher のソフトマックス温度

        Returns:
            dict with:
                cls_centered: (n_global, B, K_dino) - Sinkhorn-Knopp 正規化済み
                patch_centered: (n_masked, K_ibot) - マスクパッチのみ
                patch_pre_head: (n_global, B, P, D) - パッチ特徴 (Gram用)
        """
        n_global = self.n_global_crops
        B = global_crops.shape[0] // n_global  # バッチサイズ

        # --- Backbone Forward ---
        # 実際の実装では backbone に forward して各出力を取得
        # ここでは概念的にテンソル形状を示す
        backbone_out = self.teacher_backbone(global_crops)

        # CLS token: (n_global*B, D)
        cls_tokens = backbone_out["x_norm_clstoken"]       # (n_global*B, D=4096)
        # Storage tokens: (n_global*B, R, D)
        storage_tokens = backbone_out["x_storage_tokens"]   # (n_global*B, 4, 4096)
        # Patch tokens: (n_global*B, P, D)
        patch_tokens = backbone_out["x_norm_patchtokens"]   # (n_global*B, 256, 4096)

        # --- DINO Head (CLS token) ---
        cls_logits = self.teacher_dino_head(cls_tokens)     # (n_global*B, K_dino=256K)
        cls_logits = cls_logits.view(n_global, B, -1)       # (2, B, 256K)

        # Sinkhorn-Knopp 正規化
        cls_centered = self._sinkhorn_knopp(
            cls_logits.flatten(0, 1), teacher_temp           # (n_global*B, 256K) → (n_global*B, 256K)
        ).view(n_global, B, -1)                              # (2, B, 256K)

        # --- iBOT Head (masked patches only) ---
        # マスクされたパッチのみ抽出
        mask_indices = masks.bool()                          # (n_global*B, P)
        masked_patches = patch_tokens[mask_indices]          # (n_masked, D=4096)

        masked_logits = self.teacher_ibot_head(masked_patches)  # (n_masked, K_ibot=96K)
        patch_centered = self._sinkhorn_knopp(
            masked_logits, teacher_temp                       # (n_masked, 96K)
        )                                                     # (n_masked, 96K)

        return {
            "cls_centered": cls_centered,       # (n_global, B, K_dino)
            "patch_centered": patch_centered,   # (n_masked, K_ibot)
            "patch_pre_head": patch_tokens.view(n_global, B, -1, self.embed_dim),
            # (n_global, B, P, D) - Gram Anchoring 用
        }

    # ============================================================
    # Student 出力取得
    # ============================================================
    def get_student_output(
        self,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[Dict, Dict]:
        """
        Student は全 Crops を処理

        Args:
            global_crops: (n_global*B, 3, 256, 256) - グローバルクロップ
            local_crops: (n_local*B, 3, 112, 112) - ローカルクロップ
            masks: (n_global*B, P) - マスク (Global のみ)

        Returns:
            student_global: dict
                cls_after_head: (n_global, B, K_dino)
                masked_patch_after_head: (n_masked, K_ibot)
                patch_pre_head: (n_global, B, P, D)
            student_local: dict
                cls_after_head: (n_local, B, K_dino)
        """
        n_global = self.n_global_crops
        n_local = self.n_local_crops
        B = global_crops.shape[0] // n_global

        # --- Backbone Forward (Global + Local を一括処理) ---
        # 効率化のため list mode で処理
        # global_crops にはマスクを適用 (masked patches → mask_token)
        backbone_out = self.student_backbone(
            [global_crops, local_crops],
            masks=[masks, None]
        )
        # backbone_out[0]: global の結果, backbone_out[1]: local の結果

        # === Global Crops ===
        global_cls = backbone_out[0]["x_norm_clstoken"]         # (n_global*B, D)
        global_patches = backbone_out[0]["x_norm_patchtokens"]  # (n_global*B, P, D)

        # DINO Head on CLS
        global_cls_logits = self.student_dino_head(global_cls)  # (n_global*B, K_dino)
        global_cls_logits = global_cls_logits.view(n_global, B, -1)  # (2, B, K_dino)

        # iBOT Head on masked patches
        mask_indices = masks.bool()                              # (n_global*B, P)
        masked_patches = global_patches[mask_indices]            # (n_masked, D)
        masked_logits = self.student_ibot_head(masked_patches)   # (n_masked, K_ibot)

        student_global = {
            "cls_after_head": global_cls_logits,        # (n_global, B, K_dino)
            "masked_patch_after_head": masked_logits,   # (n_masked, K_ibot)
            "cls_pre_head": global_cls.view(n_global, B, -1),
            # (n_global, B, D) - KoLeo 用
            "patch_pre_head": global_patches.view(n_global, B, -1, self.embed_dim),
            # (n_global, B, P, D) - Gram 用
        }

        # === Local Crops ===
        local_cls = backbone_out[1]["x_norm_clstoken"]          # (n_local*B, D)
        local_cls_logits = self.student_dino_head(local_cls)     # (n_local*B, K_dino)
        local_cls_logits = local_cls_logits.view(n_local, B, -1)  # (8, B, K_dino)

        student_local = {
            "cls_after_head": local_cls_logits,  # (n_local, B, K_dino)
        }

        return student_global, student_local

    # ============================================================
    # 損失計算
    # ============================================================
    def compute_losses(
        self,
        teacher_out: Dict,
        student_global: Dict,
        student_local: Dict,
        masks: torch.Tensor,
        masks_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        全損失の計算

        Args:
            teacher_out: Teacher 出力
            student_global: Student Global 出力
            student_local: Student Local 出力
            masks: (n_global*B, P)
            masks_weight: (n_global*B, P) - パッチ重み

        Returns:
            total_loss: scalar
            loss_dict: 各損失の値
        """
        n_global = self.n_global_crops
        n_local = self.n_local_crops

        # --- スケーリング係数 ---
        # DINO loss の global/local 比率を調整
        dino_global_terms = n_global * (n_global - 1)  # 対角を無視: 2*(2-1)=2
        dino_local_terms = n_global * n_local           # 2*8=16
        total_dino_terms = dino_global_terms + dino_local_terms  # 18

        # --- 1. DINO Global Loss ---
        # Student global CLS vs Teacher global CLS (対角無視)
        loss_dino_global = self._dino_loss(
            student_logits=student_global["cls_after_head"],  # (n_global, B, K)
            teacher_probs=teacher_out["cls_centered"],         # (n_global, B, K)
            ignore_diagonal=True,
        )
        loss_dino_global *= self.dino_loss_weight * dino_global_terms / total_dino_terms

        # --- 2. DINO Local Loss ---
        # Student local CLS vs Teacher global CLS
        loss_dino_local = self._dino_loss(
            student_logits=student_local["cls_after_head"],   # (n_local, B, K)
            teacher_probs=teacher_out["cls_centered"],         # (n_global, B, K)
            ignore_diagonal=False,
        )
        loss_dino_local *= self.dino_loss_weight * dino_local_terms / total_dino_terms

        # --- 3. iBOT Patch Loss ---
        # Student masked patches vs Teacher masked patches
        loss_ibot = self._ibot_loss(
            student_masked=student_global["masked_patch_after_head"],  # (n_masked, K)
            teacher_masked=teacher_out["patch_centered"],               # (n_masked, K)
            masks_weight=masks_weight,
        )
        loss_ibot *= self.ibot_loss_weight

        # --- 4. KoLeo Loss ---
        # Student global CLS の均一性正則化
        loss_koleo = 0.0
        for crop_idx in range(n_global):
            cls_features = student_global["cls_pre_head"][crop_idx]  # (B, D)
            loss_koleo += self._koleo_loss(cls_features)
        loss_koleo *= self.koleo_loss_weight * n_global

        # --- 5. Gram Loss (オプション) ---
        loss_gram = torch.tensor(0.0)
        if self.use_gram:
            loss_gram = self._gram_loss(
                student_patches=student_global["patch_pre_head"],   # (n_global, B, P, D)
                gram_teacher_patches=teacher_out["patch_pre_head"], # (n_global, B, P, D)
            )
            loss_gram *= self.gram_loss_weight

        # --- 合計損失 ---
        total_loss = loss_dino_global + loss_dino_local + loss_ibot + loss_koleo + loss_gram

        loss_dict = {
            "dino_global": float(loss_dino_global),
            "dino_local": float(loss_dino_local),
            "ibot": float(loss_ibot),
            "koleo": float(loss_koleo),
            "gram": float(loss_gram),
            "total": float(total_loss),
        }

        return total_loss, loss_dict

    # ============================================================
    # 学習ステップ全体
    # ============================================================
    def forward_backward(
        self,
        data: Dict[str, torch.Tensor],
        teacher_temp: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        1回の学習ステップの全フロー

        Args:
            data: dict with:
                collated_global_crops: (n_global*B, 3, H_g, W_g)
                collated_local_crops: (n_local*B, 3, H_l, W_l)
                collated_masks: (n_global*B, P)
                masks_weight: (n_global*B, P)
            teacher_temp: float - Teacher ソフトマックス温度

        Returns:
            total_loss: scalar
            loss_dict: 各損失値の辞書
        """
        global_crops = data["collated_global_crops"]    # (2*B, 3, 256, 256)
        local_crops = data["collated_local_crops"]      # (8*B, 3, 112, 112)
        masks = data["collated_masks"]                   # (2*B, 256)
        masks_weight = data["masks_weight"]              # (2*B, 256)

        # Step 1: Teacher Forward (勾配なし)
        teacher_out = self.get_teacher_output(
            global_crops, masks, teacher_temp
        )

        # Step 2: Student Forward
        student_global, student_local = self.get_student_output(
            global_crops, local_crops, masks
        )

        # Step 3: 損失計算
        total_loss, loss_dict = self.compute_losses(
            teacher_out, student_global, student_local,
            masks, masks_weight
        )

        return total_loss, loss_dict

    # ============================================================
    # EMA Teacher 更新
    # ============================================================
    @torch.no_grad()
    def update_teacher(self, momentum: float = None):
        """
        Student → Teacher の EMA 更新

        teacher_param = m * teacher_param + (1 - m) * student_param

        Args:
            momentum: EMA momentum (デフォルト: self.ema_momentum=0.999)
        """
        m = momentum if momentum is not None else self.ema_momentum

        # Backbone
        for t_param, s_param in zip(
            self.teacher_backbone.parameters(),
            self.student_backbone.parameters(),
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        # DINO Head
        for t_param, s_param in zip(
            self.teacher_dino_head.parameters(),
            self.student_dino_head.parameters(),
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        # iBOT Head
        for t_param, s_param in zip(
            self.teacher_ibot_head.parameters(),
            self.student_ibot_head.parameters(),
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

    # ============================================================
    # ヘルパーメソッド (概念的実装)
    # ============================================================
    def _sinkhorn_knopp(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp 正規化

        Args:
            teacher_output: (B, K) - Teacher ロジット
            teacher_temp: float - 温度

        Returns:
            Q: (B, K) - 正規化された確率分布
        """
        B, K = teacher_output.shape
        Q = torch.exp(teacher_output / teacher_temp).T  # (K, B)
        Q /= Q.sum()                                    # 全体正規化

        for _ in range(n_iterations):
            Q /= Q.sum(dim=1, keepdim=True) * K          # 行正規化: 各protoの使用量均等
            Q /= Q.sum(dim=0, keepdim=True) * B          # 列正規化: 各サンプルの割当均等

        return Q.T  # (B, K)

    def _dino_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        ignore_diagonal: bool = False,
        student_temp: float = 0.1,
    ) -> torch.Tensor:
        """
        DINO CLS token 交差エントロピー損失

        Args:
            student_logits: (n_student_crops, B, K)
            teacher_probs: (n_teacher_crops, B, K)
            ignore_diagonal: 同一クロップペアを無視
            student_temp: Student ソフトマックス温度

        Returns:
            loss: scalar
        """
        n_s, B, K = student_logits.shape
        n_t = teacher_probs.shape[0]

        # Student softmax
        student_log_probs = F.log_softmax(
            student_logits.float() / student_temp, dim=-1
        )  # (n_s, B, K)

        # 交差エントロピー: -sum(teacher * log(student))
        # 全クロップペアの組み合わせ
        loss = -torch.einsum("sbk,tbk->st", student_log_probs, teacher_probs.float())
        # (n_s, n_t) - 各ペアの損失

        if ignore_diagonal:
            # 同一クロップのペア (s==t) を除外
            min_st = min(n_s, n_t)
            loss[:min_st, :min_st].fill_diagonal_(0)
            count = B * (n_s * n_t - min_st)
        else:
            count = B * n_s * n_t

        return loss.sum() / count

    def _ibot_loss(
        self,
        student_masked: torch.Tensor,
        teacher_masked: torch.Tensor,
        masks_weight: torch.Tensor,
        student_temp: float = 0.1,
    ) -> torch.Tensor:
        """
        iBOT パッチレベル損失

        Args:
            student_masked: (n_masked, K_ibot) - マスクパッチの Student 予測
            teacher_masked: (n_masked, K_ibot) - マスクパッチの Teacher ターゲット
            masks_weight: 各パッチの重み
            student_temp: 温度

        Returns:
            loss: scalar
        """
        student_log_probs = F.log_softmax(
            student_masked.float() / student_temp, dim=-1
        )  # (n_masked, K)

        # パッチごとの交差エントロピー
        loss = -torch.sum(teacher_masked.float() * student_log_probs, dim=-1)
        # (n_masked,)

        return loss.mean()

    def _koleo_loss(
        self,
        features: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        KoLeo (Kozachenko-Leonenko) 均一性正則化

        Args:
            features: (B, D) - CLS token 特徴

        Returns:
            loss: scalar - 最近傍距離の負対数平均
        """
        x = F.normalize(features, dim=-1, p=2)  # (B, D) L2正規化

        # 全ペアのドット積で最近傍を探索
        dots = torch.mm(x, x.T)      # (B, B)
        dots.fill_diagonal_(-1)        # 自分自身を除外
        nn_indices = dots.argmax(dim=1)  # (B,) 最近傍のインデックス

        # 最近傍とのL2距離
        nn_features = x[nn_indices]    # (B, D)
        distances = torch.norm(x - nn_features, p=2, dim=-1)  # (B,)

        # 負対数距離の平均 (小さい距離 → 大きい損失 → 分散を促進)
        loss = -torch.log(distances + eps).mean()

        return loss

    def _gram_loss(
        self,
        student_patches: torch.Tensor,
        gram_teacher_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gram Anchoring 損失

        Args:
            student_patches: (n_crops, B, P, D) - Student パッチ特徴
            gram_teacher_patches: (n_crops, B, P, D) - Gram Teacher パッチ特徴

        Returns:
            loss: scalar - Gram 行列の MSE
        """
        # 各画像ごとに Gram 行列を計算
        n_crops, B, P, D = student_patches.shape

        # Flatten: (n_crops*B, P, D)
        s_feats = student_patches.view(-1, P, D)
        g_feats = gram_teacher_patches.view(-1, P, D)

        # L2 正規化
        s_feats = F.normalize(s_feats, dim=-1)  # (n_crops*B, P, D)
        g_feats = F.normalize(g_feats, dim=-1)  # (n_crops*B, P, D)

        # Gram 行列: (n_crops*B, P, P)
        G_s = torch.bmm(s_feats, s_feats.transpose(1, 2))  # (n_crops*B, P, P)
        G_g = torch.bmm(g_feats, g_feats.transpose(1, 2))  # (n_crops*B, P, P)

        # 負値を除去 (オプション)
        G_s = torch.clamp(G_s, min=0)
        G_g = torch.clamp(G_g, min=0)

        # MSE (Frobenius ノルム)
        loss = F.mse_loss(G_s, G_g)

        return loss


# ============================================================
# 推論用ラッパー
# ============================================================
class DINOv3ForInference(nn.Module):
    """
    DINOv3の推論用ラッパー

    学習済み Student backbone を使って特徴抽出を行う
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 4096):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        画像から特徴を抽出

        Args:
            images: (B, 3, H, W) - 入力画像 (任意の解像度)

        Returns:
            dict with:
                cls_token: (B, D) - グローバル表現
                patch_tokens: (B, P, D) - 密特徴
                    P = (H/patch_size) * (W/patch_size)
        """
        out = self.backbone(images)
        return {
            "cls_token": out["x_norm_clstoken"],        # (B, D=4096)
            "patch_tokens": out["x_norm_patchtokens"],  # (B, P, D=4096)
        }

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        layer_indices: List[int] = [10, 20, 30, 40],
    ) -> List[torch.Tensor]:
        """
        中間層の特徴を取得 (検出/セグメンテーション用)

        Args:
            images: (B, 3, H, W)
            layer_indices: 取得する層のインデックス

        Returns:
            features: list of (B, P, D) - 各層の出力
                検出用: 4層の連結 → (B, P, 4*D=16384)
        """
        return self.backbone.get_intermediate_layers(
            images, n=layer_indices, reshape=False
        )
