"""
SAM3 Video Tracking - 簡略化疑似コード
====================================

動画内のオブジェクトを追跡するための時間的メモリメカニズム
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class Sam3VideoInference(nn.Module):
    """
    SAM3 動画推論モデル

    フレーム間でメモリを維持してオブジェクトを追跡
    画像モード (単一フレーム) と動画モード (時間的追跡) の両方をサポート
    """

    def __init__(
        self,
        image_model,                    # Sam3Image (画像セグメンテーションモデル)
        num_maskmem: int = 7,          # 使用するメモリフレーム数
        hidden_dim: int = 256,          # 特徴次元
        mem_dim: int = 64,              # メモリ特徴次元
        memory_temporal_stride: int = 1, # メモリサンプリングストライド
        is_image_only: bool = False     # 画像のみモード (メモリ不使用)
    ):
        super().__init__()

        self.image_model = image_model
        self.num_maskmem = num_maskmem
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.memory_temporal_stride = memory_temporal_stride
        self.is_image_only = is_image_only

        # ========================================
        # メモリエンコーダー (マスクを特徴に変換)
        # ========================================
        self.memory_encoder = SimpleMaskEncoder(
            in_dim=hidden_dim,
            out_dim=mem_dim,
            mask_downsample_stride=16
        )

        # ========================================
        # 位置エンコーディング
        # ========================================
        # 時間的位置エンコーディング (フレーム間の距離)
        self.temporal_pos_encoding = SinePositionEncoding1D(
            num_pos_feats=hidden_dim // 2
        )

        # 空間的位置エンコーディング (2D位置)
        self.spatial_pos_encoding = SinePositionEncoding2D(
            num_pos_feats=hidden_dim // 2
        )


    def init_state(
        self,
        video: torch.Tensor,
        num_frames: int
    ) -> Dict:
        """
        推論状態の初期化

        入力:
            video: (T, 3, H, W) - 動画フレーム
                T: フレーム数
                3: RGB
                H, W: 画像サイズ (1008x1008)

            num_frames: 処理するフレーム数

        出力:
            inference_state: Dict - 推論状態
                {
                    'video_input': (T, 3, H, W),
                    'num_frames': T,
                    'output_dict': {
                        'cond_frame_outputs': {},     # ユーザー指定フレーム
                        'non_cond_frame_outputs': {}  # 伝播フレーム
                    },
                    'current_frame_idx': 0
                }
        """

        inference_state = {
            'video_input': video,
            'num_frames': num_frames,
            'output_dict': {
                'cond_frame_outputs': {},      # frame_idx -> output
                'non_cond_frame_outputs': {}   # frame_idx -> output
            },
            'current_frame_idx': 0,
            'obj_id_to_idx': {},  # オブジェクトIDからインデックスへのマッピング
        }

        return inference_state


    def add_prompt(
        self,
        inference_state: Dict,
        frame_idx: int,
        obj_id: int,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        特定フレームにプロンプトを追加 (初期化用)

        入力:
            frame_idx: プロンプトを追加するフレーム番号
            obj_id: オブジェクトID
            points: (N, 2) - 点プロンプト
            boxes: (N, 4) - ボックスプロンプト
            masks: (1, H, W) - マスクプロンプト

        出力:
            inference_state: 更新された推論状態
        """

        # フレームを取得
        frame = inference_state['video_input'][frame_idx]  # (3, H, W)
        frame = frame.unsqueeze(0)  # (1, 3, H, W)

        # 画像モデルでセグメンテーション実行
        output = self.image_model(
            images=frame,
            point_prompts=points.unsqueeze(0) if points is not None else None,
            box_prompts=boxes.unsqueeze(0) if boxes is not None else None,
            mask_prompts=masks.unsqueeze(0) if masks is not None else None
        )

        # 予測マスクと特徴を取得
        pred_masks = output['pred_masks'][0]  # (200, H, W)
        vision_features = output['vision_features']

        # 最高スコアのマスクを選択
        scores = output['pred_scores'][0]  # (200,)
        best_idx = torch.argmax(scores)
        best_mask = pred_masks[best_idx:best_idx+1]  # (1, H, W)

        # メモリにエンコード
        memory_output = self._encode_memory(
            masks=best_mask.unsqueeze(0),      # (1, 1, H, W)
            vision_features=vision_features,
            is_init_cond_frame=True
        )

        # 出力を保存
        frame_output = {
            'pred_masks': best_mask,                          # (1, H, W)
            'maskmem_features': memory_output['maskmem_features'],  # (1, mem_dim, H_mem, W_mem)
            'maskmem_pos_enc': memory_output['maskmem_pos_enc'],    # (1, mem_dim, H_mem, W_mem)
            'obj_ptr': memory_output['obj_ptr'],              # (1, hidden_dim)
            'object_score': scores[best_idx],                 # スカラー
        }

        inference_state['output_dict']['cond_frame_outputs'][frame_idx] = frame_output
        inference_state['obj_id_to_idx'][obj_id] = 0  # 簡略化: 1オブジェクトのみ

        return inference_state


    def propagate_in_video(
        self,
        inference_state: Dict,
        start_frame_idx: int = 0,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        動画全体にマスクを伝播

        入力:
            inference_state: 推論状態
            start_frame_idx: 開始フレーム
            max_frame_num_to_track: 追跡する最大フレーム数
            reverse: True なら逆方向 (過去へ)

        出力:
            video_segments: Dict {frame_idx -> (num_obj, H, W) マスク}
        """

        num_frames = inference_state['num_frames']
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        # フレームの範囲を決定
        if reverse:
            frame_indices = range(start_frame_idx, -1, -1)
        else:
            frame_indices = range(start_frame_idx, min(start_frame_idx + max_frame_num_to_track, num_frames))

        video_segments = {}

        # 各フレームを順次処理
        for frame_idx in frame_indices:
            # 条件付きフレーム (ユーザー指定) をスキップ
            if frame_idx in inference_state['output_dict']['cond_frame_outputs']:
                output = inference_state['output_dict']['cond_frame_outputs'][frame_idx]
                video_segments[frame_idx] = output['pred_masks']
                continue

            # フレームを追跡
            output = self._track_step(
                inference_state,
                frame_idx,
                reverse=reverse
            )

            # 結果を保存
            inference_state['output_dict']['non_cond_frame_outputs'][frame_idx] = output
            video_segments[frame_idx] = output['pred_masks']

        return video_segments


    def _track_step(
        self,
        inference_state: Dict,
        frame_idx: int,
        reverse: bool = False
    ) -> Dict:
        """
        単一フレームの追跡ステップ

        処理フロー:
        1. 現在フレームのバックボーン特徴抽出
        2. メモリフレームの選択
        3. メモリとの融合 (Transformer Encoder)
        4. マスクデコード
        5. 新しいメモリのエンコード

        入力:
            inference_state: 推論状態
            frame_idx: 現在のフレーム番号
            reverse: 逆方向追跡フラグ

        出力:
            output: Dict {
                'pred_masks': (num_obj, H, W),
                'maskmem_features': (num_obj, mem_dim, H_mem, W_mem),
                'maskmem_pos_enc': (num_obj, mem_dim, H_mem, W_mem),
                'obj_ptr': (num_obj, hidden_dim),
                'object_score': (num_obj,)
            }
        """

        # ========================================
        # Step 1: 現在フレームのバックボーン特徴抽出
        # ========================================

        frame = inference_state['video_input'][frame_idx]  # (3, H, W)
        frame = frame.unsqueeze(0)  # (1, 3, H, W)

        # Visionバックボーンで特徴抽出 (image_modelのバックボーン使用)
        vision_features = self.image_model.vl_combiner.forward_vision(frame)
        # vision_features: List[(1, 256, H/4, W/4), ..., (1, 256, H/32, W/32)]

        # 最上位レベルを使用 (最も高解像度)
        current_vision_feat = vision_features[0]  # (1, 256, H/16, W/16)
        B, C, H_feat, W_feat = current_vision_feat.shape

        # 平坦化: (1, 256, H, W) -> (HW, 1, 256)
        current_vision_feat_flat = current_vision_feat.flatten(2).permute(2, 0, 1)
        # current_vision_feat_flat: (HW, 1, 256)


        # ========================================
        # Step 2: メモリフレームの選択
        # ========================================

        memory_frames = self._select_memory_frames(
            inference_state,
            current_frame_idx=frame_idx,
            num_maskmem=self.num_maskmem,
            temporal_stride=self.memory_temporal_stride,
            reverse=reverse
        )
        # memory_frames: List[frame_output] 長さ最大 num_maskmem


        # ========================================
        # Step 3: メモリとの融合
        # ========================================

        if len(memory_frames) > 0 and not self.is_image_only:
            # メモリ特徴とオブジェクトポインタを準備
            fused_features = self._prepare_memory_conditioned_features(
                current_vision_feat_flat,  # (HW, 1, 256)
                memory_frames,
                frame_idx
            )
            # fused_features: (HW, 1, 256) - メモリで条件付けされた特徴
        else:
            # 画像のみモード (メモリなし)
            fused_features = current_vision_feat_flat


        # ========================================
        # Step 4: マスクデコード (SAM Decoder)
        # ========================================

        # SAM Decoderでマスク生成 (プロンプトなし、メモリからの伝播)
        mask_output = self._forward_sam_heads(
            fused_features,         # (HW, 1, 256)
            vision_features,        # List[Tensor] - multi-scale
            point_prompts=None,     # 伝播時はプロンプトなし
            mask_prompts=None
        )

        pred_masks = mask_output['masks']          # (1, H, W)
        obj_ptr = mask_output['obj_ptr']           # (1, hidden_dim)
        object_score = mask_output['obj_score']    # (1,)


        # ========================================
        # Step 5: 新しいメモリのエンコード
        # ========================================

        memory_output = self._encode_memory(
            masks=pred_masks.unsqueeze(0),  # (1, 1, H, W)
            vision_features=vision_features,
            is_init_cond_frame=False
        )


        # ========================================
        # 出力の構築
        # ========================================

        output = {
            'pred_masks': pred_masks,                          # (1, H, W)
            'maskmem_features': memory_output['maskmem_features'],  # (1, mem_dim, H_mem, W_mem)
            'maskmem_pos_enc': memory_output['maskmem_pos_enc'],    # (1, mem_dim, H_mem, W_mem)
            'obj_ptr': obj_ptr,                                # (1, hidden_dim)
            'object_score': object_score,                      # (1,)
        }

        return output


    def _select_memory_frames(
        self,
        inference_state: Dict,
        current_frame_idx: int,
        num_maskmem: int,
        temporal_stride: int,
        reverse: bool
    ) -> List[Dict]:
        """
        現在フレームで使用するメモリフレームを選択

        選択戦略:
        - 位置0: 条件付きフレーム (ユーザー指定、最も重要)
        - 位置1: 直前/直後フレーム (時間的に最も近い)
        - 位置2〜num_maskmem-1: temporal_strideごとのフレーム

        入力:
            current_frame_idx: 現在のフレーム番号
            num_maskmem: 最大メモリフレーム数
            temporal_stride: メモリサンプリングストライド
            reverse: 逆方向追跡フラグ

        出力:
            memory_frames: List[frame_output] - 選択されたメモリフレーム
                各要素は以下を含む:
                {
                    'maskmem_features': (1, mem_dim, H_mem, W_mem),
                    'maskmem_pos_enc': (1, mem_dim, H_mem, W_mem),
                    'obj_ptr': (1, hidden_dim),
                    't_pos': int - 時間的距離 (フレーム数)
                }
        """

        cond_outputs = inference_state['output_dict']['cond_frame_outputs']
        non_cond_outputs = inference_state['output_dict']['non_cond_frame_outputs']

        memory_frames = []

        # 条件付きフレームを最優先で追加
        for cond_frame_idx in sorted(cond_outputs.keys()):
            if len(memory_frames) >= num_maskmem:
                break

            if reverse:
                if cond_frame_idx > current_frame_idx:
                    continue
            else:
                if cond_frame_idx >= current_frame_idx:
                    continue

            frame_output = cond_outputs[cond_frame_idx].copy()
            frame_output['t_pos'] = abs(current_frame_idx - cond_frame_idx)
            memory_frames.append(frame_output)

        # 非条件付きフレームを時間的ストライドで追加
        if reverse:
            candidate_indices = range(current_frame_idx - 1, -1, -temporal_stride)
        else:
            candidate_indices = range(current_frame_idx - 1, -1, -temporal_stride)

        for mem_frame_idx in candidate_indices:
            if len(memory_frames) >= num_maskmem:
                break

            if mem_frame_idx in non_cond_outputs:
                frame_output = non_cond_outputs[mem_frame_idx].copy()
                frame_output['t_pos'] = abs(current_frame_idx - mem_frame_idx)
                memory_frames.append(frame_output)

        return memory_frames


    def _prepare_memory_conditioned_features(
        self,
        current_vision_feat: torch.Tensor,
        memory_frames: List[Dict],
        current_frame_idx: int
    ) -> torch.Tensor:
        """
        メモリで条件付けされた特徴を準備

        処理:
        1. 空間メモリ (maskmem_features) を結合
        2. オブジェクトポインタ (obj_ptr) を結合
        3. 時間的位置エンコーディングを追加
        4. Transformer Encoderで融合

        入力:
            current_vision_feat: (HW, B, 256) - 現在フレームの特徴
            memory_frames: List[frame_output] - メモリフレーム
            current_frame_idx: 現在のフレーム番号

        出力:
            fused_features: (HW, B, 256) - メモリ融合後の特徴
        """

        HW, B, C = current_vision_feat.shape

        # ========================================
        # 空間メモリの準備
        # ========================================

        spatial_memory_list = []
        spatial_pos_list = []

        for mem_frame in memory_frames:
            maskmem_feat = mem_frame['maskmem_features']  # (1, mem_dim, H_mem, W_mem)
            maskmem_pos = mem_frame['maskmem_pos_enc']    # (1, mem_dim, H_mem, W_mem)

            # 平坦化: (1, mem_dim, H, W) -> (HW_mem, 1, mem_dim)
            maskmem_feat_flat = maskmem_feat.flatten(2).permute(2, 0, 1)
            maskmem_pos_flat = maskmem_pos.flatten(2).permute(2, 0, 1)

            spatial_memory_list.append(maskmem_feat_flat)
            spatial_pos_list.append(maskmem_pos_flat)

        if len(spatial_memory_list) > 0:
            # 全メモリフレームを連結
            spatial_memory = torch.cat(spatial_memory_list, dim=0)
            # spatial_memory: (HW_mem_total, 1, mem_dim)
            spatial_pos = torch.cat(spatial_pos_list, dim=0)
            # spatial_pos: (HW_mem_total, 1, mem_dim)
        else:
            spatial_memory = None
            spatial_pos = None


        # ========================================
        # オブジェクトポインタの準備
        # ========================================

        obj_ptr_list = []
        obj_ptr_pos_list = []

        for mem_frame in memory_frames:
            obj_ptr = mem_frame['obj_ptr']  # (1, hidden_dim)
            t_pos = mem_frame['t_pos']      # フレーム距離

            # 時間的位置エンコーディング
            t_pos_enc = self.temporal_pos_encoding(t_pos)  # (hidden_dim,)
            t_pos_enc = t_pos_enc.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)

            obj_ptr_list.append(obj_ptr.unsqueeze(0))  # (1, 1, hidden_dim)
            obj_ptr_pos_list.append(t_pos_enc)

        if len(obj_ptr_list) > 0:
            obj_ptrs = torch.cat(obj_ptr_list, dim=0)  # (num_mem, 1, hidden_dim)
            obj_ptrs_pos = torch.cat(obj_ptr_pos_list, dim=0)  # (num_mem, 1, hidden_dim)
        else:
            obj_ptrs = None
            obj_ptrs_pos = None


        # ========================================
        # Transformer Encoderで融合
        # ========================================

        # プロンプト特徴の結合
        if spatial_memory is not None and obj_ptrs is not None:
            # 空間メモリとオブジェクトポインタを結合
            prompt_features = torch.cat([spatial_memory, obj_ptrs], dim=0)
            # prompt_features: (HW_mem_total + num_mem, 1, C)

            prompt_pos = torch.cat([spatial_pos, obj_ptrs_pos], dim=0)
            # prompt_pos: (HW_mem_total + num_mem, 1, C)
        elif spatial_memory is not None:
            prompt_features = spatial_memory
            prompt_pos = spatial_pos
        elif obj_ptrs is not None:
            prompt_features = obj_ptrs
            prompt_pos = obj_ptrs_pos
        else:
            # メモリなし
            return current_vision_feat

        # Transformer Encoderで融合 (image_modelのencoderを使用)
        # 簡略化: ここでは直接的なクロスアテンションを想定
        fused_features = self.image_model.encoder(
            vision_features=[current_vision_feat.permute(1, 2, 0).reshape(B, C, -1, -1)],
            prompt_features=prompt_features
        )
        # fused_features: (HW, B, 256)

        return fused_features


    def _forward_sam_heads(
        self,
        image_features: torch.Tensor,
        vision_features: List[torch.Tensor],
        point_prompts: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        SAM Decoderでマスク予測

        入力:
            image_features: (HW, B, 256) - エンコードされた画像特徴
            vision_features: List[Tensor] - Multi-scale特徴
            point_prompts: (B, N, 2) - 点プロンプト (optional)
            mask_prompts: (B, 1, H, W) - マスクプロンプト (optional)

        出力:
            Dict {
                'masks': (B, H, W) - 予測マスク
                'obj_ptr': (B, hidden_dim) - オブジェクトポインタ
                'obj_score': (B,) - オブジェクトスコア
            }
        """

        # SAM Decoderを使用 (実装の詳細は省略)
        # 実際にはimage_model.decoderとsegmentation_headを使用

        # ダミー実装
        B = image_features.shape[1]
        H, W = 1008, 1008

        masks = torch.randn(B, H, W)
        obj_ptr = torch.randn(B, self.hidden_dim)
        obj_score = torch.rand(B)

        return {
            'masks': masks,
            'obj_ptr': obj_ptr,
            'obj_score': obj_score
        }


    def _encode_memory(
        self,
        masks: torch.Tensor,
        vision_features: List[torch.Tensor],
        is_init_cond_frame: bool
    ) -> Dict:
        """
        マスクをメモリ表現にエンコード

        入力:
            masks: (B, 1, H, W) - 予測マスク
            vision_features: List[Tensor] - Multi-scale視覚特徴
            is_init_cond_frame: 初期化フレームかどうか

        出力:
            Dict {
                'maskmem_features': (B, mem_dim, H_mem, W_mem),
                'maskmem_pos_enc': (B, mem_dim, H_mem, W_mem),
                'obj_ptr': (B, hidden_dim)
            }
        """

        # 最上位レベルの特徴を使用
        vision_feat = vision_features[0]  # (B, 256, H/16, W/16)

        # メモリエンコーダーでエンコード
        memory_output = self.memory_encoder(
            masks=masks,
            vision_features=vision_feat
        )

        return memory_output


# ============================================
# メモリエンコーダー
# ============================================

class SimpleMaskEncoder(nn.Module):
    """
    マスクをメモリ特徴に変換

    処理フロー:
    1. マスクを低解像度にダウンサンプル
    2. Vision特徴と加算融合
    3. ConvNeXtブロックで精製
    4. 位置エンコーディング追加
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 64,
        mask_downsample_stride: int = 16
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # マスクダウンサンプラー
        self.mask_downsampler = SimpleMaskDownSampler(
            total_stride=mask_downsample_stride,
            kernel_size=4,
            stride=4
        )

        # Vision特徴の投影
        self.pix_feat_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        # 融合レイヤー (ConvNeXt風)
        self.fuser = SimpleFuser(
            channels=out_dim,
            num_layers=2
        )

        # 出力投影
        self.out_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        # 位置エンコーディング
        self.pos_encoding = SinePositionEncoding2D(
            num_pos_feats=out_dim // 2
        )


    def forward(
        self,
        masks: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Dict:
        """
        マスクをメモリ表現にエンコード

        入力:
            masks: (B, 1, H, W) - マスクロジット
                B: バッチサイズ
                1: 1チャンネル (バイナリマスク)
                H, W: マスク解像度 (1008x1008)

            vision_features: (B, in_dim, H_feat, W_feat) - Vision特徴
                B: バッチサイズ
                in_dim: 特徴チャンネル (256)
                H_feat, W_feat: 特徴解像度 (例: 63x63)

        出力:
            Dict {
                'maskmem_features': (B, out_dim, H_mem, W_mem),
                'maskmem_pos_enc': (B, out_dim, H_mem, W_mem),
                'obj_ptr': (B, in_dim) - ダミー
            }
        """

        B = masks.shape[0]

        # ========================================
        # Step 1: マスクのダウンサンプル
        # ========================================

        # Sigmoid + スケーリング
        masks_sigmoid = torch.sigmoid(masks * 20.0 - 10.0)
        # masks_sigmoid: (B, 1, H, W) in [0, 1]

        # ダウンサンプル
        masks_down = self.mask_downsampler(masks_sigmoid)
        # masks_down: (B, out_dim, H_mem, W_mem)
        #   H_mem = H / 16, W_mem = W / 16  (例: 63x63)


        # ========================================
        # Step 2: Vision特徴の投影
        # ========================================

        vision_proj = self.pix_feat_proj(vision_features)
        # vision_proj: (B, out_dim, H_feat, W_feat)

        # サイズを合わせる (必要に応じてリサイズ)
        if vision_proj.shape[-2:] != masks_down.shape[-2:]:
            vision_proj = nn.functional.interpolate(
                vision_proj,
                size=masks_down.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        # vision_proj: (B, out_dim, H_mem, W_mem)


        # ========================================
        # Step 3: 加算融合
        # ========================================

        fused = vision_proj + masks_down
        # fused: (B, out_dim, H_mem, W_mem)


        # ========================================
        # Step 4: ConvNeXtで精製
        # ========================================

        fused_refined = self.fuser(fused)
        # fused_refined: (B, out_dim, H_mem, W_mem)


        # ========================================
        # Step 5: 出力投影と位置エンコーディング
        # ========================================

        maskmem_features = self.out_proj(fused_refined)
        # maskmem_features: (B, out_dim, H_mem, W_mem)

        maskmem_pos_enc = self.pos_encoding(maskmem_features)
        # maskmem_pos_enc: (B, out_dim, H_mem, W_mem)


        # オブジェクトポインタ (簡略化: Global Average Pooling)
        obj_ptr = vision_features.mean(dim=[2, 3])  # (B, in_dim)


        return {
            'maskmem_features': maskmem_features,
            'maskmem_pos_enc': maskmem_pos_enc,
            'obj_ptr': obj_ptr
        }


class SimpleMaskDownSampler(nn.Module):
    """マスクを低解像度にダウンサンプル"""

    def __init__(self, total_stride: int = 16, kernel_size: int = 4, stride: int = 4):
        super().__init__()

        num_layers = 0
        current_stride = 1
        layers = []

        while current_stride < total_stride:
            layers.append(nn.Conv2d(
                1 if len(layers) == 0 else 64,
                64,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ))
            layers.append(nn.ReLU())
            current_stride *= stride
            num_layers += 1

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimpleFuser(nn.Module):
    """ConvNeXt風の融合レイヤー"""

    def __init__(self, channels: int = 64, num_layers: int = 2):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels * 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(channels * 4, channels, kernel_size=1)
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Residual
        return x


# ============================================
# 位置エンコーディング
# ============================================

class SinePositionEncoding1D(nn.Module):
    """1D正弦波位置エンコーディング (時間用)"""

    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, t: int) -> torch.Tensor:
        """
        時間的距離から位置エンコーディングを生成

        入力:
            t: フレーム距離 (int)

        出力:
            pos_enc: (num_pos_feats*2,) - 位置エンコーディング
        """

        # 周波数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 正規化された時間
        t_normalized = float(t) / 100.0  # 最大距離を100と仮定

        # 正弦波エンコーディング
        pos = t_normalized / dim_t
        pos_enc = torch.cat([torch.sin(pos), torch.cos(pos)], dim=0)

        return pos_enc


class SinePositionEncoding2D(nn.Module):
    """2D正弦波位置エンコーディング (空間用)"""

    def __init__(self, num_pos_feats: int = 32):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D位置エンコーディングを生成

        入力:
            x: (B, C, H, W) - 任意のテンソル (shapeのみ使用)

        出力:
            pos_enc: (B, num_pos_feats*2, H, W) - 位置エンコーディング
        """

        B, C, H, W = x.shape

        # グリッド座標
        y_embed = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)

        # 正規化
        y_embed = y_embed / H
        x_embed = x_embed / W

        # 周波数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Y方向エンコーディング
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=3).flatten(2)

        # X方向エンコーディング
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=3).flatten(2)

        # 結合
        pos_enc = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)
        # pos_enc: (num_pos_feats*2, H, W)

        pos_enc = pos_enc.unsqueeze(0).repeat(B, 1, 1, 1)
        # pos_enc: (B, num_pos_feats*2, H, W)

        return pos_enc


# ============================================
# 使用例
# ============================================

def example_video_tracking():
    """動画追跡の使用例"""

    from main_flow import SAM3Image

    # 画像モデルの初期化
    image_model = SAM3Image(config={})

    # 動画モデルの初期化
    video_model = Sam3VideoInference(
        image_model=image_model,
        num_maskmem=7,
        memory_temporal_stride=1,
        is_image_only=False
    )

    # ダミー動画データ
    num_frames = 10
    video = torch.randn(num_frames, 3, 1008, 1008)

    # ========================================
    # 推論フロー
    # ========================================

    # 1. 状態の初期化
    inference_state = video_model.init_state(video, num_frames)

    # 2. 最初のフレームにプロンプト追加
    point = torch.tensor([[500, 500]])  # 画像中心付近
    inference_state = video_model.add_prompt(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=point
    )

    # 3. 全フレームに伝播 (順方向)
    video_segments = video_model.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=num_frames,
        reverse=False
    )

    # 4. 結果の取得
    for frame_idx, masks in video_segments.items():
        print(f"Frame {frame_idx}: masks shape = {masks.shape}")
        # Frame 0: masks shape = torch.Size([1, 1008, 1008])
        # Frame 1: masks shape = torch.Size([1, 1008, 1008])
        # ...

    print(f"\n全{len(video_segments)}フレームを追跡完了")


if __name__ == "__main__":
    example_video_tracking()
