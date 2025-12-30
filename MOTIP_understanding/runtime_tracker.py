"""
Runtime Tracker - 推論時のトラッキング管理

【役割】
1. フレームごとの検出結果を受け取る
2. ID Decoderで ID予測
3. ID割り当て (Assignment Protocol)
4. 軌跡情報の更新 (特徴、ボックス、ID、タイムスタンプ)
5. 非アクティブな軌跡の削除

【ID割り当てプロトコル】
MOTIPは5種類のAssignment Protocolをサポート:

1. **hungarian**: ハンガリアンアルゴリズム (最適割り当て)
   - コスト行列: 1 - ID確率
   - グローバル最適解

2. **id-max**: 貪欲法 (ID確率優先)
   - 各IDに対して最高確率の検出を割り当て
   - ID側から見た貪欲法

3. **object-max**: 貪欲法 (検出信頼度優先)
   - 各検出に対して最高確率のIDを割り当て
   - 検出側から見た貪欲法

4. **object-priority**: 優先度ベース (検出信頼度順)
   - 検出信頼度が高い順に処理
   - 各検出を最適なIDに割り当て

5. **id-priority**: 優先度ベース (ID確率順)
   - ID確率が高い順に処理
   - 各IDを最適な検出に割り当て

デフォルト: **object-max** (シンプルで高速)

【ID辞書管理】
- K個のID埋め込みをプール
- 使用中のIDと利用可能なIDをキューで管理
- 軌跡が終了したらIDを再利用

【軌跡管理】
- 各軌跡の情報:
  - features: 過去T フレームの特徴
  - boxes: バウンディングボックス
  - id_labels: 割り当てられたIDラベル
  - timestamps: タイムスタンプ
  - scores: 検出信頼度
  - miss_count: 検出されなかった連続フレーム数

- Miss Tolerance: miss_tolerance フレーム検出されないと削除
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy.optimize import linear_sum_assignment


class RuntimeTracker:
    """
    Runtime Tracker: 推論時のトラッキング管理

    【使用例】
    ```python
    tracker = RuntimeTracker(
        model=motip_model,
        num_id_vocabulary=50,
        max_trajectory_length=30,
        assignment_protocol='object-max',
        det_thresh=0.3,
        id_thresh=0.2,
        newborn_thresh=0.6,
        miss_tolerance=30,
    )

    for frame_idx, frame in enumerate(video):
        # フレームごとに更新
        tracks = tracker.update(frame, frame_idx)

        # 出力: List[Dict]
        # [
        #     {'id': 1, 'box': [x, y, w, h], 'score': 0.9},
        #     {'id': 2, 'box': [x, y, w, h], 'score': 0.85},
        #     ...
        # ]
    ```
    """

    def __init__(
        self,
        model: nn.Module,                      # MOTIPモデル
        num_id_vocabulary: int = 50,           # ID辞書サイズ
        max_trajectory_length: int = 30,       # 軌跡の最大長
        assignment_protocol: str = 'object-max',  # ID割り当てプロトコル
        det_thresh: float = 0.3,               # 検出信頼度閾値
        id_thresh: float = 0.2,                # ID確率閾値
        newborn_thresh: float = 0.6,           # 新規物体の検出信頼度閾値
        miss_tolerance: int = 30,              # 検出されないフレーム数の許容値
    ):
        self.model = model
        self.num_id_vocabulary = num_id_vocabulary
        self.max_trajectory_length = max_trajectory_length
        self.assignment_protocol = assignment_protocol
        self.det_thresh = det_thresh
        self.id_thresh = id_thresh
        self.newborn_thresh = newborn_thresh
        self.miss_tolerance = miss_tolerance

        # ========================================
        # ID辞書管理
        # ========================================
        # 利用可能なIDラベルのキュー (1 ~ K)
        self.available_ids = deque(range(1, num_id_vocabulary + 1))

        # 使用中のIDラベル → 実際のトラックID のマッピング
        self.id_label_to_track_id = {}

        # 次のトラックID
        self.next_track_id = 1

        # ========================================
        # 軌跡情報
        # ========================================
        self.trajectories = {}  # track_id -> TrajectoryInfo

    def update(
        self,
        frame: torch.Tensor,      # (3, H, W) or (1, 3, H, W)
        frame_idx: int,
    ) -> List[Dict]:
        """
        フレーム更新: 検出 → ID予測 → 割り当て → 軌跡更新

        Args:
            frame: 入力画像
            frame_idx: フレームインデックス

        Returns:
            tracks: トラッキング結果
                [{'id': track_id, 'box': [x, y, w, h], 'score': conf}, ...]
        """
        device = frame.device

        # ========================================
        # ステップ1: DETR検出
        # ========================================
        if frame.dim() == 3:
            frame = frame[None, None]  # (1, 1, 3, H, W)
        elif frame.dim() == 4:
            frame = frame[:, None]  # (B, 1, 3, H, W)

        with torch.no_grad():
            outputs = self.model(frame, mode='trajectory_modeling')

        # 検出結果を抽出
        pred_logits = outputs['pred_logits'][0, 0]  # (N, num_classes)
        pred_boxes = outputs['pred_boxes'][0, 0]    # (N, 4)
        trajectory_features = outputs['trajectory_features'][0, 0]  # (N, C)

        # ========================================
        # ステップ2: 信頼度によるフィルタリング
        # ========================================
        pred_scores = pred_logits.softmax(dim=-1)[:, 0]  # (N,) - クラス0の確率
        active_mask = pred_scores > self.det_thresh

        active_boxes = pred_boxes[active_mask]          # (M, 4)
        active_scores = pred_scores[active_mask]        # (M,)
        active_features = trajectory_features[active_mask]  # (M, C)
        M = active_boxes.shape[0]

        if M == 0:
            # 検出なし: すべての軌跡のmiss_countを増やす
            self._increment_miss_counts()
            self._filter_inactive_tracks()
            return []

        # ========================================
        # ステップ3: ID Decoder (履歴軌跡を使用)
        # ========================================
        # 履歴軌跡トークンを構築
        trajectory_tokens, trajectory_times = self._build_trajectory_tokens()
        # trajectory_tokens: (G, L, C+id_dim)
        # trajectory_times: (G, L)

        if trajectory_tokens is not None:
            # ID予測
            # current_tokens: τ^n = concat(f^n, i^spec)
            spec_token_idx = self.num_id_vocabulary
            spec_embeds = self.model.id_embeddings(
                torch.full((1, 1, M), spec_token_idx, dtype=torch.long, device=device)
            )  # (1, 1, M, id_dim)

            current_tokens = torch.cat([
                active_features[None, None],  # (1, 1, M, C)
                spec_embeds,                  # (1, 1, M, id_dim)
            ], dim=-1)  # (1, 1, M, C+id_dim)

            # ID Decoder
            id_decoder_outputs = self.model.id_decoder(
                current_tokens=current_tokens,
                trajectory_tokens=trajectory_tokens,
                trajectory_times=trajectory_times,
            )

            # ID予測ロジット
            id_logits = self.model.id_pred_head(
                id_decoder_outputs['output_features'][0, 0]
            )  # (M, K+1)
            id_probs = torch.softmax(id_logits, dim=-1)  # (M, K+1)
        else:
            # 軌跡がない場合: すべて新規物体
            id_probs = torch.zeros(M, self.num_id_vocabulary + 1, device=device)
            id_probs[:, -1] = 1.0  # 最後のクラス (新規物体) に確率1

        # ========================================
        # ステップ4: ID割り当て
        # ========================================
        id_labels = self._assign_id_labels(
            id_probs=id_probs,          # (M, K+1)
            active_scores=active_scores,  # (M,)
        )  # (M,) - 割り当てられたIDラベル (1~K or K+1)

        # ========================================
        # ステップ5: 軌跡更新
        # ========================================
        tracks = self._update_trajectories(
            id_labels=id_labels,
            active_boxes=active_boxes,
            active_scores=active_scores,
            active_features=active_features,
            frame_idx=frame_idx,
        )

        # ========================================
        # ステップ6: 非アクティブな軌跡を削除
        # ========================================
        self._filter_inactive_tracks()

        return tracks

    def _build_trajectory_tokens(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        履歴軌跡トークンを構築

        Returns:
            trajectory_tokens: (G, L, C+id_dim) or None
            trajectory_times: (G, L) or None
        """
        if len(self.trajectories) == 0:
            return None, None

        # 各軌跡のトークンを構築
        tokens_list = []
        times_list = []

        for track_id, traj_info in self.trajectories.items():
            # 過去の特徴とIDラベルを取得
            features = traj_info['features']  # List[(C,)]
            id_label = traj_info['id_label']  # int
            timestamps = traj_info['timestamps']  # List[int]

            if len(features) == 0:
                continue

            # ID埋め込みを取得
            id_embeds = self.model.id_embeddings(
                torch.tensor([id_label] * len(features), dtype=torch.long)
            )  # (L, id_dim)

            # トークン: τ^{m,km} = concat(f^m, i^km)
            features_tensor = torch.stack(features)  # (L, C)
            tokens = torch.cat([features_tensor, id_embeds], dim=-1)  # (L, C+id_dim)

            tokens_list.append(tokens)
            times_list.append(torch.tensor(timestamps, dtype=torch.long))

        if len(tokens_list) == 0:
            return None, None

        # パディング (最大長に合わせる)
        max_len = max(t.shape[0] for t in tokens_list)
        max_len = min(max_len, self.max_trajectory_length)

        padded_tokens = []
        padded_times = []

        for tokens, times in zip(tokens_list, times_list):
            L = tokens.shape[0]
            if L > max_len:
                # 最新max_lenフレームのみ使用
                tokens = tokens[-max_len:]
                times = times[-max_len:]
                L = max_len

            # パディング
            pad_len = max_len - L
            if pad_len > 0:
                tokens = torch.cat([
                    torch.zeros(pad_len, tokens.shape[1]),
                    tokens,
                ], dim=0)
                times = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    times,
                ], dim=0)

            padded_tokens.append(tokens)
            padded_times.append(times)

        trajectory_tokens = torch.stack(padded_tokens)  # (G, L, C+id_dim)
        trajectory_times = torch.stack(padded_times)    # (G, L)

        return trajectory_tokens, trajectory_times

    def _assign_id_labels(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        active_scores: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        """
        ID割り当て (Assignment Protocol)

        Returns:
            id_labels: (M,) - 割り当てられたIDラベル (1~K or K+1)
        """
        M, K_plus_1 = id_probs.shape
        K = K_plus_1 - 1

        # 使用中のIDラベル
        tracked_id_labels = set(
            traj_info['id_label']
            for traj_info in self.trajectories.values()
        )

        if self.assignment_protocol == 'hungarian':
            return self._assign_hungarian(id_probs, tracked_id_labels)
        elif self.assignment_protocol == 'id-max':
            return self._assign_id_max(id_probs, tracked_id_labels)
        elif self.assignment_protocol == 'object-max':
            return self._assign_object_max(id_probs, tracked_id_labels)
        elif self.assignment_protocol == 'object-priority':
            return self._assign_object_priority(id_probs, active_scores, tracked_id_labels)
        elif self.assignment_protocol == 'id-priority':
            return self._assign_id_priority(id_probs, active_scores, tracked_id_labels)
        else:
            raise ValueError(f"Unknown assignment protocol: {self.assignment_protocol}")

    def _assign_object_max(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        tracked_id_labels: set,
    ) -> torch.Tensor:
        """
        object-max: 各検出に対して最高確率のIDを割り当て

        デフォルトのプロトコル (シンプルで高速)
        """
        M, K_plus_1 = id_probs.shape

        # 各検出の最大ID確率とラベル
        max_probs, max_labels = torch.max(id_probs[:, :-1], dim=1)  # (M,), (M,)
        max_labels = max_labels + 1  # 0-indexed -> 1-indexed

        # 閾値以下は新規物体
        id_labels = torch.where(
            max_probs > self.id_thresh,
            max_labels,
            torch.tensor(K_plus_1 - 1, dtype=torch.long),  # 新規物体
        )

        # 使用中のIDでないものは新規物体
        for i in range(M):
            if id_labels[i].item() not in tracked_id_labels:
                id_labels[i] = K_plus_1 - 1

        # 重複排除: 同じIDが複数の検出に割り当てられた場合、
        # 最高確率の検出のみ保持、他は新規物体
        unique_ids = {}
        for i in range(M):
            id_label = id_labels[i].item()
            if id_label == K_plus_1 - 1:
                continue  # 新規物体はスキップ

            if id_label not in unique_ids:
                unique_ids[id_label] = (i, max_probs[i].item())
            else:
                # 既存の割り当てと比較
                prev_idx, prev_prob = unique_ids[id_label]
                if max_probs[i].item() > prev_prob:
                    # 現在の検出の方が確率が高い
                    id_labels[prev_idx] = K_plus_1 - 1  # 前の検出を新規物体に
                    unique_ids[id_label] = (i, max_probs[i].item())
                else:
                    # 前の検出の方が確率が高い
                    id_labels[i] = K_plus_1 - 1

        return id_labels

    def _assign_hungarian(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        tracked_id_labels: set,
    ) -> torch.Tensor:
        """
        hungarian: ハンガリアンアルゴリズムで最適割り当て
        """
        M, K_plus_1 = id_probs.shape

        # コスト行列: 1 - ID確率
        cost_matrix = 1 - id_probs[:, :-1].cpu().numpy()  # (M, K)

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # IDラベル (1-indexed)
        id_labels = torch.full((M,), K_plus_1 - 1, dtype=torch.long)
        for r, c in zip(row_ind, col_ind):
            id_label = c + 1
            if id_label in tracked_id_labels and id_probs[r, c].item() > self.id_thresh:
                id_labels[r] = id_label

        return id_labels

    def _assign_id_max(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        tracked_id_labels: set,
    ) -> torch.Tensor:
        """
        id-max: 各IDに対して最高確率の検出を割り当て
        """
        M, K_plus_1 = id_probs.shape

        id_labels = torch.full((M,), K_plus_1 - 1, dtype=torch.long)

        # 各IDに対して最高確率の検出を見つける
        for id_label in tracked_id_labels:
            id_idx = id_label - 1  # 0-indexed
            max_prob_idx = torch.argmax(id_probs[:, id_idx])
            if id_probs[max_prob_idx, id_idx].item() > self.id_thresh:
                id_labels[max_prob_idx] = id_label

        return id_labels

    def _assign_object_priority(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        active_scores: torch.Tensor,  # (M,)
        tracked_id_labels: set,
    ) -> torch.Tensor:
        """
        object-priority: 検出信頼度が高い順に処理
        """
        M, K_plus_1 = id_probs.shape

        # 検出信頼度でソート
        sorted_indices = torch.argsort(active_scores, descending=True)

        id_labels = torch.full((M,), K_plus_1 - 1, dtype=torch.long)
        assigned_ids = set()

        for idx in sorted_indices:
            # 最高確率のID
            max_prob, max_label = torch.max(id_probs[idx, :-1], dim=0)
            max_label = max_label.item() + 1  # 1-indexed

            if (max_label in tracked_id_labels and
                max_label not in assigned_ids and
                max_prob.item() > self.id_thresh):
                id_labels[idx] = max_label
                assigned_ids.add(max_label)

        return id_labels

    def _assign_id_priority(
        self,
        id_probs: torch.Tensor,       # (M, K+1)
        active_scores: torch.Tensor,  # (M,)
        tracked_id_labels: set,
    ) -> torch.Tensor:
        """
        id-priority: ID確率が高い順に処理
        """
        M, K_plus_1 = id_probs.shape

        id_labels = torch.full((M,), K_plus_1 - 1, dtype=torch.long)
        assigned_detections = set()

        # すべてのID確率をフラット化してソート
        flat_probs = id_probs[:, :-1].flatten()
        sorted_flat_indices = torch.argsort(flat_probs, descending=True)

        for flat_idx in sorted_flat_indices:
            det_idx = flat_idx // (K_plus_1 - 1)
            id_idx = flat_idx % (K_plus_1 - 1)
            id_label = id_idx + 1

            prob = flat_probs[flat_idx].item()

            if (prob > self.id_thresh and
                id_label in tracked_id_labels and
                det_idx.item() not in assigned_detections):
                id_labels[det_idx] = id_label
                assigned_detections.add(det_idx.item())

        return id_labels

    def _update_trajectories(
        self,
        id_labels: torch.Tensor,        # (M,)
        active_boxes: torch.Tensor,     # (M, 4)
        active_scores: torch.Tensor,    # (M,)
        active_features: torch.Tensor,  # (M, C)
        frame_idx: int,
    ) -> List[Dict]:
        """
        軌跡更新

        Returns:
            tracks: [{'id': track_id, 'box': [x, y, w, h], 'score': conf}, ...]
        """
        M = id_labels.shape[0]
        K = self.num_id_vocabulary

        tracks = []

        # すべての軌跡のmiss_countを増やす (後で更新された軌跡は0にリセット)
        for traj_info in self.trajectories.values():
            traj_info['miss_count'] += 1

        for i in range(M):
            id_label = id_labels[i].item()
            box = active_boxes[i].cpu().numpy()
            score = active_scores[i].item()
            feature = active_features[i]

            if id_label == K:  # 新規物体
                # 新規物体として追加するか判定
                if score > self.newborn_thresh:
                    track_id = self._create_new_track(id_label, box, score, feature, frame_idx)
                    tracks.append({'id': track_id, 'box': box, 'score': score})
            else:
                # 既存の軌跡に追加
                if id_label in self.id_label_to_track_id:
                    track_id = self.id_label_to_track_id[id_label]
                    self._append_to_track(track_id, box, score, feature, frame_idx)
                    tracks.append({'id': track_id, 'box': box, 'score': score})
                else:
                    # IDラベルが割り当てられていない (新規)
                    if score > self.newborn_thresh:
                        track_id = self._create_new_track(id_label, box, score, feature, frame_idx)
                        tracks.append({'id': track_id, 'box': box, 'score': score})

        return tracks

    def _create_new_track(
        self,
        id_label: int,
        box: torch.Tensor,
        score: float,
        feature: torch.Tensor,
        frame_idx: int,
    ) -> int:
        """
        新規軌跡を作成

        Returns:
            track_id: 新しいトラックID
        """
        # 利用可能なIDラベルを取得
        if len(self.available_ids) > 0:
            new_id_label = self.available_ids.popleft()
        else:
            # IDプールが枯渇: 最も古い軌跡のIDを再利用
            oldest_track_id = min(
                self.trajectories.keys(),
                key=lambda tid: self.trajectories[tid]['timestamps'][-1]
            )
            old_id_label = self.trajectories[oldest_track_id]['id_label']
            del self.trajectories[oldest_track_id]
            del self.id_label_to_track_id[old_id_label]
            new_id_label = old_id_label

        # 新しいトラックIDを割り当て
        track_id = self.next_track_id
        self.next_track_id += 1

        # 軌跡情報を初期化
        self.trajectories[track_id] = {
            'id_label': new_id_label,
            'features': [feature],
            'boxes': [box],
            'scores': [score],
            'timestamps': [frame_idx],
            'miss_count': 0,
        }

        self.id_label_to_track_id[new_id_label] = track_id

        return track_id

    def _append_to_track(
        self,
        track_id: int,
        box: torch.Tensor,
        score: float,
        feature: torch.Tensor,
        frame_idx: int,
    ):
        """
        既存の軌跡に追加
        """
        traj_info = self.trajectories[track_id]

        traj_info['features'].append(feature)
        traj_info['boxes'].append(box)
        traj_info['scores'].append(score)
        traj_info['timestamps'].append(frame_idx)
        traj_info['miss_count'] = 0  # リセット

        # 軌跡長を制限
        if len(traj_info['features']) > self.max_trajectory_length:
            traj_info['features'] = traj_info['features'][-self.max_trajectory_length:]
            traj_info['boxes'] = traj_info['boxes'][-self.max_trajectory_length:]
            traj_info['scores'] = traj_info['scores'][-self.max_trajectory_length:]
            traj_info['timestamps'] = traj_info['timestamps'][-self.max_trajectory_length:]

    def _increment_miss_counts(self):
        """すべての軌跡のmiss_countを増やす"""
        for traj_info in self.trajectories.values():
            traj_info['miss_count'] += 1

    def _filter_inactive_tracks(self):
        """非アクティブな軌跡を削除"""
        inactive_track_ids = [
            track_id
            for track_id, traj_info in self.trajectories.items()
            if traj_info['miss_count'] > self.miss_tolerance
        ]

        for track_id in inactive_track_ids:
            id_label = self.trajectories[track_id]['id_label']

            # IDラベルをプールに返却
            self.available_ids.append(id_label)

            # 削除
            del self.trajectories[track_id]
            del self.id_label_to_track_id[id_label]


# ========================================
# 形状ガイド
# ========================================
"""
【RuntimeTracker処理フロー】

1. フレーム入力: (3, H, W) or (1, 3, H, W)

2. DETR検出:
   - pred_boxes: (N, 4)
   - pred_scores: (N,)
   - trajectory_features: (N, C)

3. フィルタリング (det_thresh=0.3):
   - active_boxes: (M, 4)
   - active_scores: (M,)
   - active_features: (M, C)

4. ID Decoder:
   - current_tokens: (1, 1, M, C+id_dim)
   - trajectory_tokens: (G, L, C+id_dim) - 履歴軌跡
   - trajectory_times: (G, L)
   → id_probs: (M, K+1)

5. ID割り当て (Assignment Protocol):
   - id_labels: (M,) - 1~K or K+1

6. 軌跡更新:
   - 新規軌跡作成 (newborn_thresh=0.6)
   - 既存軌跡に追加
   - miss_count更新

7. 非アクティブ軌跡削除 (miss_tolerance=30):
   - miss_count > 30 の軌跡を削除

8. 出力:
   - tracks: [{'id': track_id, 'box': [x, y, w, h], 'score': conf}, ...]

【ID辞書管理】
- available_ids: deque([1, 2, ..., K]) - 利用可能なIDラベル
- id_label_to_track_id: {id_label: track_id} - マッピング
- next_track_id: 次のトラックID (単調増加)

【軌跡情報】
trajectories[track_id] = {
    'id_label': int,               # 割り当てられたIDラベル (1~K)
    'features': List[tensor],      # 過去の特徴 [(C,), ...]
    'boxes': List[numpy],          # バウンディングボックス [(4,), ...]
    'scores': List[float],         # 検出信頼度
    'timestamps': List[int],       # フレームインデックス
    'miss_count': int,             # 検出されなかった連続フレーム数
}

【ハイパーパラメータ】
- det_thresh: 0.3 - 検出信頼度閾値
- id_thresh: 0.2 - ID確率閾値
- newborn_thresh: 0.6 - 新規物体の検出信頼度閾値
- miss_tolerance: 30 - 検出されないフレーム数の許容値
"""
