"""
Kronos Understanding - Inference Pipeline (簡略化疑似コード)

自己回帰推論パイプライン。事前学習済みモデルを使用して
未来のK線データを生成する。Monte Carloロールアウトにより
複数サンプルを平均化してロバストな予測を実現。

対応する公式実装:
  - model/kronos.py: auto_regressive_inference(), KronosPredictor, calc_time_stamps()

論文参照: Section 3 "Inference", Section 4 "Test-Time Scaling"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# Top-k / Top-p (Nucleus) Sampling
# ============================================================

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Top-k / Top-p (nucleus) フィルタリング

    確率分布からサンプリングする前に、低確率のトークンをマスクする。
    LLMのテキスト生成と同じ手法を金融トークン生成に適用。

    入力: logits (B, vocab_size) - スケーリング前のロジット
    出力: logits (B, vocab_size) - フィルタリング済みロジット

    Top-k: 上位k個のトークンのみ残す
    Top-p: 累積確率がp以上になるまでのトークンを残す (nucleus)
    """
    if top_k > 0:
        # --- Top-k フィルタリング ---
        # 上位k個以外のロジットを -inf に設定
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1, None]
        # kth_value: (B, 1) - k番目に大きい値
        logits[logits < kth_value] = filter_value
        return logits

    if top_p < 1.0:
        # --- Top-p (Nucleus) フィルタリング ---
        # 累積確率が top_p を超えるトークンをマスク

        # 1. ロジットを降順ソート
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # sorted_logits: (B, vocab_size) - 降順

        # 2. 累積確率を計算
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # cumulative_probs: (B, vocab_size) - [0.3, 0.55, 0.75, 0.88, 0.95, ...]

        # 3. 閾値を超えたトークンを除去 (ただし最低1つは残す)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False  # 最も確率の高いトークンは必ず残す

        # 4. 元のインデックス順に戻してマスク適用
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits

    return logits


def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.9):
    """
    ロジットからトークンをサンプリング

    入力: logits (B, vocab_size)
    出力: sampled_ids (B, 1)

    処理:
        1. Temperature スケーリング: logits / T
           - T < 1: シャープな分布 (高確信サンプリング)
           - T = 1: 元の分布
           - T > 1: フラットな分布 (多様なサンプリング)
        2. Top-k / Top-p フィルタリング
        3. Softmax → 多項分布サンプリング
    """
    # Temperature スケーリング
    logits = logits / temperature
    # logits: (B, vocab_size)

    # Top-k / Top-p フィルタリング
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Softmax → サンプリング
    probs = F.softmax(logits, dim=-1)
    # probs: (B, vocab_size) - 確率分布

    sampled = torch.multinomial(probs, num_samples=1)
    # sampled: (B, 1) - サンプリングされたトークンID

    return sampled


# ============================================================
# 自己回帰推論
# ============================================================

def auto_regressive_inference(
    tokenizer,      # KronosTokenizer (凍結)
    model,          # Kronos モデル (凍結)
    x,              # (B, T, 6) - 正規化済み入力K線
    x_stamp,        # (B, T, 5) - 入力の時間特徴
    y_stamp,        # (B, H, 5) - 予測先の時間特徴
    max_context=512,# 最大コンテキスト長
    pred_len=48,    # 予測長 H
    clip=5,         # クリッピング閾値
    T=0.6,          # Temperature
    top_k=0,        # Top-k (0 = 無効)
    top_p=0.9,      # Top-p (nucleus)
    sample_count=10,# Monte Carloサンプル数 N
):
    """
    自己回帰推論メイン関数

    入力:
        x:       (B, T, 6) - 正規化済みK線 [Open, High, Low, Close, Vol, Amt]
        x_stamp: (B, T, 5) - 履歴の時間特徴 [min, hour, wday, day, month]
        y_stamp: (B, H, 5) - 予測先の時間特徴
        sample_count: Monte Carloサンプル数 (Test-Time Scaling)

    出力:
        preds: (B, T+H, 6) - 予測K線 (入力 + 生成部分)

    処理フロー:
        1. 入力を sample_count 回複製 → (B*N, T, 6)
        2. トークン化 → s1_ids, s2_ids (B*N, T)
        3. 自己回帰ループ H回:
            a. スライディングウィンドウ準備 (最大 max_context)
            b. model.decode_s1() → s1サンプリング
            c. model.decode_s2() → s2サンプリング
            d. バッファ更新
        4. トークナイザでデコード → (B*N, T+H, 6)
        5. N個のサンプルを平均化 → (B, T+H, 6)
    """
    with torch.no_grad():
        # === クリッピング ===
        x = torch.clip(x, -clip, clip)

        device = x.device
        B_orig = x.size(0)

        # === Step 1: Monte Carloサンプル用に入力を複製 ===
        # (B, T, 6) → (B*N, T, 6)
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))
        # x: (B*N, T, 6), x_stamp: (B*N, T, 5), y_stamp: (B*N, H, 5)

        # === Step 2: 履歴をトークン化 ===
        x_token = tokenizer.encode(x, half=True)
        # x_token = [s1_ids (B*N, T), s2_ids (B*N, T)]

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)  # B*N
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        # full_stamp: (B*N, T+H, 5)

        # === Step 3: バッファ初期化 ===
        # スライディングウィンドウ用バッファ (最大 max_context)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)   # s1用
        post_buffer = x_token[1].new_zeros(batch_size, max_context)  # s2用

        # 履歴トークンをバッファにコピー
        buffer_len = min(initial_seq_len, max_context)
        start_idx = max(0, initial_seq_len - max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        # 生成結果格納用
        generated_pre = x_token[0].new_empty(batch_size, pred_len)   # (B*N, H)
        generated_post = x_token[1].new_empty(batch_size, pred_len)  # (B*N, H)

        # === Step 4: 自己回帰ループ ===
        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            # --- ウィンドウ取得 ---
            if current_seq_len <= max_context:
                input_s1 = pre_buffer[:, :window_len]    # (B*N, window_len)
                input_s2 = post_buffer[:, :window_len]   # (B*N, window_len)
            else:
                input_s1 = pre_buffer   # (B*N, max_context)
                input_s2 = post_buffer  # (B*N, max_context)

            # --- 時間スタンプ取得 ---
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :]
            # current_stamp: (B*N, window_len, 5)

            # --- s1予測 (Coarse) ---
            s1_logits, context = model.decode_s1(input_s1, input_s2, current_stamp)
            s1_logits = s1_logits[:, -1, :]  # 最後のステップのみ
            # s1_logits: (B*N, 1024)

            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p)
            # sample_pre: (B*N, 1)

            # --- s2予測 (Fine, s1条件付き) ---
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]  # 最後のステップのみ
            # s2_logits: (B*N, 1024)

            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p)
            # sample_post: (B*N, 1)

            # --- 生成結果を保存 ---
            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            # --- バッファ更新 (スライディングウィンドウ) ---
            if current_seq_len < max_context:
                # バッファに空きがある場合: 末尾に追加
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                # バッファが満杯: 1つシフトして末尾に追加
                pre_buffer = torch.roll(pre_buffer, shifts=-1, dims=1)
                post_buffer = torch.roll(post_buffer, shifts=-1, dims=1)
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        # === Step 5: 全トークンをデコード ===
        # 履歴 + 生成をconcat
        full_pre = torch.cat([x_token[0], generated_pre], dim=1)    # (B*N, T+H)
        full_post = torch.cat([x_token[1], generated_post], dim=1)  # (B*N, T+H)

        # コンテキストウィンドウ内のトークンのみデコード
        total_seq_len = initial_seq_len + pred_len
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len],
            full_post[:, context_start:total_seq_len],
        ]

        z = tokenizer.decode(input_tokens, half=True)
        # z: (B*N, window, 6)

        # === Step 6: Monte Carlo平均化 ===
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        # z: (B, N, window, 6)

        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)
        # preds: (B, window, 6) - N個のサンプルの平均

        return preds


# ============================================================
# 時間特徴計算
# ============================================================

def calc_time_stamps(timestamps):
    """
    DatetimeIndex → 時間特徴テーブル

    入力: timestamps (pandas DatetimeIndex or Series)
    出力: DataFrame with columns [minute, hour, weekday, day, month]

    例:
        2024-07-15 09:35:00 → [35, 9, 0, 15, 7]
        2024-07-15 14:00:00 → [0, 14, 0, 15, 7]
    """
    time_df = pd.DataFrame()
    time_df['minute'] = timestamps.dt.minute    # 0-59
    time_df['hour'] = timestamps.dt.hour        # 0-23
    time_df['weekday'] = timestamps.dt.weekday  # 0=月曜, 6=日曜
    time_df['day'] = timestamps.dt.day          # 1-31
    time_df['month'] = timestamps.dt.month      # 1-12
    return time_df


# ============================================================
# KronosPredictor (高レベルAPI)
# ============================================================

class KronosPredictor:
    """
    推論用の高レベルAPI

    使用方法:
        1. モデル・トークナイザのロード
        2. predict() で単一系列予測 / predict_batch() でバッチ予測

    処理:
        1. DataFrameの前処理 (正規化、クリッピング、時間特徴抽出)
        2. auto_regressive_inference() で自己回帰生成
        3. 逆正規化して元のスケールに戻す
    """

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        """
        Args:
            model: Kronos モデル
            tokenizer: KronosTokenizer
            device: 推論デバイス (自動検出可)
            max_context: 最大コンテキスト長 (512 or 2048)
            clip: Z-score正規化後のクリッピング閾値
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_col = 'amount'

        # デバイス自動検出
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer.to(self.device)
        self.model.to(self.device)

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        単一K線系列の予測

        入力:
            df: DataFrame with [open, high, low, close, volume, amount]
            x_timestamp: 履歴のDatetimeIndex (len = len(df))
            y_timestamp: 予測先のDatetimeIndex (len = pred_len)
            pred_len: 予測ステップ数
            T: Temperature (0.6=予測向け, 1.0=生成向け)
            top_p: Nucleus sampling閾値
            sample_count: Monte Carloサンプル数

        出力:
            pred_df: DataFrame (pred_len, 6) - 予測K線
                     columns=[open, high, low, close, volume, amount]
                     index=y_timestamp

        処理:
            1. データ抽出 + 欠損値処理
            2. Z-score正規化 (各特徴独立)
            3. クリッピング [-clip, clip]
            4. 自己回帰推論
            5. 逆正規化
            6. DataFrame化
        """
        df = df.copy()

        # === Step 1: データ準備 ===
        # Volume/Amount が欠損なら0埋め
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_col] = 0.0

        # 時間特徴計算
        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        # 数値配列化
        x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
        # x: (T, 6)
        x_stamp = x_time_df.values.astype(np.float32)  # (T, 5)
        y_stamp = y_time_df.values.astype(np.float32)   # (H, 5)

        # === Step 2: Z-score正規化 (各特徴独立) ===
        x_mean = np.mean(x, axis=0)  # (6,) 各特徴の平均
        x_std = np.std(x, axis=0)    # (6,) 各特徴の標準偏差
        x = (x - x_mean) / (x_std + 1e-5)

        # === Step 3: クリッピング ===
        x = np.clip(x, -self.clip, self.clip)
        # x: (T, 6), 値 ∈ [-5, 5]

        # === Step 4: バッチ次元追加 ===
        x = x[np.newaxis, :]           # (1, T, 6)
        x_stamp = x_stamp[np.newaxis, :]  # (1, T, 5)
        y_stamp = y_stamp[np.newaxis, :]  # (1, H, 5)

        # === Step 5: 自己回帰推論 ===
        preds = self._generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count)
        # preds: (1, H, 6)

        preds = preds.squeeze(0)  # (H, 6)

        # === Step 6: 逆正規化 ===
        preds = preds * (x_std + 1e-5) + x_mean
        # preds: (H, 6) - 元のスケール

        # === Step 7: DataFrame化 ===
        pred_df = pd.DataFrame(
            preds,
            columns=self.price_cols + [self.vol_col, self.amt_col],
            index=y_timestamp
        )
        return pred_df

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kwargs):
        """
        バッチ予測 (複数系列を並列処理)

        入力:
            df_list: List[DataFrame] - 複数のK線系列
            x_timestamp_list: List[DatetimeIndex] - 各系列の履歴タイムスタンプ
            y_timestamp_list: List[DatetimeIndex] - 各系列の予測先タイムスタンプ
            pred_len: 予測ステップ数

        出力: List[DataFrame] - 各系列の予測結果

        注意:
            - 全系列の履歴長が同一である必要がある
            - 全系列の予測長が同一である必要がある
        """
        num_series = len(df_list)
        x_list, x_stamp_list, y_stamp_list = [], [], []
        means, stds = [], []

        for i in range(num_series):
            df = df_list[i].copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_col] = 0.0

            x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
            x_stamp = calc_time_stamps(x_timestamp_list[i]).values.astype(np.float32)
            y_stamp = calc_time_stamps(y_timestamp_list[i]).values.astype(np.float32)

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

        # バッチ化: (N_series, T, 6)
        x_batch = np.stack(x_list, axis=0)
        x_stamp_batch = np.stack(x_stamp_list, axis=0)
        y_stamp_batch = np.stack(y_stamp_list, axis=0)

        # 推論
        preds = self._generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, **kwargs)
        # preds: (N_series, H, 6)

        # 逆正規化 + DataFrame化
        results = []
        for i in range(num_series):
            pred_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(
                pred_i,
                columns=self.price_cols + [self.vol_col, self.amt_col],
                index=y_timestamp_list[i]
            )
            results.append(pred_df)

        return results

    def _generate(self, x, x_stamp, y_stamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1):
        """内部生成関数: numpy → tensor → 推論 → numpy"""
        x_t = torch.from_numpy(x).to(self.device)
        xs_t = torch.from_numpy(x_stamp).to(self.device)
        ys_t = torch.from_numpy(y_stamp).to(self.device)

        preds = auto_regressive_inference(
            self.tokenizer, self.model, x_t, xs_t, ys_t,
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count
        )
        return preds[:, -pred_len:, :]


# ============================================================
# 使用例
# ============================================================

if __name__ == "__main__":
    """
    # 実際の使用方法 (公式リポジトリより):

    from model.kronos import KronosTokenizer, Kronos, KronosPredictor

    # Step 1: モデルロード (Hugging Face Hub)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # Step 2: Predictor初期化
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # Step 3: データ準備
    df = pd.read_csv("kline_data.csv")
    x_timestamp = pd.to_datetime(df['timestamps'][:480])
    y_timestamp = pd.date_range(start=x_timestamp.iloc[-1], periods=96, freq='5min')[1:]

    # Step 4: 予測
    pred_df = predictor.predict(
        df.iloc[:480],         # 履歴480本
        x_timestamp,
        y_timestamp,
        pred_len=96,           # 96本先まで予測
        T=0.6,                 # 予測向けTemperature
        top_p=0.9,             # Nucleus sampling
        sample_count=10,       # 10回サンプリング → 平均
    )

    print(pred_df)
    # Output:
    #                      open    high     low   close    volume  amount
    # 2024-07-15 10:00:00  182.45  183.12  182.00  182.88  15000   ...
    # 2024-07-15 10:05:00  182.88  183.55  182.45  183.22  12000   ...
    # ...
    """
    print("Inference pipeline pseudo-code. See usage example in docstring.")
