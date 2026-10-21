# CycleNet — 位相を揃えた周期テンプレートを引いて残差を学習

## 概要

CycleNet は「周期らしい特徴を暗黙的にネットワークに発見させる」のではなく、長さ `W`・チャンネル数 `C` の学習可能な周期表 `Q:[W,C]` を明示的に持つ。履歴から該当位相の `Q` を引き、残差だけを軽い Linear/MLP で予測し、未来の位相に対応する `Q` を足し戻す。この操作を *Residual Cycle Forecasting*（RCF）と呼ぶ。論文は RCF が他の予測器にも追加できる点も検証する。

ここではその周期除去・残差変換・周期復元を `x:[B,T,Cin] -> h:[B,T,Cout]` の encoder として使う。原著の forecast horizon `H` は固定せず、タスク head や loss を block に含めない。

## 新規性・既存手法との差分

| 方式 | 周期の扱い |
| --- | --- |
| moving-average 系の trend/season 分解 | 窓の局所平滑化で成分を推定する。絶対的な周期位相の共有表は持たない。 |
| 周波数選択/FFT 系 | 窓から周期を検出・周波数特徴として処理する。 |
| CycleNet RCF | データセット全体で共有する位相ごとの学習パラメータ `Q[w,c]` を使い、既知の窓開始位相で明示的に位置合わせする。 |

テンプレートはチャンネルごとに異なる波形を持つが、残差用 Linear/MLP はチャンネル間で重み共有。`W` が分かり位相が揃う安定周期データに適する。未知/変動周期、異なるチャンネル周期、外れ値の多いデータでは原著も限界を指摘する。効率表では RCF 自体のパラメータ増加は `W*C` だが、位相整列に伴い DLinear より学習時間が必ず短いわけではない。

## モデルアーキテクチャ全体像（論文の意図する `L→H`）

CycleNet では `Q` を一度算出した Fourier 係数や入力サンプルごとの隠れ状態として扱わない。`Q∈R^{W×C}` は**学習パラメータそのもの**で、窓ごとの位相 `i_b` を使って必要な長さに gather する。二つの gather の開始位置が違うことが RCF の中心。

```text
                            学習パラメータ Q [W,C]
                             ├───────────────┐
              phase i [B] ───┤               │
                             ▼               ▼
                   Q_history[(i+t)%W]   Q_future[(i+L+t)%W]
                       [B,L,C]                [B,H,C]
                          │                      │
過去観測 X [B,L,C]       │                      │
      │ window μ,σ [B,1,C]│                      │
      ▼                   │                      │
RevIN normalization       │                      │
      │ Z [B,L,C]         │                      │
      └────────(-)────────┘                      │
                │ residual R [B,L,C]             │
                ▼                                │
       [B,C,L] （チャンネルを独立に）           │
                │ 同一 Linear(L,H) または MLP    │
                ▼                                │
       residual forecast [B,C,H]                │
                │ transpose [B,H,C]              │
                └───────────(+)──────────────────┘
                            │ [B,H,C]
                            ▼ inverse RevIN
                       prediction [B,H,C]
                            │
          将来実測 GT [B,H,C] と要素平均 MSE → scalar
```

図の `i` は**入力窓先頭に揃えたい場合**の位相。未来を足し戻す方は `i+L` から始める。ただし公開 DataLoader の `cycle_index=self.cycle_index[s_end]` は窓先頭 `s_begin` ではなく**窓の直後 `s_end=s_begin+L`**の位相を渡している。公開 `Model` はその index をそのまま `Q_history` の起点にするため、実際の公開実験では Q の行が入力時刻に対して L だけ回転している。本 block/finetuning 例は窓先頭の位相を渡す設計であり、この点は公式データ経路と異なる。固定 L なら Q 自体が回転した位相を学習できる余地はあるが、公式 `cycle_index` と本版 `phase` を無変換で交換してはいけない。

## 原著の forward と objective

`B`: batch、`L`: 履歴長、`H`: 未来長、`C`: 変数、`W`: 周期長。以下の `i_b` は**窓先頭に整列した説明用位相** `[B]`。公式 `RecurrentCycle` 自体は渡された index に `(index+t)%W` を適用するだけで、その index が窓先頭かどうかは確認しない。実際の公式 DataLoader は直後時刻の index を渡す。

```text
history X                     [B,L,C]
phase i                       [B] integer; start index mod W
μ,σ = window statistics       each [B,1,C]
Z=(X-μ)/σ                     [B,L,C]
Q[(i+t)%W]                    [B,L,C]
R=Z-Q_history                 [B,L,C]
Linear(L→H) または MLP        [B,H,C]  各チャンネル独立に同じ重み
Q[(i+L+t)%W]                  [B,H,C]
forecast=(R_hat+Q_future)*σ+μ [B,H,C]
GT=観測された未来             [B,H,C]
loss=MSE(forecast,GT)         scalar
```

`Q` は zero 初期化だが、loss が出力を経由して `Q` にも勾配を流す。「周期そのもの」を別 GT で教師あり学習するのではない。式で書けば `R_b(t)=Z_b(t)-Q[(i_b+t)%W]`、`ŷ_b(t)=M(R_b)(t)+Q[(i_b+L+t)%W]`。周期の見かけを維持する復元項と、残差の個別変動を担う M が分業する。

## 本実装の encoder アーキテクチャ

```text
x [B,T,Cin] + phase [B]
  → optional window norm [B,T,Cin]
  → gather Q[(phase+t)%W] [B,T,Cin]
  → subtract [B,T,Cin]
  → temporal Linear(T→T) or MLP(T→D→T), shared over Cin [B,T,Cin]
  → 公式と同じ future Q[(phase+T+t)%W] を足し戻す [B,T,Cin]
  → optional denorm [B,T,Cin]
  → Linear(Cin→Cout) [B,T,Cout] = h
  → 任意の pooling/head は利用者側
```

本 block も原著と同じ**未来位相**の Q を足す。出力長を T に設定した公式演算と揃えるため、入力から差し引く Q は `phase+t`、出力へ戻す Q は `phase+T+t` である。`seq_len` 固定なのは時間軸全体への learned Linear のため。`phase=None` は全窓開始位相を 0 とみなす便宜的なデフォルトであり、窓が揃っていないデータで RCF の意味は失われる。既知の時刻から `phase=(window_start_index % cycle_len)` を渡す。`cycle_len` もデータの ACF やドメイン知識で選ぶ。`out_dim≠in_dim` の Linear は block 用の追加で、公式 forecast 本体にない。公式の `torch.var` は不偏推定だが、ここは長さ 1 にも有限な `unbiased=False` を既定値とした。公式と揃える場合は `variance_unbiased=True`（T>1）を使う。

## 使い方・DataFrame 学習

```python
from ts_block import CycleNetBlock

block = CycleNetBlock(in_dim=6, out_dim=32, seq_len=96,
                      cycle_len=24, model_type="mlp")
h = block(x, phase=window_start_phase)  # [B,96,6], [B] → [B,96,32]
prediction = head(h.mean(dim=1))         # Linear(32,K) → [B,K]
```

[finetuning_example.py](finetuning_example.py) は `series_id,timestamp`、特徴列、次行の目的列を持つ DataFrame から train/validation を時刻順に作り、encoder→平均 pooling→回帰/分類 head を学習する。`phase_col` を渡す場合、その列は**各入力時刻の周期位相**でなければならない。サンプルは履歴先頭行の phase を選ぶ。位相を渡さない場合は全窓 0 という仮定を明示的に受け入れる必要がある。これは原著の未来系列 MSE の再現ではなく、下流タスクの一例。

## 実装照合・限界

論文 TeX の RCF 式と公式 `models/CycleNet.py` を照合。`RecurrentCycle` の gather、位相 shift、残差 Linear/MLP、RevIN の順序を追った。原著と本 block の差は `H=T`、同位相への復元、特徴射影、および分散の扱い。位相情報のないデータに「汎用的にただ差すだけ」で優位になることは保証しない。

## 1 サンプルの位相を手で追う

`W=4`, `L=6`, `H=3`, 窓先頭位相 `i=2` とする。`Q` の行は `0,1,2,3`。履歴側が使う位相列は `2,3,0,1,2,3`、未来側は `(i+L)%4=0` からの `0,1,2`。したがって周期表は学習可能だが、予測時に未来の実測を参照しているわけではない。必要なのは窓開始位相と既知の周期長だけ。もしサンプルごとに開始位相が違うのに全て 0 とすると、異なる周期位置へ同じ `Q` 行が割り当てられ、学習される平均周期がぼける。

`Q` は `[W,C]` のため同じチャネル c の同じ位相 w は、訓練中の全窓で共有される。`Q` のパラメータ数は `W*C`。大きな時刻埋込テーブルではなく、各周期位置と変数の予測可能な基準値を学習するという意味である。GT は将来の実測 `Y:[B,H,C]` のみで、`Q` に対応する「周期ラベル」は無い。勾配は `loss→forecast→Q_future` と `loss→forecast→M(X-Q_history)→Q_history` の両方を通る。

## RevIN と周期除去の順序が重要

原著/公開コードの順序は窓ごとの `μ,σ` による正規化、`Q_history` の減算、残差の Linear/MLP、`Q_future` の加算、逆正規化。`Q` は正規化空間の尺度で学習される。もし生の X から Q を引いてから RevIN すると、サンプルごとにテンプレートの見かけの振幅が変わり、同じ計算にはならない。`use_revin=False` なら Q は生スケールのまま学習される。

各変数を `[B,C,L]` へ並べ替えて最後の L 軸へ一つの `nn.Linear(L,H)` を適用する。Linear の重みは C ごとに複製されず**全変数で共有**される。MLP 版も `Linear(L,d_model)→ReLU→Linear(d_model,H)` を同様に共有する。変数間の相関を残差 backbone で直接学ぶ構成ではない。Q にはチャンネル別列があるが、これは他チャンネルを条件に各時刻を予測する cross-channel attention とは違う。

## 原著実験から言える範囲

TeX は RCF を PatchTST/iTransformer に付けた実験、周期長 W を変えた実験、parameter/MAC と学習速度の比較を含む。`W` がデータの最大安定周期と合わないと RCF の利点が薄れ、ほぼ RCF 無しと同じ精度になると論文は述べる。Traffic の外れ値では MAE は改善しても MSE が悪化する例があり、周期平均テンプレートが極端値へ引かれる可能性が議論されている。これらは「RCF は必ず精度を改善する」という結論ではなく、安定周期・正しい位相・外れ値の少なさが前提であることを示す。

効率面でも `Q` の追加 MAC はほぼない一方、各サンプルの位相 gather が発生する。論文は CycleNet/Linear が DLinear より**訓練速度では遅い**点も記す。モデルサイズが小さいことと wall-clock が常に速いことは別。ここでの `Linear(T,T)` は seq_len に比例してパラメータが増えるため、T を長くすると Q より temporal mixer が主な容量となる。

## 原著と携帯版の対応・検証

| 部品 | 公式 `models/CycleNet.py` | 本版 [ts_block.py](ts_block.py) | 判定 |
| --- | --- | --- | --- |
| 共有周期表 | `RecurrentCycle.data:[W,C]` | `RecurrentCycle.data:[W,Cin]` | 同じ gather 規則 |
| normalization | `torch.mean`, `torch.var` | 同じ順序、分散推定は切替可 | `variance_unbiased=True` で公式と揃う |
| residual mixer | `Linear(L,H)` / MLP | `Linear(T,T)` / MLP | H=T の場合だけ重み shape が一致 |
| 復元位相 | `(cycle_index+L)%W` | `(phase+T)%W` | `L=T` と同じ index で一致 |
| index の供給 | DataLoader は `cycle_index[s_end]` | Dataset は `phase[s_begin]` | **公開実験と本版で L だけ異なる** |
| channel output | C を維持 | `Linear(Cin,Cout)` | task-neutral adapter |

親ディレクトリの [validate_new_blocks.py](../validate_new_blocks.py) は `T=16,17`、`W=4,H=T` で公式モデルから Q と Linear または MLP の重みをコピーし、出力射影を単位行列にして**同じ index を明示的に渡した場合**の FP32 出力を比較する。`T%W=0` でも `T%W≠0` でも公式 `Model.forward` と一致する。ただし DataLoader が供給する index の意味は別問題であり、同じ元データから作った窓でも本サンプルの窓先頭 phase と公式の窓末尾 index を混同してはいけない。

## 汎用 encoder としての利用判断

位相が分かる周期データなら Q によって説明可能な反復を先に除き、残差の変動を軽い mixer へ任せられる。窓長 T や周期 W が異なる複数データセットを同一 block で横断したい場合、`Q:[W,C]` と `Linear(T,T)` の両方が固定なので通常は再構築が必要。タイムスタンプが非等間隔・休日飛び・欠測を含むと `time_index%W` の意味が変わるため、業務カレンダー上の位相を別列として与える。未知周期や不規則時系列には「phase=0 でとりあえず動く」ことを RCF が機能した証拠としない。

~~~text
train:  start_phase = window_start の位相 [B]
        history [B,T,Cin] → CycleNetBlock(history,start_phase)
        h [B,T,Cout] → mean_time [B,Cout] → head [B,K]
        next-row GT [B,K] / class [B] → MSE / CE → backward
infer:  同じ feature scaler・同じ W・正しい start_phase で呼ぶ
~~~

## 同長 encoder で周期が打ち消されないのか

入力の周期 `Q_in` を引き、出力に未来位相の `Q_out` を足すので、「結局 Q はキャンセルされるだけでは？」という疑問が自然に出る。しかし時間 mixer M が恒等関数でも `M(X-Q_in)+Q_out=X-Q_in+Q_out` であり、`T%W≠0` なら二つの Q は通常異なる。線形 M なら `M(X-Q_in)+Q_out=M(X)-M(Q_in)+Q_out`。`T%W=0` の場合に限って `Q_out=Q_in` となり、`M(X)+(I-M)Q_in` と書ける。いずれも周期表は残差変換の基準として作用する。

ただし現実に `Q` が役立つかは、周期が安定し、位相が正確で、head が周期に関係する target を持つかで決まる。下流 target が周期と無関係なら Q がほぼ 0 に留まる、または mixer がキャンセルする可能性もある。loss を見ずに Q を可視化して「周期が学べた」と結論するのは危険。周期表の各行に十分な訓練例があるか、位相別の残差平均や validation 指標と一緒に検討する。

## 公式の位相 gather とサンプル生成

`RecurrentCycle.forward(index,length)` は Python loop で Q を回転させるのではなく、`gather_index=(index[:,None]+arange(length)[None,:])%W` を作る。これは `[B,T]` 個の整数 index を一度に計算し、`Q[gather_index]` で `[B,T,C]` を取得する。学習では同一 Q の行に複数窓から勾配が集まる。DataFrame 例の `phase_col` は**元 DataFrame 各行に対する位相**を保持し、WindowDataset が窓の最初の行 `label_index-T` の値を渡す。

例えば営業日だけからなる市場系列で「暦日 % 24」を用いても、欠場日で位相が飛ぶため意味が薄い。何を周期とみなすか（観測 index の周期、曜日、時間帯など）に応じて、`phase_col` の定義と `cycle_len` をセットで決める。説明可能な位相を用意できないなら別モデルや位相推定器を検討し、`phase=None` の便宜的 0 を性能評価の本設定とみなさない。

## RCF と時刻 embedding・固定季節分解の違い

時刻 embedding も曜日・時刻 ID に対する学習ベクトルを持てるが、多くの場合それは隠れ特徴へ**追加**され、残差予測の入力・出力から同じ周期成分を明示的に除去/復元するとは限らない。RCF は `X-Q_history` という分解を予測器の前に固定した設計であり、`M(residual)+Q_future` という形の帰納バイアスを持つ。古典的な季節分解は過去窓から成分を推定するが、RCF の Q は訓練データ全体で共有される。新しい窓に周期が十分現れなくても、学習済み Q の対応位相を取り出せるのが強み。ただし周期が時間とともに変わると、全期間共有の Q が古くなる。

`Q` の zero 初期化は「学習開始時には RCF なし」に近い状態を作る。そこから GT との誤差が Q を更新するので、何らかの周期波形が予め与えられるわけではない。Q を良く学ぶには、各位相 w が複数のサンプル窓で観測される必要がある。例えば W を観測期間より長く設定すると未観測の行が残り、未来で初めてその位相へ到達する際に一般化できない。原著の W は物理的に意味のある最大周期を選ぶことを推奨し、ACF のピークで確認している。

## 予測と分類で同じ `h` をどう使うか

原著は未来全区間 `[B,H,C]` の点ごと MSE を使う。本版の `[B,T,Cout]` は履歴の各時刻に対応する特徴だから、系列単位の回帰なら `mean_t h` または `h[:,-1]` を head に渡す。時刻別の分類なら `Linear(Cout,K)` を各 `h[:,t]` に適用して `[B,T,K]` を得る。この場合 GT は `[B,T]` の class index となり、loss は観測済み時刻だけで CE を計算する。**同じ block でも GT の shape と損失は downstream task で変える。**

周期表は `Cin` 個の特徴列に対応するため、入力特徴列の順序や意味を checkpoint と異なるまま交換すると、Q の列が別特徴に適用される。外部データセットへ移すときは単に `seq_len` を合わせるだけでは不十分で、周期長・位相定義・sampling interval・特徴列の意味も合わせる必要がある。

## 抽象的なコストではなく、どこが計算されるか

| 部分 | 入力規模 | 主なコスト |
| --- | --- | --- |
| Q gather | `phase:[B]`, `Q:[W,C]` | `[B,T,C]` の index 展開と memory read |
| Linear 残差 mixer | `[B,C,T]` | `T→T` 行列を各 B,C に適用 |
| MLP 残差 mixer | `[B,C,T]` | `T→d_model→T` の二つの時間行列 |
| RevIN | `[B,T,C]` | 時刻方向の平均・分散、復元 |
| output projection | `[B,T,C]` | 各時刻の `Cin→Cout` 行列 |

`Q gather` 自体には乗算行列はほぼ無いが、GPU で小規模バッチなら indexing launch とメモリアクセスが相対的に支配的になり得る。原著の parameter/MAC の少なさをこの携帯版のレイテンシ保証と読まない。公式 core との parity 検証はその演算順が正しいかの確認で、実測速度や精度を代替しない。

## DataFrame からの位相列の例

等間隔で欠損なく各 `series_id` が始まるデータなら、各行の観測 index を `cycle_len` で割った余りを `phase_col` として用意できる。複数系列で絶対位相が揃う場合に限り、系列ごとの連番ではなく共通の物理時刻から計算する。以下は**データ生成の形**だけを示し、block 内で label を加工しない。

```python
df = df.sort_values(["series_id", "timestamp"]).copy()
df["phase"] = df.groupby("series_id").cumcount() % 24
result = fit(df, ["sensor_a", "sensor_b"], ["target"],
             seq_len=96, phase_col="phase", epochs=10)
model = result["model"]  # encoder + pooling + head が学習済み
```

この例は系列ごとに位相 0 から始める仮定を置く。週周期なのに各系列の開始曜日が異なる場合は誤りなので、timestamp から曜日・時間帯を直接符号化する。validation 区間の phase を作るのに future target 値は不要だが、時刻のカレンダー情報は推論時にも入手できることが前提。

## 一次資料

原論文: [CycleNet (NeurIPS 2024)](https://arxiv.org/abs/2409.18479)。公式実装: [ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet)。参照したのは論文 `CycleNet.tex` の RCF/Instance normalization/Loss/Efficiency/Discussion と公開 `models/CycleNet.py`。論文の forecast benchmark 精度をこの同長 encoder が再現するとは主張しない。
