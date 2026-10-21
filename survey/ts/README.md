# 時系列モデル survey と再利用ブロック

各ディレクトリには論文・公開実装の survey、独立した `ts_block.py`、pandas DataFrame から実際に学習する `finetuning_example.py` を置いた。**主目的は、回帰・分類など任意の下流タスクへ差し込める系列 encoder を提供すること。** 各ブロックの共通入出力は **`x:[B,T,Cin] → h:[B,T,Cout]`**（B: batch、T: 入力の時間長、C: 特徴/ノード数）。`ts_block.py` は GT・loss・pooling・task head を決めない。原著の予測モデルが全て同長出力という意味でもない。

一例としては、`h = block(x)` の後に `pooled = h.mean(dim=1)`（`[B,Cout]`）と `nn.Linear(Cout,K)` を組み合わせ、系列単位の回帰値・分類 logits を作れる。pooling は attention pooling などに、head は MLP などに自由に置き換えられる。時刻ごとのタスクなら pooling せず `h:[B,T,Cout]` に時刻別 head を付ける。`finetuning_example.py` は downstream の**一例**であり、ブロックに GT・予測 head・MSE/CE が内蔵されていることを意味しない。モデルの論文学習objectiveとこの汎用学習例は各READMEで分けて説明する。

| モデル | 公開年 | 原著の系列間処理 | 本ブロック: out projection 前に融合？ | 本ブロックの系列間処理 | 主な機構・注意 |
| --- | --- | --- | :---: | --- | --- |
| [P-sLSTM](PsLSTM/README.md) | 2024 | **独立**。`[B,L,C]→[B·C,L]` 相当。変数 ID 埋込なし | × | sLSTM は独立。最後の `Cin→Cout` 射影のみ融合 | patch 化 sLSTM。recurrent 逐次計算、patch 内は非因果 |
| [Mamba](Mamba/README.md) | 1: 2023、2: 2024、3: 2026 | 汎用系列 mixer。時系列の単変量／多変量という原著の区分はない | ○ | 入力 `Cin→d_model` で融合してから Mamba に渡す | 版ごとに異なる因果 SSM。高速化には公式 fused kernel が重要 |
| [DST-Mamba](DSTMamba/README.md) | 2025 | **融合**。交通センサを node として双方向に scan | ○ | node 軸の双方向 scan で融合 | node の意味と順序が固定される前提 |
| [WaveTS](WaveTS/README.md) | 2025 | **独立**。wavelet/FRU を各変数に同じ重みで適用 | × | wavelet/FRU は独立。最後の学習可能な射影で融合 | 多尺度 wavelet。原著の予測 FRU を同長へ変更 |
| [ModernTCN](ModernTCN/README.md) | 2024 | **融合**。変数間 FFN2 を明示的に設ける | ○ | FFN2 で融合 | 大 kernel DWConv。非因果、BN 融合は eval 後のみ |
| [AdaWaveNet](AdaWaveNet/README.md) | 2024 | **融合**。粗尺度で変数間 attention | ○ | 粗尺度の変数間 attention で融合 | 学習可能 lifting。固定 `seq_len`、独立 P/U を持つ公式 decoder に一致 |
| [WFTNet](WFTNet/README.md) | 2023 | **融合**。入力変数を embedding で結合 | ○ | 入力 `Conv1d(Cin→D)` で融合 | FFT 周期 map + 公式 `ptwt.cwt`。公式指定版の精度 10 に固定 |
| [WPMixer](WPMixer/README.md) | 2024 | **独立**。帯域別 mixer を変数ごとに適用 | × | mixer は独立。最後の `Cin→Cout` 射影のみ融合 | 帯域別 patch mixer。携帯版は Haar/db1 のみ、固定 `seq_len` |
| [CycleNet](CycleNet/README.md) | 2024 | **変数別条件付けのみ**。変数ごとの周期表 `Q[:,c]`、残差 mixer は独立・重み共有 | × | 同じ独立処理の後、最後の `Cin→Cout` 射影のみ融合 | 位相と周期長が必要。固定 `seq_len` |
| [TimeMixer](TimeMixer/README.md) | 2024 | 設定次第。CI=0 は入力埋込で**融合**、CI=1 は**独立** | CI=False: ○ / CI=True: × | CI=False は入力で融合。CI=True は最後の射影のみ融合 | seasonal/trend の多尺度 PDM。原著 FMM は予測 head |
| [FITS](FITS/README.md) | 2024 | **独立**。周波数変換は重み共有または変数別だが系列間は混ぜない | × | FFT/複素 Linear は独立。最後の `Cin→Cout` 射影のみ融合 | 低周波複素 Linear。高周波を削除 |
| [Timer-XL](TimerXL/README.md) | 2024（ICLR 2025） | **融合**。同時刻以前の他変数patchをTimeAttentionで参照 | ○ | 因果mask付き系列間attention | 次patch予測由来。`T%patch_len=0`、長さ可変、密attentionは`(Cin·N)²` |
| [PatchTST](PatchTST/README.md) | 2022（ICLR 2023） | **独立**。各変数patchを`B·Cin`系列として重み共有処理 | × | 最後の`Cin→Cout`射影のみ融合 | overlap patch＋residual attention。固定`seq_len` |
| [TQNet](TQNet/README.md) | 2025 | **融合**。周期queryが全変数履歴をkey/valueとして参照 | ○ | out projection前に`Cin×Cin`attention | 窓先頭の周期位相が必要。固定`seq_len` |
| [TimeXer](TimeXer/README.md) | 2024 | **融合**。対象global tokenが全外部変数tokenを読む | ○ | global-to-exogenous cross-attention | 全Cを対象にする公開M経路は`Cin²`コスト。固定`seq_len` |
| [TimeBridge](TimeBridge/README.md) | 2024（ICML 2025） | **融合**。縮約patchごとに変数間attention | ○ | 公式既定はaxial系列間attention | 時間内attention→patch縮約→系列間attention。固定`seq_len` |
| [iTransformer](iTransformer/README.md) | 2023（ICLR 2024） | **融合**。1変数全履歴を1tokenにして系列間attention | ○ | `Cin×Cin`attention | 時間軸をLinearで埋込。固定`seq_len` |

公開年はプレプリントが先行する場合、その初公開年を記した（例: WPMixer は 2024 年プレプリント、AAAI 2025）。ここでは**系列間の情報交換をモデル本体で行うものを「融合」**とし、`[B,L,C]` を受け取るだけで各変数を `[B·C,L]` 相当の独立系列として処理するものは「多変量モデル」と呼ばない。P-sLSTM・WaveTS・WPMixer・FITS・PatchTST の原著はこの独立型で、**重み共有だけでは変数種類を識別できず、他系列の値も参照できない**。FITS の変数別重みは変数 ID に相当する違いを持てるが、それでも他系列は参照しない。CycleNet の変数別周期表 `Q[:,c]` は変数ごとの条件付けだが、残差 mixer で系列間 fusion はしない。本ブロックの「最後の射影のみ融合」は汎用 `Cout` 出力のために追加した浅い変換であり、**原著の backbone が多変量相互作用を学ぶことを意味しない**。全 17 ブロックは `Cin=Cout` でも学習可能な系列間経路を持つ。特に WaveTS と DST-Mamba の末尾 `output` は幅が同じ場合も `Linear(Cin,Cout)` を適用する。Mamba-1/2/3 の SISO/MIMO は SSM 内部の入出力構造を指し、時系列の単変量／多変量とは別概念である。

## 入力系列数 `Cin≈1000` のときの計算・メモリ

**`Cin=1000` だけで一律に可否は決まらない。** 通常の LSTM と同様、最初に `Linear(Cin,d_model)` 等で圧縮する構成では、その後の backbone は幅 `d_model` で動く。Mamba と WFTNet、TimeMixer の `channel_independence=False` がこの型。一方、入力の1000系列をそれぞれ保持する構成では、時間処理や中間 activation のコストが `Cin` に比例し、ModernTCN の変数間 FFN と AdaWaveNet の channel attention はさらに `Cin²` の項を持つ。以下は**この repository の `ts_block.py`** の構造から読める増加傾向で、実測速度・精度の保証ではない。`B`: batch、`T`: 系列長、`D`: 隠れ幅、`P`: patch 数、`K`: FITS の保持周波数数、`W`: CycleNet の周期長とする。

| ブロック | `Cin` を増やしたときの主なコスト | `Cin≈1000` での見方 |
| --- | --- | --- |
| P-sLSTM | `B·Cin` 本を独立に patch 化・sLSTM 処理。中間 activation は概ね `B·Cin·P·D`、末尾射影は `B·T·Cin·Cout` | **線形増加だが軽量とは限らない**。1000系列ぶんの recurrent 計算・保存が必要。patch 数と batch を含め実測する |
| Mamba-1/2/3 | 入力射影 `Cin→D` と出力射影 `D→Cout`。SSM 本体の幅は `D` | **圧縮型なので候補**。ただし純 PyTorch scan は系列長 `T` に沿う loop が遅い。GPU では版に合う公式 kernel を測る |
| DST-Mamba | node 数=`Cin` として双方向 scan、node 別 adapter、中間 `[B,Cin,D]`、末尾 `Cin→Cout` | **node 数に概ね線形だが実測必須**。純 PyTorch backend の1000 step scan は特に遅くなり得る。列順・node 意味も固定 |
| WaveTS | 各系列の wavelet/FRU と時間方向 dense Linear。概ね `Cin` 倍の帯域別計算と `B·Cin·T²` 系の時間射影、末尾 `Cin→Cout` | **`T` との積が重要**。短窓なら候補だが、長窓・大 batch・多 order では重い。`seq_len` 固定 |
| ModernTCN | 各 stage の FFN2 が変数軸1000本を dense に混ぜ、パラメータ・演算に `Cin²` の項 | **現実装のまま1000系列は重い**。既定 `D=32`、expansion=2 では FFN2 だけで1層約1.28億、既定2層で約2.56億パラメータ。FP32 weight だけで約1.0 GB、学習時は勾配・optimizer state がさらに必要 |
| AdaWaveNet | 最粗尺度の channel attention が系列1000本を token として扱い、計算に `O(B·Cin²·D)`、attention 行列を実体化する場合は `O(B·heads·Cin²)` メモリ | **1000系列では要注意**。例: `B=16, heads=4, Cin=1000` の attention 行列だけで FP32 約256 MB。実装・kernel により実際の peak は異なる |
| WFTNet | 入力 `Conv1d(Cin→D)` の後は FFT/CWT/2D Conv が主に `D` 幅で動く | **圧縮型なので候補**。ただし CWT と複数周期の 2D map は `T`・scale 数・top-k に依存して重い |
| WPMixer | 各帯域で `Cin` 本の patch activation `[B,Cin,N_i,D]` を処理。末尾 `Cin→Cout` | **概ね系列数に線形**。帯域数・patch 数・batch で activation が膨らむ。`seq_len` 固定 |
| CycleNet | 周期表の重み `W·Cin`、各系列への共有時間 Linear/MLP、末尾 `Cin→Cout` | **比較的単純だが `T` 依存**。Linear 版は各系列に長さ `T` の dense 写像を適用。既知位相と固定 `seq_len` が必要 |
| TimeMixer | CI=False: `Cin→D` の入力畳み込み後は `D` 幅。CI=True: `[B·Cin,T_i,D]` を各尺度・各層で保持し、末尾は `Linear(Cin·D,Cout)` | **CI=False は圧縮型の候補、CI=True は1000系列で重い**。後者は隠れ activation が `Cin·D` 倍で、多尺度分さらに増える |
| FITS | 共有複素 Linear は `K²` パラメータで、適用演算と FFT は系列数に概ね線形。`individual=True` は重みが `Cin·K²`、かつ現実装では系列ごとの Python loop | **`individual=False` を先に検討**。`True` の1000個の小モジュールは Python/launch overhead も大きい。末尾 `Cin→Cout` も加算される |
| Timer-XL | `Cin·N`個のtokenへ密attention、`O(B·(Cin·N)²·D)` | **1000系列の直投入は重い**。`N=8`でも1headのscoreは6400万要素。patch長・系列選択と精度を比較する |
| PatchTST | 各系列独立patch attentionで`O(B·Cin·N²·D)` | **系列数には線形**。ただし1000系列ぶんのactivationとBNが必要。融合は最終channel Linearのみ |
| TQNet | 系列軸attention`O(B·Cin²·T)`、query表`W·Cin` | **1000系列では要注意**。`B=16,heads=4`ならscoreだけでFP32約256 MB |
| TimeXer | 全Cを対象に各々がC個のexogenous tokenへcross-attention | **公開M経路は概ね`Cin²`**。対象選択等は原著の全変数同時処理からの変更になる |
| TimeBridge | 公式既定axial系列間attentionは`R≈√Cin`の行・列へ分割 | full `Cin²`より軽いがpatch×系列activationは必要。`Cin≈1000`で実測必須 |
| iTransformer | 変数tokenのself-attention`O(B·Cin²·D)` | **1000系列では要注意**。`B=16,heads=4`ならscoreだけでFP32約256 MB |

全ブロック共通で出力を dense な `Cin→Cout` にする型では、その射影に `Cin·Cout` 個の重みと、概ね `B·T·Cin·Cout` の積和が要る。`Cin=1000, Cout=64` なら重みは6.4万個だが、`Cout=1000` なら100万個になる。Mamba・WFTNet・TimeMixer(CI=False) は先に `D` へ圧縮するため、この **`Cin→Cout` の末尾射影を必ずしも持たない**。元の入力テンソルだけでも `B=16,T=96,Cin=1000` の FP32 で約6.1 MB、学習では中間 activation と勾配が別に必要。例えば TimeMixer(CI=True) の単一 `[B·Cin,T,D]`（`D=64`）だけで約393 MB になり、多尺度・複数層なら増える。

先に任意の `Linear(Cin,Cmid)` を追加してから各 block へ渡す方法はある。ただしそれは**元の1000系列を1000本として処理する原著設計ではなく、`Cmid` 個の混合特徴を処理する別設計**になる。特に DST-Mamba の「各 channel=交通 node」、CycleNet の変数別周期表、AdaWaveNet の channel attention、ModernTCN の変数間 FFN は解釈が変わる。性能上この圧縮が有利かはデータ依存であり、系列を残す設定と同じ検証 split で比較する。実運用では想定する `B,T,Cin,Cout` と train/inference の両方で peak memory・latency を測り、OOM の有無だけでなく実際の throughput も確認する。

## 下流タスクへの組み込み

以下は全 17 種類の `ts_block.py` に共通する使い方。`encoder` は構築済みの任意のブロックで、出力幅が `feature_dim` と一致するとする。CycleNetとTQNetは位相が既知なら `encoder(x, phase=phase)` とし、`phase:[B]` は各窓先頭の周期位相にする。

```python
import torch
from torch import nn


class PooledTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int,
                 num_outputs: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)        # [B,T,Cin] → [B,T,feature_dim]
        pooled = h.mean(dim=1)     # [B,T,feature_dim] → [B,feature_dim]
        return self.head(pooled)   # [B,num_outputs]
```

`num_outputs=1` の回帰、`num_outputs=K` の K クラス分類などは、head の出力と外側の GT/loss で決める。平均 pooling は一例で、attention pooling などに交換できる。時刻別の回帰・分類なら pooling を外して `h` に時刻別 head を適用する。

## 使い分けと再現性

全ての `ts_block.py` は短い動作確認が可能。WFTNet は公式 CWT に合わせるため `ptwt` と NumPy が必要で、他は原則 PyTorch と `einops` を使う。Mamba/DST-Mamba は CUDA が利用可能かつ `mamba_ssm` があるとき公式 mixer を優先する。ない場合の PyTorch scan は数式どおりでも Python loop なので GPU 長系列の速度要件には向かない。WaveTS は公開実装が依存する wavelet ライブラリを避けて Haar/db1 を実装した。**DST-Mamba・WaveTS・PatchTST・TQNet・TimeXer・TimeBridge・iTransformer は公式の時間方向learned Linear等のため `seq_len` を構築時に固定する。Timer-XLはpatch長の倍数なら可変長。** 入出力の時間長は同じでも、任意Tを同じ重みで扱えるかは別。いずれも原著の固定 horizon 予測 head や公開 checkpoint と互換であるとは主張しない。原著の学習 objective は各 README に別記した。

### 公式 encoder 処理との入出力一致

結論：**予測 head は判定対象外。** 同じ入力・対応する重みで、公式の **encoder 相当部分の出力**を比較する。汎用ブロックの `[B,T,Cout]` にするための最終 readout・幅変換も判定対象外。wavelet を使う3件は、対応する公式ライブラリの係数・逆変換まで数値比較した。

| ブロック | encoder 一致 | 比較した処理／未確認箇所 |
| --- | :---: | --- |
| P-sLSTM | Yes | raw 入力の patch 埋込から公式 xLSTM stack 末尾 `[B·Cin,N,D]` まで一致。公式 `Linear(N·D,H)` は予測 head なので比較しない。 |
| Mamba-1/2/3 | Yes | 3版の mixer 出力が公式参照と一致。公式 `Block` の norm・residual 順序も2層で一致。 |
| DST-Mamba | Yes | 公式の Mamba 依存箇所のみ両側で同じ PyTorch mixer に置換すると、分解・node encoder・trend・RevIN を含む `L=H` の最終出力が一致。公式 fused Mamba kernel 自体は未比較。 |
| WaveTS | Yes | `db1`・`reflect` で公式 `pytorch_wavelets` の DWT/iDWT 係数が偶数・奇数長で一致。同長 `H=0` の公式モデル最終出力も一致。非 Haar wavelet は本版の対象外。 |
| ModernTCN | Yes | raw 入力から stem・段間 downsampling・複数 stage の最終出力 `[B,Cin,D,P]` まで一致。公式の固定 horizon head は比較しない。 |
| AdaWaveNet | Yes | 独立 P/U の逆 lifting、channel encoder、cluster 別 trend を含む同長経路の最終出力が一致。 |
| WFTNet | Yes | 公式指定 `ptwt==0.1.6` の CWT は精度 10。導入した 1.0.1 の `precision=10` が旧版係数と完全一致することを確認し、公式 `Model.forward` の同長経路の最終出力も一致。 |
| WPMixer | Yes | `haar/db1` で公式同梱版・公開版 `pytorch_wavelets` の DWT/iDWT 係数が一致。実ライブラリを使う公式同長 core の最終出力も一致。非 Haar wavelet は本版の対象外。 |
| CycleNet | Yes | 同じ位相を入力したとき、周期復元を含む同長経路の最終出力が一致。 |
| TimeMixer | Yes | raw 入力から尺度別正規化・埋込・PDM の最細尺度 encoder 出力まで一致。公式 FMM は予測 head なので比較しない。 |
| FITS | Yes | 偶数長の周波数処理から最終出力まで一致。奇数長は公式が `T−1` 点を返すため、同長契約の本版は明示的に拒否する。 |
| Timer-XL | Yes | 公式README指定のOpenLTM `Model.forward`から`TimerBlock`出力をhookし、多変量・covariate両maskで一致。公開non-flash経路のみ。 |
| PatchTST | Yes | 公開`PatchTST_backbone`を直接実行し、RevIN→patch→複数層の出力と`H=T`同長出力が一致。 |
| TQNet | Yes | 公開`Model.forward`の入力からMLP residualまでのencoder出力が非ゼロ位相で一致。 |
| TimeXer | Yes | 公開`forecast_multi`の入力から`features='M'` encoder末尾まで一致。marker無し設定。 |
| TimeBridge | Yes | 公開`Model.forward`の入力からIntegrated→Sampling→Cointegratedのencoder末尾まで一致。axial・marker無し設定。 |
| iTransformer | Yes | 公開`Model.forward`の入力から変数token encoder末尾まで一致。marker無し設定。 |

ここでの Yes は表の対応設定での encoder 照合結果であり、予測 head・他の wavelet family・任意設定の一致を意味しない。

