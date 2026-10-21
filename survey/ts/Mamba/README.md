# Mamba survey

指定資料の `main.tex` は **A Survey of Mamba** であり原著ではない。Mamba-1/2 の理論は提供 TeX、[Mamba-1 原著](https://arxiv.org/abs/2312.00752)、[Mamba-2 原著](https://arxiv.org/abs/2405.21060)、Mamba-3 は[原著のテキスト版](https://arxiv.org/html/2603.15569)と照合した。実装は[公式 Mamba リポジトリ](https://github.com/state-spaces/mamba)の `mamba_simple.py`、`mamba2.py` / `mamba2_simple.py`、`mamba3.py`、および公式の scan/step 参照関数に対応する。本ディレクトリは **Mamba-1/2/3 を別々の mixer として実装**する。前版の「Mamba-1 のみ」は誤りだった。

## このディレクトリのブロックの使い方

主用途は **Mamba-1/2/3 のいずれかを系列 encoder として組み込むこと**。`MambaBlock` は `x:[B,T,Cin]→h:[B,T,Cout]` を返すだけで、GT・loss・pooling・task head は持たない。`h.mean(dim=1):[B,Cout]` を系列単位の回帰/分類 head に渡すことも、`h:[B,T,Cout]` に時刻別 head を付けることもできる。ブロックは T を固定せず使えるが、各 forward で state を初期化するため、別々の window 間で状態を引き継ぐ streaming API ではない。

以下の将来予測 `H` 点と MSE は **下流タスクの一例**であり、Mamba mixer の objective ではない。[finetuning_example.py](finetuning_example.py) もその一例を学習するサンプルで、ブロックの利用先を forecasting に限定しない。

## 概要

Mamba は、**入力に応じて状態の保持・書込み・読出しを変える状態空間モデル（SSM）を、長い系列に適用する**系列 backbone である。入力 `u:[B,T,D]` を時刻順に読み、各時刻の表現 `y:[B,T,D]` を返す。`B` は batch、`T` は系列長、`D` はモデル幅。基本の mixer は因果的で、時刻 `t` の出力は `t` 以前だけを見る。Transformer の自己注意とは異なり、過去 token 全体の attention 行列を作らず、固定幅の再帰 state を更新する。

Mamba 自体に「未来 H 点を予測する」という固有 head/loss はない。時系列予測に使う場合は、過去 `X:[B,L,Cin]` の表現に別の forecast head を付け、将来実測 `Y:[B,H,Ctarget]` との MSE を学習する。このディレクトリの `MambaBlock` は Mamba-1/2/3 のいずれかを選ぶ **同長特徴変換器** `[B,T,Cin]→[B,T,Cout]` である。

## 新規性・既存手法との差分

| 比較対象・版 | 何が異なるか | 何を狙うか |
| --- | --- | --- |
| 固定係数 SSM / 畳み込み | Mamba-1 は `Δ_t,B_t,C_t` を入力から作る | 内容に応じて情報を選択・忘却する |
| Transformer attention | 明示的な全時刻間 attention の代わりに state scan | 長さ T に対する計算・state サイズを抑える。ただし表現能力が同じとは限らない |
| Mamba-1 → Mamba-2 | head 分割と SSD/chunked 計算、grouped B/C を導入 | 行列演算に適した scan と計算効率を狙う。単なる kernel 差し替えではない |
| Mamba-2 → Mamba-3 | 入力依存の減衰 A、回転 Q/K、前/現時刻を混ぜる台形更新、任意の MIMO | 状態更新・読み出しの表現を拡張する。Mamba-2 の重みとは非互換 |

Mamba-3 は「Mamba-1/2 の上に層を追加する」版ではなく **代替 mixer** である。以下の図でも、版ごとの内部経路は並列の選択肢として示す。

## モデルアーキテクチャ全体像

```text
このディレクトリの再利用ブロック
入力 X [B,T,Cin]
  │ Linear(Cin,D)
  ▼
h [B,T,D]
  │  以下を depth 回（各層は別パラメータ）
  ├──────── residual ───────────────────────────────────┐
  └─ LayerNorm ─> 選択した版の Mamba mixer [B,T,D] ──> + ┘
  │ Linear(D,Cout)
  ▼
出力 [B,T,Cout]

選択した mixer の内部（どれか一つを使用）
  Mamba-1: [B,T,D]
    ├─ in_proj → signal/gate 各[B,T,I]、I=expand·D
    ├─ signal → causal DWConv → Δ[B,T,I], B/C[B,T,N]
    ├─ state scan: s_t[B,I,N] = exp(Δ_t A)·s_(t-1) + Δ_t B_t signal_t
    └─ C_t による読出し + D skip → gate → out_proj → [B,T,D]

  Mamba-2: [B,T,D]
    ├─ in_proj → z[B,T,I], x/B/C[B,T,I+2GN], Δ[B,T,J]
    ├─ x/B/C をまとめて causal DWConv → x[B,T,J,P], B/C[B,T,G,N]
    ├─ head ごとの SSD/recurrent state [B,J,P,N] を更新
    └─ readout + D skip → gated RMSNorm → out_proj → [B,T,D]

  Mamba-3: [B,T,D]
    ├─ in_proj → x/z[B,T,J,P], B/C[B,T,R,G,N], Δ/A/trap/angle
    ├─ B/C を正規化・head へ配布し、角度を累積 → 回転 Q/K
    ├─ 前時刻と現時刻の K⊗V を台形係数で混ぜ、state[B,J,P,N] 更新
    └─ state×Q + D skip → gate → rank R を統合 → out_proj → [B,T,D]
```

`I` は inner 幅、`J=I/P` は head 数、`P` は head 幅、`G` は B/C group 数、`N` は state 幅、`R` は MIMO rank（SISO は `R=1`）。図の Mamba-3 は **公式経路に causal DWConv がない**点にも注意。Mamba-1/2/3 の内部式・速度上の違いは以下の各節で展開する。図の外側の pre-norm residual と入出力 projection は本ディレクトリの再利用 adapter であり、各論文の裸 mixer そのものではない。

## 先に結論: 3 バージョンは何が違うか

| | Mamba-1 | Mamba-2 | Mamba-3 |
| --- | --- | --- | --- |
| 内部 state | channel ごと `[B,Hinner,N]` | head ごと `[B,H,P,N]` | head ごと `[B,H,P,N]` |
| 選択係数 | token 依存 `Δ,B,C`、固定 `A` | head ごと scalar `A,Δ`、token 依存 group `B,C` | token 依存 `A,Δ,trap,angle,B,C` |
| 入力前処理 | `x` に causal DWConv | `x/B/C` をまとめて causal DWConv | 公式 `Mamba3` に DWConv **なし** |
| scan 入力項 | `Δ B x` | `Δ B x` | 前時刻と現時刻の `KV` を台形係数で混ぜる |
| 特有の機構 | selective scan | SSD/chunked scan、head 化、後段 gated RMSNorm | cumulative data-dependent RoPE、exponential-trapezoidal、SISO/MIMO |
| 本実装 | `TorchMamba` | `TorchMamba2` | `TorchMamba3` (`is_mimo` も選択可) |

ここで `Hinner=expand·d_model`、`H=Hinner/P`、`P=headdim`、`N=d_state`。3 バージョンは同じ重み形式ではなく、checkpoint 互換ではない。いずれもこのディレクトリの `MambaBlock` では `input:[B,T,Cin] → output:[B,T,Cout]` であり、**T が時間、Cin/Cout が同時刻の特徴軸**である。

再利用するときは **`ts_block.py` と `mamba_versions.py` の2ファイルを同じディレクトリへコピー**する。前者は版を選ぶ共通 `MambaBlock` と `[B,T,Cin]→[B,T,Cout]` アダプタだけ、後者に **Mamba-1/2/3 の全 mixer** と補助関数をまとめた。通常の依存は PyTorch と `einops` で、純 PyTorch scan backend に外部 `mamba_ssm` は不要。公式参照との比較スクリプト `verify_official.py` はテスト時だけ必要で、公式 checkout を参照する。

## 核心: input-dependent selective SSM

固定 SSM は各時刻で同じ動力学を使う。Mamba-1 は入力 token `u_t` から `Δ_t,B_t,C_t` を生成するので、保持/忘却/読み出しが内容依存になる。内側特徴数 `H=expand·D`、state 次元 `N=d_state` とすると公式コードの経路は次の通り。

| 処理 | 形状 | 役割 |
| --- | --- | --- |
| 入力・分岐 | `x:[B,T,D] → (u,z):各[B,T,H]` | `in_proj`、z は出力 gate |
| 短距離混合 | `u:[B,T,H] → [B,T,H]` | causal depthwise Conv1d + SiLU |
| 選択パラメータ | `u → Δ:[B,T,H], B_t,C_t:[B,T,N]` | `x_proj` と `dt_proj` |
| state | `s_t:[B,H,N]` | scan、時刻ごとに更新 |
| readout | `y_t:[B,H] → [B,D]` | `C_t` で state 読み出し、D skip、SiLU(z)、out_proj |

公式実装は `A=-exp(A_log)∈R^{H×N}`、`Δ=softplus(dt_proj(...))>0` を使う。各時刻の再帰は `s_t=exp(Δ_t A)⊙s_{t-1}+Δ_t B_t⊙u_t`、`y_t=C_t·s_t+D⊙u_t`。ここで B/C は H へ broadcast される。厳密な連続系 ZOH 入力項ではなく、**公式 Mamba-1 の B 項は ΔB の Euler 型**である。最後の gate と線形投影まで含めて block 出力になる。因果的なので位置 t の出力は `x_{≤t}` のみ依存する。

## なぜ速いか / 何が速くないか

state サイズは系列長 T に依存しない。Mamba-1 の scan は概ね `O(B·T·Hinner·N)`、Mamba-2/3 の再帰を素朴に評価すると `O(B·T·H·P·N)`。「線形時間」は Python `for` が高速という意味ではない。公式は fused selective/SSD/chunked kernel 等で GPU メモリ転送を抑える。[ts_block.py](ts_block.py) は `backend="auto"` で CUDA と公式 package が使える場合にその版の公式 mixer、そうでなければ**同じ版**の純 PyTorch 再帰を使う。fallback は数式検証と小規模学習向けで、長系列 GPU の throughput を求める場合は公式 kernel が必要。初期化・数値細部や checkpoint の直接互換性は保証しない。

## 下流 forecasting の一例と objective

Mamba 自体は系列変換器であり、特定の forecasting loss を定義しない。`MambaBlock(in_dim,out_dim)` は `[B,T,Cin]→[B,T,Cout]`、入力線形→pre-norm Mamba residual stack→出力線形。予測する場合は観測窓 `X:[B,L,Cin]` の最終表現 `h_L:[B,D]` から別 head が `Ŷ:[B,H,Ctarget]` を出し、将来の実測 `Y` に対し MSE を取る。[finetuning_example.py](finetuning_example.py) が DataFrame からこの経路を実行する。分類や異常検出なら head と GT/loss を変更する。未来の説明変数を入力に混ぜると leakage なので、利用可能時刻を明示する。

```python
from ts_block import MambaBlock
block = MambaBlock(8, 32, d_model=64, version=3, backend="auto")
h = block(history)  # [B,L,8] -> [B,L,32]
```

## 注意

Mamba は時間方向に順序を持つ。時系列で双方向化すれば将来側の情報を使うため、逐次予測にはそのまま使えない。提供されたサーベイ中の速度比や複雑度表現は公式特定 GPU/設定の保証値ではない。

## 論文サーベイと今回の実装対象を分ける

提供された *A Survey of Mamba* は Mamba-1/2、block design、scan direction、memory 管理、複数分野への応用と限界を整理した二次資料で、Mamba-3 は出版時期のため含まれない。本実装は公式の **Mamba-1/2/3 本流の mixer** であり、サーベイに列挙された全派生モデルの実装ではない。Mambaformer や双方向 Mamba は別 architecture である。

### SSM から selective SSM までの因果関係

連続時間の線形 SSM は `dh/dt=A h+B x`、`y=C h+D x`。一定の `A,B,C` を離散化すれば、固定した時間不変 kernel と畳み込みとしても計算できる。しかし固定係数のままでは、各時刻の内容に応じて「今の観測を強く記憶する」「以前の情報を忘れる」を柔軟に切り替えにくい。Mamba は畳み込み可能な時間不変性を一部捨て、入力 `u_t` から `Δ_t,B_t,C_t` を予測する。

```text
u_t = SiLU(depthwise_causal_conv(in_proj(x)_signal)_t)  [B,H]
Δ_t = softplus(dt_proj(dt_rank(u_t)))                 [B,H]
B_t = B_proj(u_t), C_t = C_proj(u_t)                 各[B,N]
A = -exp(A_log)                                      [H,N]
s_t = exp(Δ_t A) * s_{t-1} + (Δ_t B_t) * u_t        [B,H,N]
r_t = sum_N(C_t * s_t) + D * u_t                     [B,H]
y_t = out_proj(r_t * SiLU(z_t))                     [B,D]
```

`H=expand·D` は内部 channel、`N=d_state` は各 channel に紐づく state 成分数。元入力 `x_t:[B,D]` と SSM state `s_t:[B,H,N]` は異なる物体である。出力 `y_t` は連続値ベクトルであり、token ID を予測する機構ではない。時系列の将来値にするには別 head が必要。

### 何が「selective」か

`Δ_t` が大きいと負の `A` による前 state の減衰が強くなり、同じ時刻の input 注入量も変わる。`B_t` は何を state へ書くか、`C_t` はどの state を読むかを決める。これらは毎時刻の `u_t` から作られる。一方、`A_log` と `D` は系列内で共有される学習 parameter であり、`B,C,Δ` と同様に毎時刻独立に生成されるわけではない。ここを区別すると数式と公式コードの対応が追いやすい。

公式 Mamba-1 の `x_proj` は `[B·T,H]→[B·T,r+2N]` で、`r=ceil(D/16)` が dt の低 rank 幅。dt だけを `dt_proj:[B·T,r]→[B·T,H]` で拡張する。全 `H×N` の B/C を直接生成せず、B/C は N 次元を H channel に broadcast するため、parameter と activation を抑える。

### 学習 scan と 1-step inference は同じ再帰か

数学上は同じ state 更新を、学習時は全 T 点に対する scan、自己回帰的・オンライン推論時は現在 state と新観測 1 点で実行する。公式 `mamba_simple.py` には全系列 `forward` と `step`、`conv_state:[B,H,d_conv]` と `ssm_state:[B,H,N]` の cache がある。次の時点の計算には T 個の過去 token 全部を再入力する必要はない。ただし、固定窓 forecaster のように毎回別の重複窓を `MambaBlock` に渡す使い方では、本ディレクトリの wrapper は毎回 state を初期化する。**cache を跨いで使う streaming API はこの wrapper に実装していない**。オンライン低遅延を厳密に要する場合は公式の stateful `step` を使う。

### 「線形時間」と wall-clock の区別

一層の理論計算は T に概ね線形だが、state tensor は `B×H×N` で、training の naive autograd は T step の計算グラフを保持する。本コードの pure PyTorch fallback は可読性と依存削減のため `for t in range(T)` で scan し、全 step の autograd node を作る。公式は hardware-aware selective scan と backward recomputation で中間 state の保存・転送を減らす。したがって同じ式でも実効 GPU 速度と peak memory は一致しない。

ほかに `d_conv` の深さ方向畳み込み、入力/出力線形、`x_proj`、`dt_proj` の演算もある。極端に短い系列では launch overhead や線形層が支配的になり得る。「attention より常に速い」という主張はできず、対象 B/T/D/device で測るべきである。

## 時系列での利用パターンを区別する

### A. 因果的履歴 encoder + 一括 horizon head

`X:[B,L,Cin]` を `MambaBlock` へ渡し `h:[B,L,D]` を得る。`h[:,-1]:[B,D]` を `Linear(D,H·Ctarget)` に通し `Ŷ:[B,H,Ctarget]` へ reshape、将来観測 `Y` との MSE を取る。[finetuning_example.py](finetuning_example.py) はこの形。horizon 内の各将来時刻は head から同時に出るので、**将来を Mamba で H 回 recurrent 生成する方式ではない**。この head は軽いが、H が長いと `D→H·Ctarget` の線形層が大きくなる。

### B. 時点ごとの表現・分類・異常検知

`h:[B,T,D]` に `Linear(D,K)` を時点ごとに適用する。GT が時点ラベル `[B,T]` なら CE、連続 target `[B,T,Ctarget]` なら MSE 等を選ぶ。loss は Mamba ではなく task が定義する。因果 Mamba では `h_t` が未来 `x_{>t}` を見ないため online 推定に使える。一方、欠損補完のように両側文脈が必要なら、逆向き scan を別途加えるなど architecture 自体を変更する。

### C. 多変量時系列での軸の意味

本 `MambaBlock` が scan するのは **T 軸**で、同時刻の複数変数は `Cin` 次元に詰めて最初の線形層で混合する。DST-Mamba のように変数/node 軸を scan するものではない。複数銘柄を channel に並べる場合、銘柄順を入れ替えても入力線形層の重みは自動で入れ替わらない。固定 schema を保つ必要がある。

## Mamba-2: SSD と head 化は単なる Mamba-1 の高速版ではない

公式 `Mamba2Simple.forward` の処理を、`H=expand·D/P`、`G=ngroups`、`P=headdim`、`N=d_state` として追う。最初の `in_proj` は `u:[B,T,D]` を `z:[B,T,H·P]`, `xBC:[B,T,H·P+2G·N]`, `dt_raw:[B,T,H]` に分割する。`xBC` **全体**を causal depthwise Conv1d + SiLU に通して、`x:[B,T,H,P]`、`B,C:[B,T,G,N]` を得る。`B,C` は各 group 内の head に broadcast する。`dt=softplus(dt_raw+dt_bias):[B,T,H]`、`A=-exp(A_log):[H]` は head scalar。

時刻 t の再帰は `S_t:[B,H,P,N]` に対して

```text
S_t = exp(dt_t A) S_{t-1} + dt_t · x_t ⊗ B_t
r_t = contraction_N(S_t,C_t) + D · x_t       [B,H,P]
g_t = RMSNorm(r_t · SiLU(z_t))               [B,H·P]
out_t = out_proj(g_t)                        [B,D]
```

ここで `⊗` は `[P]` と `[N]` の外積。Mamba-1 の A は `[Hinner,N]` と channel/state ごとに異なるが、Mamba-2 は `[H]` scalar である。この制約により SSM と構造化 causal attention の対応（SSD）が使え、学習時に chunk 内の行列演算と chunk 間の state 伝播へ分解できる。**再帰と SSD は同じ演算の二つの計算法**であり、PyTorch 版は再帰で実装した。[公式 Mamba-2 論文](https://arxiv.org/html/2405.21060)と `ssd_minimal.py` の独立参照関数で数値一致を検査する。

Mamba-2 の `norm_before_gate=False` は `RMSNorm(r·SiLU(z))` を意味する。Mamba-3 の任意 `is_outproj_norm=True` とは gate/Norm の順番が異なる。`learnable_init_states` と `dt_limit` は `TorchMamba2` で使用可能。`MambaBlock` の便宜的な `d_state=16`、`d_model=64` 等は論文で性能を報告した規模の再現設定ではない。

## Mamba-3: 推論を意識した 3 つの変更

[Mamba-3 原著](https://arxiv.org/html/2603.15569)の主要な変更は、(1) exponential-trapezoidal discretization、(2) 複素状態と等価な data-dependent RoPE、(3) SISO から MIMO への拡張。公式 `mamba3.py` は Mamba-2 と異なり causal DWConv を置かず、入力射影で `z,x,B,C,dd_dt,dd_A,trap,angle` を直接生成する。`x,z:[B,T,H,P]`、`B,C:[B,T,R,G,N]`、`dd_dt,dd_A,trap:[B,T,H]`、`angle:[B,T,K]`、`K=floor(rope_fraction·N/2)`。`R=1` は SISO、`R=mimo_rank` は MIMO。B/C は RMSNorm 後に learned bias を足す。`dt=softplus(dd_dt+dt_bias)`、`A_t=-max(heavy_tail(dd_A),A_floor)`。

時刻 t で cumulative angle `θ_t=θ_{t-1}+π·tanh(angle_t)·dt_t` を作り、`Q_t=C_t+b_C`、`K_t=B_t+b_B` の state 次元の偶奇ペアに RoPE を施す。これは固定位置埋め込みではなく、**入力と dt に依存する回転**。`α_t=exp(A_t dt_t)`、`β_t=(1-sigmoid(trap_t))dt_t α_t`、`γ_t=sigmoid(trap_t)dt_t` とすると、公式 step 参照の状態更新は

```text
S_t = α_t S_{t-1} + β_t (K_{t-1} ⊗ V_{t-1}) + γ_t (K_t ⊗ V_t)   [B,H,P,N]
r_t = S_t Q_t + D V_t                                          [B,H,R,P]
```

SISO (`R=1`) では `V_t=x_t:[B,H,P]`。MIMO では `mimo_x:[H,R,P]` で R 個の V を作り、R 個の K/V 外積を **合算して同じサイズ `[B,H,P,N]` の state** を更新する。読み出しは R 個の Q で行い、`mimo_o` で R を縮約する。`mimo_z` は gate を R に拡張する。したがって MIMO は R 倍の state を持つのではない。公式論文がいう「decode の arithmetic intensity を高める」はこの行列積化を指す。ただし論文の速度・精度結果は言語モデル/対象 GPU の条件付きで、時系列タスクでの向上保証ではない。

公式の default は `is_mimo=False`, `is_outproj_norm=False`。前者を有効化すると MIMO、後者を有効化すると head-wise RMSNorm **→** `SiLU(z)` gate の経路を選ぶ。`TorchMamba3` は両方実装。公式 MIMO は TileLang、step は CuteDSL などの専用 kernel を使うが、portable 版は time loop と PyTorch contraction のため大規模 GPU 推論の速度目標を満たさない。`backend="official"` は必要な kernel が実際に利用できる環境で選ぶ。

## 論文サーベイに挙がる限界と実務上の限界

- 固定サイズ state は過去の任意区間を完全に保持するメモリではない。サーベイは copying/state-tracking 型の課題での限界を紹介する。長い金融履歴から稀な出来事を正確に再検索したいなら、attention や明示的メモリとの比較が必要。
- 因果性が長所である一方、系列全体が既知の補完・分類では双方向文脈を活かせない。双方向化は用途に応じて、future leakage を確認して行う。
- `A=-exp(A_log)` は連続系として負で安定な形だが、任意の深い residual stack が常に数値安定という意味ではない。activation・勾配・混合精度を監視する。
- `backend="auto"` は CUDA が利用可能かつ公式 package を import できる場合に公式 mixer を優先する。ただし package が入っているだけで kernel 構築条件が揃うとは限らない。portable 動作を確実にしたいテストでは `backend="torch"` を指定する。

## 公式コードとの対応と検証境界

| 版 | 公式 forward / 独立参照 | このディレクトリ |
| --- | --- | --- |
| 1 | `mamba_simple.py::Mamba.forward` / `selective_scan_interface.py::selective_scan_ref` | `mamba_versions.py::TorchMamba` |
| 2 | `mamba2.py::Mamba2.forward`, `mamba2_simple.py::Mamba2Simple.forward` / `ssd_minimal.py::ssd_minimal_discrete` | `mamba_versions.py::TorchMamba2` |
| 3 SISO | `mamba3.py::Mamba3.forward` / `test_mamba3_siso.py::mamba3_siso_step_ref` | `mamba_versions.py::TorchMamba3(is_mimo=False)` |
| 3 MIMO | 同上 / `test_mamba3_mimo.py::mamba3_MIMO_step_ref` | `mamba_versions.py::TorchMamba3(is_mimo=True)` |

`verify_official.py --official-repo <公式 checkout>` は、公式ソースの**独立した参照関数**を読み込み、Mamba-1/2/3 の mixer について出力投影後までの出力と入力勾配を比較する。比較は FP32 の短系列で出力 `rtol=atol=2e-5`、勾配 `rtol=3e-4, atol=3e-5`。汎用 `MambaBlock` の前後の入出力射影・residual は原著の mixer 外に追加した adapter であり、公式 mixer と同一モデルの最終出力だとは主張しない。公式 fused CUDA/Triton/TileLang kernel との parity や BF16、長系列、公式学習済み checkpoint の互換性はここでは検証していない。production で使う前に対象 device・dtype・系列長でもベンチマークと検証を行うこと。`backend="auto"` は CUDA 時に公式 package を import できれば公式 mixer を試すが、kernel 実行可能性まで事前保証しない。

`MambaBlock` は公式 `Block` と同じ pre-norm residual 順序を使う。`input: Linear(Cin,D) → [LayerNorm(D) → mixer(D) + residual] × depth → Linear(D,Cout)`。公式 mixer は `[B,T,D]→[B,T,D]` 固定なので、任意の `Cin,Cout` を選ぶには外側の幅変換が必要。[verify_official.py](verify_official.py) は前後に同じ幅変換を置いた公式 `Block` 2層と最終出力を比較する。入出力 `[B,T,Cin]→[B,T,Cout]` のため、Mamba 原著の task head/loss は含めない。純 PyTorch 版は全系列 forward を実装するが、**state を引き継ぐ streaming step API と fused GPU kernel は実装しない**。これらは公式側の `step`/cache と専用 kernel を利用する。

## 一次資料・対象範囲

- [提供された論文: A Survey of Mamba](https://arxiv.org/abs/2408.01129)（本調査では提供 TeX を読解）
- [Mamba-1 原著: Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-2 原著 / HTML](https://arxiv.org/html/2405.21060)
- [Mamba-3 原著 / HTML](https://arxiv.org/html/2603.15569)
- [公式 Mamba 実装](https://github.com/state-spaces/mamba)

## 動作確認と DataFrame 学習の入口

このディレクトリで `python ts_block.py` を実行すると 3 版の shape/backward を検証する。`python verify_official.py --official-repo <公式 checkout>` は版ごとの独立参照と数値比較する。`python ../validate_blocks.py` は5モデル横断の確認で、速度比較ではない。

CSV を用意する場合は `series_id,timestamp` と特徴列・target 列が必要。例えば次のコマンドは過去64点から次8点の実測 `target` を予測する。

```bash
python finetuning_example.py data.csv --features feature_a feature_b --targets target --lookback 64 --horizon 8 --epochs 10 --version 3 --backend torch
```

既に DataFrame `df` があるなら `from finetuning_example import fit; model, info = fit(df, ["feature_a", "feature_b"], ["target"], version=3, backend="torch")`。Mamba-3 MIMO は `is_mimo=True`、CLI は `--mimo`。返す `info` には train 行だけから求めた feature mean/std と最良の validation MSE が入る。推論時は同じ列順・同じ正規化統計で過去窓を作る。原著の benchmark 成績を再現するには入力長、head、split、正規化、optimizer、使用 kernel を揃える必要がある。

## 再利用ブロックの処理と下流タスク

次は `MambaBlock` の処理。Mamba-1/2/3 mixer 自体には forecast・分類などの固定 head はなく、ブロックは同長特徴を返した時点で終了する。

```text
x                                      [B,T,Cin]
  -> Linear(Cin,D)                     [B,T,D]
  -> LayerNorm + 選択した版の Mamba + residual
                                       [B,T,D]
  -> Linear(D,Cout)                    h:[B,T,Cout]
  -> ここで MambaBlock.forward は終了

任意の下流処理:
  pool(h, time=T)                      [B,Cout]
  regression/classification head      [B,K]
  GT・loss はタスク側で定義
```

pooling は必須ではなく、`h:[B,T,Cout]` に時刻別 head を直接適用してもよい。[finetuning_example.py](finetuning_example.py) は任意の下流例として `h` の最終時刻から将来 H 点を一括回帰し、実測との MSE を学習する。これは Mamba 固有の objective ではなく、公式 Mamba の stateful `step` cache を使う逐次生成とも別の設計である。
