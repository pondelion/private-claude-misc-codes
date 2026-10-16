# Context Language Models (CLMs) Understanding - 簡略化疑似コード集

**Context Language Models: context as a file, managed natively by the LM itself**
の理解を目的とした簡略化疑似コード集です。

論文: [arXiv:2609.37725](https://arxiv.org/abs/2609.37725)
公式コード: ローカルクローン `gh/context-language-models` ([github.com/facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models))

## 📋 目次

- [概要](#概要)
- [通常のエージェントハーネスとの差分 (新規性はどこにあるか)](#通常のエージェントハーネスとの差分-新規性はどこにあるか)
- [公開物の実態 (重要)](#公開物の実態-重要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [CLMの主要イノベーション](#clmの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [FAQ](#faq)

---

## 概要

**CLM は何なのか:** モデルのアーキテクチャではなく、**LLMエージェントの
「コンテキストの管理を誰がやるか」を根本から変えるハーネス (エージェント実行
フレームワーク) の設計**である。長時間 (何百〜何千ターン、最大24時間) 動く
コーディングエージェントやDeep Researchエージェントは、会話が進むにつれて
コンテキストウィンドウを使い切ってしまう。通常のハーネスはこれに対して、
「古いやり取りを要約する」「ツール結果をファイルに退避する」といった固定の
圧縮/退避戦略をハーネス側のコードとして人間が事前に書き、LM自身はそれを
使うだけだった。CLMはこの役割を丸ごとLM自身に渡す: LMの生きたコンテキストを
ディスク上の1個のファイルとしてミラーし、LMに普通のBashコマンド
(`grep`/`sed`/`mv`等、何でも) でそのファイルを自由に読み書きさせる。つまり
「コンテキストをどう圧縮・検索・整理するか」というメタな判断そのものを、
事前に決めたツールではなく、LMが状況に応じて書くコードとして創発的に
実行させる、という発想である。

**どんな場合に使えるか:** 単発のQ&Aではなく、ターミナルでの長時間コーディング・
Deep Research・数理最適化・複数リポジトリをまたぐ最適化タスクなど、
「コンテキストウィンドウに収まらない量のやり取りを、何らかの形で圧縮・取捨選択
しながら進める必要がある」長時間エージェントタスク全般が対象になる。既存の
任意のLM (Qwen3.5/3.6, GPT-5.x, Claude 4.6等) にzero-shotで適用できるため、
モデルを再学習せずにハーネスだけ入れ替えて効果を確認できる点も実用上の
ポイントになる。

**CLM (Context Language Model) の特徴:**
- **Context as a file**: LM の生きた会話コンテキストをディスク上の1ファイルとして
  ミラーし、LM に通常の Bash コマンドで自由に読み書きさせる。圧縮・退避・検索・
  部分編集など、既存ハーネスが固定ツールとして事前定義していた操作を、LM自身が
  自由形式のコードとして創発的に実行する
- **Zero-shot で既存の context-management 戦略を上回る**: BrowseComp-Plus で
  11.4%高い精度・21.5%少ないFLOPs、12時間のEdgeBenchで5%高いスコア・59%少ない
  FLOPs、24時間のマルチエージェントタスクで同コストで65%大きい改善
- **In-context learning**: 自然言語の指示だけでcontext管理戦略をその場で変更
  (steering)、またGEPA系のテキスト進化ループでスキル文書を最適化 (evolving)
- **Reinforcement learning**: 「成功ゲート付き効率アドバンテージ」を使う
  stepwise GRPO で、Qwen3.5-9BのBrowseComp-Plus性能を12%少ないFLOPsで47.6%改善
- **Suffix Cache Reuse (SCR)**: コンテキスト編集後も生存した接尾辞トークンのKV
  キャッシュを再利用するSGLangパッチ。標準サービングの65.0%のFLOPsで同等性能

**タスク:**
- 長時間 (数百〜数千ターン、最大24時間) のエージェントタスク全般
  (ターミナルコーディング・Deep Research・数理最適化・マルチリポジトリ最適化)

**ベースモデル:** 既存の任意のLM (Qwen3.6-27B, Qwen3.5-9B, GPT-5.4/5.6, Claude 4.6等)。
CLM はモデルのアーキテクチャを一切変更しない。変更するのはハーネス側の
「コンテキストの扱い方」であり、既存モデルにzero-shotで適用できる点が特徴。

---

## 通常のエージェントハーネスとの差分 (新規性はどこにあるか)

### 変わらない部分

LM自体のアーキテクチャ・重みは一切変更しない。**通常のエージェントハーネスと
同じ「1つのBashツールを持つReActループ」がベース**であり、LMへの入力は
system+user+assistant+toolの通常のチャットメッセージ列のまま。

### 変わる部分は2段階ある

**① Context as a file: 「何を・いつ圧縮するか」をLM自身に委ねる**

既存のハーネス (Codex style summarization, Context Folding, Self-Compact,
ACM 等) は、「いつ (固定の長さで)」「どう (要約/オフロード/取得等の固定アクション)」
コンテキストを変えるかを**人間がハーネスに事前定義**する。CLMはこの制約を外し、
LMの生きたコンテキストをそのまま1枚の編集可能ファイルとしてミラーし、LMが
`sed`/`python3`/`cat` 等、使いたい任意のBashコマンドで自由に書き換えられるようにする:

```
標準LM:       c_{t+1} = c_t ⊕ f_θ^LM(c_t)             (追記のみ)
CLM:          c_{t+1} = f_θ^CLM(c_t)                   (任意の変換)
```

これは「モデルの新しい能力」ではなく「ハーネスがモデルに与える自由度の拡張」であり、
既存のLMに**zero-shotでそのまま適用できる** (論文の最大の主張の一つ)。

**② 学習層: ICLとRLでcontext管理戦略そのものを最適化する**

①の自由度があるからこそ、「良いcontext管理戦略」を学習対象にできる:
- **In-context**: 自然言語の一文を指示に追加するだけで戦略を変えられる (steering)。
  さらにGEPA系のテキスト進化ループで、ロールアウトの対比ノートから候補戦略を
  生成・評価・選択するループを回せる (evolving, `skill_evolution.py`)
- **パラメトリック**: 通常のGRPOは「成果」だけを報酬にするが、CLMは
  「コンテキスト編集ターンのトークンだけ」に効く効率チャンネルを追加した
  dual-channel advantageで、精度を落とさず効率の良い編集を強化学習できる
  (`rl_loss.py`)

まとめると、**①が「ゼロショットでの構造的な新規性」、②が「そのうえに乗る学習可能性」**
であり、モデル自体のアーキテクチャ変更は一切ない代わりに、「ハーネスの自由度 +
その自由度を使った学習」という2層構造になっている点が本手法の核心である。

### 学習・推論の仕組み (特別な圧縮モジュールではなく通常のBashツール呼び出し)

アーキテクチャ上、コンテキストを圧縮する専用モジュールは存在しない。CLMは
「`bash` ツールを呼び、たまたまそのコマンドがミラーファイルを書き換える」だけ
であり、ハーネス側がそれを検知して会話メッセージ列に反映する。

- **推論時**: ハーネスは毎ターン、現在の会話をミラーファイルに書き出し (`render_editable`)、
  LMが返したBashコマンドを実行し、ファイルが変化していたら読み戻して
  (`parse_back`) 合法なメッセージ列を再構成する。成長する編集は「fit/shrink
  ゲート」で拒否されうる (`edit_gate.py` / `context_env.py`)
- **学習時 (RL)**: 「編集ターンかどうか」のトークン単位のタグ (`role_mask`) を
  ロールアウト時に記録しておき、学習ステップでその部分だけに効率アドバンテージを
  加算する (`rl_loss.py`)。これはアーキテクチャやlossの形自体を変えるのではなく、
  **advantageの値をどこに配るか** を変えるだけ
- **学習時 (ICL)**: スキル文書 (自然言語の指示) をテキストとして進化させるだけで、
  モデルの重みは一切変えない (`skill_evolution.py`)

---

## 公開物の実態 (重要)

本サーベイを作る過程で、ローカルにクローンされた公式リポジトリ
(`gh/context-language-models`) を実際に調べ、以下を確認した。

| コンポーネント | 公開状況 | 確認した事実 |
|---|---|---|
| `clm/clm_harness/context_env/`, `context_utils/context_string.py` | **フルコード公開** | `render_editable`/`parse_back`/`edit_gate` は "stdlib onlyでユニットテスト可能" と明記された依存ゼロの純粋関数。本サーベイの `context_env.py` はこれをほぼそのまま移植している |
| `clm/clm_harness/clm_agent/harness.py`, `utils/budget.py` | **フルコード公開** | `ClmAgent.run()` のメインループ・`BudgetController` (予算ナッジ/ロールバック/最終ターン) は実運用コードがそのまま公開されている。本サーベイの `main_flow.py` はHarborサンドボックス依存を取り除いた簡略版 |
| `clm/clm_icl/` (evolve.py, propose.py, note.py, tasks.py) | **フルコード公開** | `score()`/`diff_se()`/`gate()`/`pick()`/`pareto()` は外部依存のない純粋関数で、`skill_evolution.py` にそのまま移植している |
| `clm/clm_rl/` | **学習コードは非公開** | `clm_rl/README.md`: "RL training for CLM is not part of this repository." 公開されているのは外部フレームワーク (ProRL-Agent-Server / Slime, いずれも別リポジトリ) に当てる**約70行のパッチ**のみ (`patches/0001-slime_bridge-dual-channel-advantage.patch`)。このパッチに書かれたアルゴリズムをそのまま `rl_loss.py` としてスタンドアロン実装した |
| `clm/clm_harness/flops_metrics/kv_cache_flops.py` | **フルコード公開** | 本番用の `kv_cache_flops.py` (vLLM/SGLangの実運用ログを食わせる版) が公開されている。`prefix_reuse_flops.py` は同じ論文Appendix Bの数式を、ログ無しでも動く簡略版として再実装 |
| `suffix_cache_reuse/` | **フルコード公開 (SGLangパッチ)** | `suffix_cache_reuse/overlay.py` 等、SGLang 0.5.16 に対するモンキーパッチとして公開されている。RadixTree/GPU側のKVキャッシュスロット操作に深く依存し、ピン止めされた特定バージョンのSGLang無しには動作しないため、本サーベイでは**再実装せず**、アルゴリズム (K個の生存スパンを再配置・RoPE再回転・線形注意層の状態スナップショット) をREADMEで解説するのみとした |

**結論:** ハーネス本体 (context-as-a-file) とICL進化ループは公開コードがそのまま
動く設計になっており、本サーベイはそれをほぼ忠実に移植した。一方、RL学習コードは
非公開で、公開されているのはアルゴリズムを示す小さなパッチのみのため、
`rl_loss.py`/`finetune_example.py` はそのパッチの記述に基づく独立した再実装である。

---

## アーキテクチャ全体像

```
各ターン:
┌─────────────────────────────────────────────────────────┐
│ 1. ミラー書き込み (context_env.py: render_editable)        │
│    会話メッセージ列 -> "[[CTX_TURN n role=...]]" 形式の     │
│    1枚のテキストファイルへ (保護された system+task は除く)   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. LM が Bash コマンドを1つ発行 (通常のツール呼び出し)        │
│    - 何もミラーに触れないコマンド: 普通のタスク進行           │
│    - ミラーを書き換えるコマンド: コンテキスト管理 (圧縮/退避等) │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 実行 (実サブプロセス) -> ミラー読み戻し                  │
│    変化していれば parse_back で合法なメッセージ列に再構成     │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Edit Gate (edit_gate.py)                               │
│    編集後にコンテキストが「成長」していたら、予算に収まる場合  │
│    のみ許可 (fit) / 常に拒否 (shrink)。拒否時はLMに通知      │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Budget Controller (main_flow.py)                        │
│    予算の25/50/75%でナッジ、超過で最終ターン、再超過で停止    │
│    純粋な圧縮ターン(タスク出力なし)はステップ消費が無償        │
└─────────────────────────────────────────────────────────┘
```

学習層は上記ループの「外側」に乗る:
- **ICL**: 複数ロールアウトの結果から対比ノートを作り、スキル文書を進化 (`skill_evolution.py`)
- **RL**: 編集ターンのトークンに効率アドバンテージを配って stepwise GRPO (`rl_loss.py`)

---

## ファイル構成

### 1. [context_env.py](context_env.py)
**モデルブロック: Context-as-a-File の中核機構**

- `render_editable()` / `parse_back()`: 会話メッセージ列 <-> 編集可能文字列の
  ラウンドトリップ (公式 `context_string.py` をほぼそのまま移植)
- `may_grow()` / `reject_rule()` / `rejected_note()`: fit/shrink の Edit Gate
  (公式 `edit_gate.py` を移植)
- `ContextEnv`: 実際のローカルファイル + 実サブプロセス (`subprocess.run`) で
  「ミラー書き込み→Bash実行→読み戻し→ゲート判定」を実行する
- `example_usage()`: 実際にPythonヘッダ-docのBashコマンドでコンテキストを圧縮
  させ、成長する編集がゲートで拒否されることを確認する

### 2. [prefix_reuse_flops.py](prefix_reuse_flops.py)
**効率指標: Prefix-Reuse FLOPs (論文 Eq.3, Appendix B)**

- `c_token_and_c_attn()`: モデル幾何 (層数・隠れ次元等) から1トークンあたりの
  線形コストとattentionペアあたりのコストを計算 (Qwen3.6-27Bの実測値を使用)
- `accumulate_prefill()`: trie (prefix木) で軌跡のターンを順に通し、
  naive (毎回フル再計算) vs cache-aware (最長一致接頭辞だけ再利用) な
  プリフィルトークン数を算出
- `compute_trajectory_flops()`: 1軌跡の合計FLOPsを計算する高レベルAPI
- `example_usage()`: 追記のみ/途中編集/先頭編集の3パターンでFLOPsを実際に
  比較し、編集位置が前にあるほどコストが上がることを確認する

### 3. [rl_loss.py](rl_loss.py)
**ロス: Success-Gated Efficiency Advantage を使った Dual-Channel GRPO**

- `success_gated_efficiency_advantage()`: 論文 Eq.5 の a_eff (成功軌跡のみ、
  グループ平均FLOPsとの相対差をclipしたもの)
- `build_role_mask()` / `dual_channel_returns()`: 「コンテキスト編集ターンの
  トークンだけ」に効率アドバンテージを加算する (公式パッチの
  `custom_advantage_function` をそのまま再実装)
- `grpo_clipped_loss()`: token-mean のクリップ付きポリシー勾配損失 + 低分散KL
  (論文の clipping eps 0.2/0.28, KL係数0.01 を使用)

### 4. [main_flow.py](main_flow.py)
**メインフロー: ClmAgent の1タスク実行ループ**

- `BudgetController`: 予算ナッジ (25/50/75%)・最終ターン・停止の状態機械
  (公式 `utils/budget.py` の中核ロジックを簡略移植)
- `ctx_turn_is_free()`: 純粋な圧縮ターンはタスクステップを消費しない判定
- `ClmAgentLoop`: `context_env.ContextEnv` + `BudgetController` を統合した
  1タスク実行ループ。LM呼び出し部分は `policy` callable として差し替え可能
- `example_usage()`: 「圧縮しないポリシー」と「予算ナッジを見て実際に圧縮する
  ポリシー」を同一タスクで走らせ、前者が予算超過で失敗し後者が完走することを
  実際に比較する

### 5. [skill_evolution.py](skill_evolution.py)
**ICL: スキル文書のテキスト進化ループ (論文 Eq.4)**

- `score()` / `diff_se()` / `gate()` / `pick()` / `pareto()`: 開発セットでの
  候補選別の統計的手続き (公式 `clm_icl/evolve.py` の純粋関数をそのまま移植)
- `SkillEvolution`: 「現行スキルでロールアウト→対比ノート作成→候補生成→評価→
  ゲート判定」のラウンドループ
- `example_usage()`: 合成グレーダー (特定の語を含むスキルほど高精度・低コスト)
  で実際に3ラウンド進化させ、優れたスキルに収束することを確認する

### 6. [finetune_example.py](finetune_example.py)
**実際に動く強化学習ファインチューニングスクリプト (Trainer不使用)**

- `TrajectoryDataset` / `GRPOCollator`: pandas DataFrame (グループID・成否・
  FLOPs・トークン列・ターン構造) から dual-channel advantage 付きバッチを構築
- `_ToyPolicyLM`: デモ用極小causal LM (実運用ではQwen3.5-9B等に差し替え)
- `train()`: AdamW (lr 1e-6 constant, wd 0.1) + GRPOクリップ損失 + KL の手動
  訓練ループ (論文 `clm_rl/README.md` のハイパーパラメータに準拠)
- `main()`: 実行すると合成ロールアウトDataFrameで全パイプラインを実際に
  数ステップ学習させるデモが走る (`python finetune_example.py`)

### 7. [live_demo_qwen.py](live_demo_qwen.py)
**実際のローカルLM (Ollama上のQwen3:8b) に CLM を取り付けて試すデモ**

CLMはモデルを改造する手法ではなく「既存の任意のLMに取り付けるハーネス」なので、
ローカルのOllama (`ollama pull qwen3:8b`) に対して `context_env.ContextEnv` を
実際に取り付けて試せる (litellm等は使わず `requests` でOllamaのnative API
`/api/chat` を直接叩く)。

- タスク: ContextBenchのKV Storeを単純化したもの。SET操作を1つずつ送り込み、
  モデルは好きなタイミングで ` ```bash ... ``` ` ブロックでコンテキスト
  ファイルを圧縮してよい。最後にGETで値を復元できるか検証する
- `python live_demo_qwen.py` で実行。コンテキスト予算をきつくしておくと、
  ナッジ通知を受けて実際にモデル自身が `cat > <ctx_file>` 等のコマンドで
  コンテキストを要約・圧縮する様子が観察できる

**実際に動かして観察できたこと** (budget=500, Qwen3:8b, thinking有効):
モデルは `CONTEXT BUDGET NUDGE` を見ると自発的に ` ```bash ``` ` ブロックで
`LIVE_CTX_MAIN.txt` を書き換え、コンテキストを実際に ~396→227トークンまで
圧縮した。ただしモデル自身が選んだ要約戦略 ("Summary: stored k0..k13 so far")
は各キーの値そのものを残していなかったため、後続のGET操作で一部の値を
復元できなかった (3問中1問正解)。これは本サーベイのFAQ Q6で触れている
ContextBenchのNeedle Retentionタスクが突く問題点と全く同じ失敗モードが、
実際の小型モデルで再現された例である。

---

## CLMの主要イノベーション

### 1. **Context as a File**
**問題**: 既存ハーネスは「いつ・どう圧縮するか」を人間が固定ツールとして事前定義する。

**解決**:
```python
# context_env.py (抜粋)
rendered = self.write_mirror(messages)       # 会話 -> 1枚のファイル
proc = self.execute(command)                 # LMの任意のBashコマンドを実行
read_back = self.read_mirror()
if read_back.strip() != rendered.strip():
    self._apply_edit(messages, rendered, read_back, before)  # 編集を反映
```

**効果**: BrowseComp-Plusで11.4%高精度・21.5%少ないFLOPs (Codex-style summary比)。

**実装**: [context_env.py](context_env.py)

---

### 2. **Edit Gate (fit / shrink)**
**問題**: 無制限の編集を許すと、LMが誤って内容を重複させてコンテキストを
むしろ肥大化させるリスクがある。

**解決**: 編集後に「成長」していた場合、fitゲートなら予算に収まる場合のみ許可、
shrinkゲートなら常に拒否。拒否時はLMにその場で理由を通知する。

**実装**: [context_env.py](context_env.py) の `may_grow()`/`reject_rule()`

---

### 3. **Success-Gated Efficiency Advantage**
**問題**: 単純に「編集頻度」や「削除量」を報酬にすると、重要な情報を消す
reward hackingを誘発する。

**解決**:
```python
# rl_loss.py (抜粋): Eq.5
a_eff[succ_idx] = torch.clamp((mean_flops - flops[succ_idx]) / mean_flops, -1.0, 1.0)
# 失敗軌跡・成功が2未満のグループは常に0
```

**効果**: Qwen3.5-9BのBrowseComp-Plusが28.8%→42.5%、FLOPsは1.52→1.34 PFLOPs/問。

**実装**: [rl_loss.py](rl_loss.py)

---

### 4. **テキスト進化によるスキル最適化**
**問題**: 強化学習は重みの更新が必要で、即時には使えない。

**解決**: ロールアウトの対比ノートから候補スキル文書(自然言語)を生成し、
開発セットでゲート判定して選別するループ (GEPA系)。

**効果**: ContextBenchのKV Store課題で、開発セット精度を22.3%→83.8%に改善。

**実装**: [skill_evolution.py](skill_evolution.py)

---

## 処理フロー詳細

### 推論フロー (main_flow.py)

```python
from main_flow import ClmAgentLoop

def my_policy(messages):
    # 実運用では litellm.completion(...) 等でLMを呼び、
    # tool_call が bash なら command 文字列を返す
    ...
    return "python3 - <<'PY'\n...\nPY"  # または "DONE"

loop = ClmAgentLoop(policy=my_policy, max_steps=30, context_budget_tokens=32000)
result = loop.run(system_prompt="...", task_prompt="...")
print(result.free_ctx_turns, result.steps_used)
```

### 学習フロー: RL (rl_loss.py + finetune_example.py)

```python
# 1. ロールアウトのメタデータ(pandas DataFrame)からグループ単位で advantage を構築
a_eff = success_gated_efficiency_advantage(success, flops)         # Eq.5
returns = dual_channel_returns(samples, a_eff, w_eff=0.25)         # Eq.6

# 2. 現在ポリシー / 旧ポリシー / 参照モデルのlogprobsからGRPO損失を計算
out = grpo_clipped_loss(logprobs_new, logprobs_old, logprobs_ref, advantages, loss_mask)
out["loss"].backward()
```

### 学習フロー: ICL (skill_evolution.py)

```python
evo = SkillEvolution(task_ids, run_episode, proposer, n_candidates=4, reps=2)
result = evo.run(rounds=5)   # 各ラウンド: ロールアウト -> ノート -> 候補生成 -> ゲート
```

---

## 学習データフォーマット

`finetune_example.py` が期待する pandas DataFrame のスキーマ (1行 = 1ロールアウト軌跡):

| カラム | 型 | 説明 |
|---|---|---|
| `group_id` | str | 同じプロンプトからサンプリングされたGRPOグループのID |
| `trajectory_id` | str | 軌跡の一意なID |
| `success` | bool | タスクに成功したか (論文: バイナリ報酬) |
| `raw_reward` | float | グループ正規化前の生の報酬 |
| `flops` | float | 軌跡の prefix-reuse FLOPs (`prefix_reuse_flops.py` の出力) |
| `token_ids` | list[int] | 応答トークンID列 |
| `turn_lengths` | list[int] | 応答をターン単位に分割した各ターンの長さ |
| `turn_is_edit` | list[bool] | 各ターンが「純粋なコンテキスト編集ターン」かどうか |

論文の実際のRL設定 (Appendix "Reinforcement learning"): OpenResearcher 3,040プロンプト、
1プロンプトあたり32ロールアウト、バッチ8プロンプト×32サンプル、Adam lr 1e-6、
weight decay 0.1、clip [0.2, 0.28]、KL係数0.01、70ステップ、efficiency weight 0.25。

---

## FAQ

### Q1: なぜRL学習コードを完全に再実装しないのか?

**A**: 公式リポジトリの `clm_rl/README.md` が "RL training for CLM is not part
of this repository" と明記しており、実際に確認してもそのディレクトリには
README と2つのパッチファイルしか存在しない (学習基盤である ProRL-Agent-Server /
Slime は別リポジトリで、本サーベイの対象外)。公開されている約70行のパッチ
(`dual_channel_advantage.py` の新規追加分) にアルゴリズムが完全に書かれているため、
本サーベイの `rl_loss.py` はそのアルゴリズムをそのままスタンドアロンの PyTorch
実装として再構成している。

### Q2: なぜ Suffix Cache Reuse (SCR) を実装しないのか?

**A**: SCRは特定バージョン (SGLang 0.5.16) のRadixTree/GPU側KVキャッシュ
スロット操作へのモンキーパッチであり、GPU上で実際にSGLangサーバーを起動しないと
意味のある動作確認ができない。アルゴリズム自体 (K個の生存スパンを診断して
再配置し、RoPEを再回転、線形注意層は直前の再帰状態をスナップショットして継続)
は「公開物の実態」の章と「アーキテクチャ全体像」で文章として解説した。

### Q3: Edit Gate の fit と shrink の違いは?

**A**: 編集によってコンテキストが「成長」した場合の扱いが異なる。
`fit`: 成長後も予算に収まるなら許可 (デフォルト)。`shrink`: 成長は常に拒否
(圧縮だけが許される)。どちらでも拒否された編集はメッセージ列に反映されず、
LMにその場で理由が通知される (`context_env.py` の `may_grow()`)。

### Q4: 「純粋な圧縮ターン」はなぜステップを消費しないのか?

**A**: 論文の設計思想は「コンテキスト管理はタスクの進行ではない」という
ことであり、`max_steps` の消費を圧縮ターンに課すと、モデルが予算に収まるよう
圧縮する行為そのものがタスク達成の足を引っ張ってしまう。そこで
「ミラーが変わり、タスク出力(stdout/stderr)が空で、exit 0」の場合のみ無償ターン
として扱う (`main_flow.py` の `ctx_turn_is_free()`)。

### Q5: role_mask はどう作られるのか?

**A**: ロールアウト時に、各ターンのassistantアクションが「ミラーファイルへの
書き込みのみ (ctx_turn_is_free と同じ基準)」であれば1、それ以外は0をそのターン
の全トークンに付与する (`rl_loss.py` の `build_role_mask()`)。これは学習時に
dual-channel advantageのefficiencyチャンネルをどこに配るかを決める、
推論アーキテクチャには影響しないタグ付けである。

### Q6: ContextBench (論文の `ourbench`) とは?

**A**: 推論や知識を要さない4つの合成タスク (Needle Retention, Sudoku
Sketchpad, KV Store, Log Triage) で、純粋にcontext管理能力だけを測る診断用
ベンチマーク。32,768トークンの予算に対し最大24倍の「context pressure」まで
負荷をかけ、既存の固定戦略 (Summary, Self-Compact, Context Folding, ACM, RLM等)
がどこで崩れるかを明らかにする (論文 Sec 2, Appendix C)。本サーベイでは実装対象
外とし、`skill_evolution.py` の例は同じ構造を持つ合成KV Storeタスクで代替した。

### Q7: 実運用でこのサーベイのコードをそのまま使えるか?

**A**: `context_env.py` の `render_editable`/`parse_back`/`may_grow` は依存ゼロ
の純粋関数で、公式実装とほぼ同一なのでそのまま使える。`main_flow.py` の
`ClmAgentLoop` は `policy` callable を実際のLM呼び出し (litellm等) に差し替える
だけで動く。`rl_loss.py` の損失計算はPyTorchの標準APIのみに依存している。
`finetune_example.py` の `_ToyPolicyLM` は、本物の `transformers.AutoModelForCausalLM`
(Qwen3.5-9B等) に差し替える前提のデモ用スタンドインである。

---

## まとめ

CLM は以下の2層構造でエージェントのコンテキスト管理を統一的に扱う:

1. **Context as a File (ゼロショットの構造的新規性)**: LMの生きたコンテキストを
   1枚の編集可能ファイルとしてミラーし、既存ハーネスの固定アクション空間を
   撤廃する
2. **学習層 (ICL + RL)**: ①の自由度の上で、スキル文書のテキスト進化 (ICL) と
   dual-channel advantageによる stepwise GRPO (RL) で、context管理戦略自体を
   最適化する

**付随する効率化**: Prefix-Reuse FLOPs という正しい計算コスト指標を定義し、
サービング側でも Suffix Cache Reuse でその恩恵を実際に回収する。

**公開状況**: ハーネス本体・ICL進化ループは公開コードがそのまま動く。
RL学習コードは非公開 (外部フレームワークへの小さなパッチのみ)。
SCRは特定SGLangバージョンへのGPUモンキーパッチ。本サーベイはこれらのギャップを
論文・パッチの記述から実際に動くコードとして埋めている。

---

## 参考文献

- 論文: [Context Language Models](https://arxiv.org/abs/2609.37725)
- 公式リポジトリ: [github.com/facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models)
- 関連: [Harbor](https://github.com/laude-institute/harbor) (エージェント評価基盤),
  [SGLang](https://github.com/sgl-project/sglang) (推論サービングエンジン),
  GEPA (Agrawal et al., "Reflective prompt evolution can outperform reinforcement
  learning", ICLR 2026) (テキスト進化によるプロンプト最適化),
  [Slime](https://github.com/THUDM/slime) (RL学習フレームワーク)

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。
`context_env.py`/`skill_evolution.py` の中核関数は公式実装 (依存ゼロの純粋関数)
からほぼそのまま移植していますが、RL学習コード (`rl_loss.py`/`finetune_example.py`)
は非公開コードに対する公開パッチの記述からの再構成であり、実際の学習基盤
(ProRL-Agent-Server/Slime) の実装とは異なる場合があります。
