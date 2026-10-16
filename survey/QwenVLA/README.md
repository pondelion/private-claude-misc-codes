# Qwen-VLA 理解のためのサーベイまとめ

**論文**: Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
**著者**: Qwen Team (Alibaba)
**arXivソース**: `arXiv-2605.30280v2.tar.gz`（本ディレクトリの元資料に同梱、arXiv ID: 2605.30280 v2 と推定）

> **注意**: 公式実装コードは非公開。このまとめは論文本文（`.tex`ソース一式）のみを読んで作成した。
> 図表（PDF/JPG）は内容を読んでいない（テキストソースのみ参照という方針のため）。数値・設計は全て本文中の記述に基づく。

---

## 目次

1. [概要](#概要)
2. [問題設定：異種の身体化タスクを1つのモデルに統一](#問題設定異種の身体化タスクを1つのモデルに統一)
3. [アーキテクチャ全体像](#アーキテクチャ全体像)
4. [Embodiment-aware Prompt Conditioning](#embodiment-aware-prompt-conditioning)
5. [統一Action/Trajectory表現](#統一actiontrajectory表現)
6. [学習目的関数](#学習目的関数)
7. [4段階の学習レシピ（T2A→CPT→SFT→RL）](#4段階の学習レシピt2acptsftrl)
8. [事前学習データミクスチャ](#事前学習データミクスチャ)
9. [Post-Training: SFT と RL](#post-training-sft-と-rl)
10. [主要な実験結果](#主要な実験結果)
11. [Ablation Studies](#ablation-studies)
12. [Limitations & Future Work](#limitations--future-work)
13. [FAQ](#faq)
14. [参考文献](#参考文献)

---

## 概要

Qwen-VLAは、**マニピュレーション・ナビゲーション・人間エゴセントリック動作・軌跡予測**という見た目には異種のembodiedタスク群を、**単一のVision-Language-Actionモデル**で統一的に扱えるかを検証した研究。Qwen3.5-4Bのvision-languageバックボーンに、DiTベースのflow-matching action decoderを接続し、大規模なjoint pretrainingで学習する。

### キーコントリビューション

| 要素 | 内容 |
|---|---|
| **統一action-and-trajectory空間** | マニピュレーション動作、ナビゲーションwaypoint、人間の手動作を同じ予測フォーマットに統一 |
| **Embodiment-aware prompt conditioning** | ロボット種別・制御方式をテキストプロンプトで指定するだけで、アーキテクチャ変更なしに多embodimentに対応 |
| **4段階の学習レシピ** | T2A(text-to-action)→CPT(継続事前学習)→SFT→RLという「圧縮の視点」に基づく段階的学習 |
| **大規模joint pretraining** | ロボット軌跡・人間エゴセントリックデータ・合成シミュレーション・ナビゲーション・VLデータを混合 |

### 性能ハイライト（Qwen-VLA-Instruct、単一モデルで同時達成）

| ベンチマーク | スコア | 備考 |
|---|---|---|
| LIBERO | 97.9% | 単一arm tabletop |
| Simpler-WidowX | 73.7% | real-to-sim |
| RoboTwin-Easy / Hard | 86.1% / 87.2% | dual-arm、specialist(ABot-M0)を上回る |
| R2R Val-Unseen OSR | 69.0% | vision-language navigation |
| RxR Val-Unseen SR | 59.6% | vision-language navigation |
| ALOHA実機 OOD平均 | 76.9% | color/instance/position/background/instruction汎化 |
| DOMINO zero-shot SR | 26.6% | 動的（動くオブジェクト）マニピュレーション、fine-tuneなし |

**最重要な主張**：これらは**1つのgeneralistモデル**が、per-benchmark適応なしに同時達成した数値であり、多くの設定でタスク特化specialistモデル（$\pi_0$, $\pi_{0.5}$, GR00T N1.6, ABot-M0等）を上回る。

---

## 問題設定：異種の身体化タスクを1つのモデルに統一

マニピュレーション・ナビゲーション・軌跡予測・人間エゴセントリック動作は、観測フォーマット・制御周波数・予測horizon・行動次元・評価プロトコルが全部異なる。しかし共通の計算構造を持つ：**視覚観測に条件づけて、言語指示と身体制約を踏まえ、物理的/意味的に整合した将来行動/軌跡を予測する**。

統一定式化：

```
p_θ( y_{t:t+H-1} | o_t, x, e, z )

o_t : 視覚コンテキスト（1枚 or 複数枚の画像、動画、履歴ウィンドウ）
x   : 言語指示
e   : embodiment記述（ロボット種別・制御方式のテキストプロンプト）
z   : タスクファミリー識別子（任意）
y   : 予測対象（horizon H の行動/軌跡列）
```

`y`の意味はタスク依存だが表現形式は統一：
- マニピュレーション → 将来のエンドエフェクタ位置等
- ナビゲーション → navigation waypoint
- 軌跡中心タスク（自動運転等）→ 連続座標空間での将来軌跡
- 人間エゴセントリックデータ → MANO等の構造化ポーズ空間での手/身体動作

---

## アーキテクチャ全体像

```
入力: 視覚観測 o_t (画像/動画/履歴window) + 言語指示 x + embodiment prompt e
                              │
                ┌─────────────┴─────────────┐
                │   Qwen3.5 VLM backbone      │  ← ネイティブmultimodal
                │   (ViT + spatial merging    │     early fusion
                │    → text token streamに    │
                │    直接interleave)          │
                │                             │
                │   hybrid attention:         │
                │   ・大半の層: gated linear  │
                │     attention               │
                │   ・一定間隔: grouped-query │
                │     softmax attention       │
                └─────────────┬───────────────┘
                              │ VLM hidden states
                              │
                 ┌────────────┴────────────┐
                 │  Linear (3.9M params)     │  ← VLM次元→DiT chダイム
                 └────────────┬─────────────┘
                              │
   noisy action chunk Y_τ ────┤ concat（1つのシーケンスに結合）
   (flow-matchingのτ時点)     │
                              ▼
                ┌──────────────────────────────┐
                │  Action Expert (single-stream │
                │  DiT-style flow matching)     │
                │                                │
                │  joint self-attention          │
                │  + AdaLN timestep conditioning │
                │  + multi-section RoPE          │
                │    (backboneと位置整合)        │
                │                                │
                │  16 DiT blocks (70.8M each     │
                │  = 1.13B params)                │
                │  + action projection MLP(4.9M) │
                │  + timestep embed(2.8M)        │
                │  + output AdaLN modulation(4.7M)│
                │  合計 ≈ 1.15B params            │
                └────────────┬───────────────────┘
                              │ 数stepのEuler積分（τ=1→τ=0）
                              ▼
                    action chunk ŷ_{t:t+H-1}
                    (H=16ステップ、低遅延・リアルタイム制御向け)
```

VLM backboneは事前学習済み（強力な視覚理解・多言語指示追従・構造化推論）、一方でDiT action decoderはランダム初期化 — この**非対称性**が、後述の4段階学習レシピを必要とする理由になる。

---

## Embodiment-aware Prompt Conditioning

複数ロボットプラットフォームを1つの共有モデルで扱うため、各学習サンプルの先頭に**ロボット固有のテキストプロンプト**を付与する。これが唯一のembodiment指定インターフェース：

```
The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base].
The control frequency is {FPS} Hz.
Please predict the next {chunk_size} control actions to execute the following task: {ori_instruction}.
```

- `robot_tag`、腕数、waist/mobile baseの有無はembodimentごとに設定
- `FPS`と`chunk_size`はそのデータセット本来の制御周波数・予測horizonを反映
- このプロンプトだけでaction decoderの出力次元・制御規約を切り替える（アーキテクチャ変更なし）

**デプロイ時の汎化**：実機に展開する際も、embodiment promptを対象ロボットの記述に差し替えるだけでゼロショット適応できる（VLM backbone / DiT decoderは不変）。これが可能な理由は2つ：
1. CPT段階でシミュレーション/実世界の両方の視覚分布を経験済み（backboneがrenderer特有でない）
2. RL段階のtask-success指標は視覚ドメインに不変（renderer固有のテクスチャではなく意味的な視覚特徴に依存するよう促される）

---

## 統一Action/Trajectory表現

各学習サンプルは目標テンソル $\mathbf{Y} \in \mathbb{R}^{H \times K}$（$H$=固定予測horizon、$K$=全制御モード共通の固定チャネル数）に変換される。**物理的な行動空間を1つに統一するのではなく**、テンソルのインターフェースとマスキング機構だけを統一し、各データセットは自身のネイティブな制御規約（embodiment promptとデータセット別正規化で指定）を維持する。

### 制御信号の2ファミリー

| ファミリー | 内容 |
|---|---|
| **マニピュレーション** | delta EEF位置($\Delta x,\Delta y,\Delta z$)、EEF回転（オイラー角/クォータニオン）、絶対関節位置、グリッパー開閉、dexterous-hand関節角 |
| **ナビゲーション軌跡** | VLN規約に従い waypointごとに $(\Delta x, \Delta y, \Delta\theta)$（平面上の相対変位＋方位変化） |

物理的意味は違うが、どちらも「horizonに渡って予測される実数値ベクトル列」として同一に扱われる。

### チャネルレイアウト（マスキング）

- ある制御モードが使うチャネル数 $c \le K$ は、$\mathbf{Y}$の**先頭$c$次元**に配置し、残り$K-c$次元はゼロパディング
- バイナリマスク $\mathbf{M} \in \{0,1\}^{H\times K}$：$M_{h,k}=1 \iff$（$k<c$ かつ $h$がそのタスクのchunk長$H_\text{task}\le H$以内）
- embodiment固有の出力headは不要 — 1組のDiTパラメータで全制御モードを処理し、マスクがパディング部分の勾配を遮断する

### Task-aware conditioning

embodiment-aware promptのトークンはVLM backboneで処理され、その隠れ状態がnoisy action chunkと結合してDiTに入力される。アーキテクチャ変更なしに、サンプルごとの正確な制御仕様に常に条件づけられる。

---

## 学習目的関数

### Flow-matching action loss（連続制御ターゲット全般）

クリーンターゲット $\mathbf{Y}_0$、ノイズ $\mathbf{Y}_1\sim\mathcal{N}(0,\mathbf{I})$ から線形補間 $\mathbf{Y}_\tau = (1-\tau)\mathbf{Y}_0 + \tau\mathbf{Y}_1$（$\tau\in[0,1]$）を作り、action expert $v_\theta$ に条件付き速度場を予測させる。

パディングに勾配が支配されないよう、**チャネル単位・ステップ単位**の2段階平均を取る：

```
各有効チャネル k (k < c) のMSE:
  ℓ_k = Σ_h M_{h,k} ‖(v_θ(Y_τ,τ | o,x,e,z) − (Y_1−Y_0))_{h,k}‖²
        ────────────────────────────────────────────────
                        Σ_h M_{h,k}

c チャネルで平均:
  L_act = E_{τ,Y_0,Y_1} [ (1/c) Σ_{k=0}^{c-1} ℓ_k ]
```

これにより、embodimentごとに使うチャネル数が違っても各制御次元の寄与が均等になり、パディング位置は完全に勾配から除外される。**推論時**は $\tau=1\to0$ の数stepのEuler積分でaction chunkを生成。

### Vision-language loss（next-token prediction）

補助VLデータ・fine-grained action caption・自動運転VQA・一般VL事前学習コーパスに対して標準的なnext-token予測損失：

```
L_vl = − Σ_i log p_θ(w_i | w_{<i}, o_{1:t})
```

これがembodied co-training下での言語grounding安定化・知覚/推論スキルのcatastrophic forgetting防止に効く。

### 合成損失

```
L = λ_act · L_act + λ_vl · L_vl
```

$\lambda_\text{act}, \lambda_\text{vl}$ は勾配スケールのバランスを取るよう調整。ミニバッチ内は固定サンプリング比で全タスクファミリーを混合し、毎stepでbackboneとaction expertを同時更新する。

---

## 4段階の学習レシピ（T2A→CPT→SFT→RL）

### なぜ段階分けが必要か：「圧縮」の視点

VLM backboneは強い事前学習済み表現を持つ一方、DiT action decoderはランダム初期化 — この非対称な状態からnaiveにjoint trainingを始めると、decoderは①行動分布の形、②言語/embodimentへの条件付け方、③flow-matchingダイナミクス、④視覚へのgrounding、を同時に学ぶ必要があり非効率・不安定。

キーとなる洞察：`"pick up the red cup"`のような言語指示＋embodiment promptは**わずか数トークンでタスク意図をコンパクトに符号化**しているのに対し、対応する行動軌跡は数百次元の高頻度joint-position値になりうる。この次元ギャップを埋めるのは構造化された**「解凍(decompression)」問題**である。

### Stage I: Text-to-Action DiT Pretraining (T2A)

- VLMを**凍結**し、DiTのみを学習。**画像入力を意図的に与えない**
- テキスト＋embodiment promptのみから高次元行動分布を再構成させる純粋な「言語→行動」解凍器として機能させる
- これにより「言語が行動空間のどの領域を選ぶか」「embodiment promptがモータープログラムをどう変調するか」「軌跡レベルの時間的一貫性・構成性」を、視覚のショートカットなしに学習する

### Stage II: Continued Pretraining (CPT)

- 両モジュールを**unfreeze**し、実世界＋シミュレーションを両方含む多様な混合データで学習
- T2Aが対処できなかった「視覚観測への行動のgrounding」「embodied知覚へのbackbone適応」に集中
- 実世界・sim両方を見たcheckpointができることで、後段のSFT/RLがどちらのドメインにも特化しやすくなる

### Stage III: Supervised Fine-tuning (SFT) — 2つの並列トラック

1. **Multi-task SFT**：VQA・spatial grounding・マニピュレーション・ナビゲーションをembodiment/task-balancedサンプリングでjoint fine-tune
2. **実機track**：社内teleoperationデータでfine-tune、CPTのクロスドメイン事前知識が実機ハードウェアに転移するかを検証

### Stage IV: Reinforcement Learning (RL)

- Multi-task SFT checkpointから開始
- **単一のシミュレーション環境(SimplerEnv)のみ**でsparse binary success rewardを使いfine-tune → 最終モデル**Qwen-VLA-Instruct**
- 目的：SFTは「デモに対する尤度」を最適化するに過ぎず、真に重要な「closed-loopのtask success」を直接最適化できない、そのギャップを埋める
- 意図的に**1つの環境だけ**でRLし、そこで得たtask-success改善が他のOOD環境に転移するかをテストする設計

---

## 事前学習データミクスチャ

| データソース | 割合(%) |
|---|---|
| ロボットマニピュレーション軌跡 | 74.2 |
| 人間エゴセントリック軌跡 | 6.0 |
| ナビゲーション軌跡 | 7.5 |
| 合成シミュレーション軌跡（独自） | 3.7 |
| 一般vision-languageデータ | 3.4 |
| 空間grounding(2D) | 2.5 |
| 自動運転VQA | 2.4 |
| Fine-grained embodied action caption | 0.2 |
| **合計** | **100.0** |

### ロボットマニピュレーション軌跡（74.2%）

**公開データセット**：RoboSet, Galaxea, AgiBot World, RoboCOIN, RoboMIND V1/V2, RDT-1B, DROID, BridgeData V2, RH20T, RT-1, BC-Z — 10,000時間超の異種embodiment/タスクデータ。加えてInternData-A1・GR00T-X-Embodiment-Simのシミュレーション軌跡。

**独自データ**：社内実機teleoperationデータ1,000時間超（全体の約20%）＋独自シミュレーションパイプラインで生成した800万軌跡超（3.7%）。

**代表的なembodiment一覧**：

| ロボット | 腕 | 行動タイプ |
|---|---|---|
| WidowX | Single | ΔEEF + Gripper |
| Google Robot | Single | ΔEEF + Gripper |
| Franka Panda | Single/Dual | ΔEEF+G; Abs Joint+G |
| ARX5 | Dual | ΔEEF + G |
| Fourier GR-1 | Dual | ΔEEF + G |
| Mobile ALOHA | Dual | ΔEEF+G; Abs Joint+G |
| AgiBot A2-D | Dual | Abs Joint+G; Abs Joint+DH(dexterous hand) |
| Galaxea R1 | Dual | Abs Joint + G |
| AIRBOT MMK2 | Dual | Abs Joint + DH |
| TienKung | Dual | Abs Joint+G; Abs Joint+DH |
| Real Human | Dual | ΔEEF (MANO由来) |

**行動正規化**：データセット$k$の各行動次元$d$について、1st/99thパーセンタイル$q^k_{01}, q^k_{99}$を計算し

$$\tilde{a}_d = 2\cdot\frac{a_d - q^k_{01}}{q^k_{99}-q^k_{01}} - 1$$

で$[-1,1]$にクリップ正規化（元の相対的な動きの構造は保ったままスケール差を除去）。

**カメラビュー表現**：`<|tag_start|>〈image〉<|tag_end|>`のようなview固有の境界トークンでカメラ由来（ego, cam_left_wrist等）を明示。

### 人間エゴセントリックデータ（6.0%）

ロボットteleoperationより**スケーラブルな実世界マニピュレーション経験源**として活用。

**4つのデータソース**：
1. Ego4D / EPIC-KITCHENS（VITRA処理：自動セグメント化＋fine-grained言語注釈＋frame単位3D手/カメラ軌跡）
2. EgoDex（Apple Vision Pro、829時間、194タスク、3D手/指tracking付き）
3. EgoVerse（1,300時間超、1,965タスク、240シーン）
4. Xperience（同期first-person記録＋depth＋手/身体motion capture＋階層的言語注釈）

**行動表現**：各手について、手首の相対SE(3)変換（並進3 + axis-angle回転3 = 6次元）＋ 手articulationを45次元axis-angle jointにPCAをかけ上位10主成分（**eigengrasp**）で圧縮表現。両手合計 **32次元/タイムステップ**。

### 合成シミュレーションデータ（3.7%）

IsaacLab（シミュレータ）＋ cuRobo（衝突回避モーションプランニング）ベースの独自パイプライン。

**Vision-language-actionデータ**：`RoboInF`パイプライン（内部早期版）でscene生成→task生成→success check合成→motion program生成→domain randomizationでrollout。
- 20 tabletop scenes × 10 object配置 = 200 base scene構成
- 450タスク（short-horizon「2本の緑ステープラーを並べる」〜long-horizon「飲み物をまとめてスポンジだけ離す」）
- タスクあたり300成功軌跡（lighting/camera/背景/table texture/robot初期状態/object初期姿勢/controller動特性をrandomize。背景プール約3K、テーブルテクスチャ約1K）
- 完全軌跡をsubtask軌跡にも分割 → 合計**359,848**の成功軌跡（subtask含む）

**Language-actionデータ**（テキストのみ、視覚入力なし）：
- 6つのタスクテンプレート族（pick-and-place, linear push, linear pull, 回転+repositioning, 視点方向への回転, 2物体のpositional swap）
- 6つのsingle-arm robot configuration（Franka Panda, UR10e, UR5e, Kinova Gen3, TM12, xArm7）× 各テンプレート ≈ 200kずつ
- 合計 **約720万軌跡**、**14,000時間超**（物理シミュレーション/レンダリングなし、GPU並列のバッチモーションプランナーでkinematically validな軌跡のみ保持、50Hzでjoint position/velocity/EEF pose/gripper状態を記録）
- **Stage I (T2A)の主要コーパス**として使用

### ナビゲーションデータ（7.5%）

移動ロボットは3自由度（平面並進＋heading回転）と仮定。動画は2FPSサンプリング。

| サブカテゴリ | 割合 | 内容 |
|---|---|---|
| Instruction following | 4.3% | 移動プリミティブ＋テキストランドマークが混在する指示への追従 |
| Object searching | 2.3% | 大規模屋内環境でテキスト記述されたオブジェクトを探索（既訪問エリアの記憶＋効率的探索） |
| Target tracking | 1.0% | 移動ターゲット（例：人）を安全距離を保って追跡 |

### Vision-Languageデータ（8.5%、内訳は上表参照）

- **Fine-grained embodied action caption**（0.2%）：粗いタスクラベル（"pick up the ceramic bowl"）では同一ラベルが左右どちらの把持か等曖昧なため、13次元（action primitive, actor identity, object認識/曖昧性解消, contact region, 始点/終点, 軌跡と向き, gripper状態, 身体動作）に渡るdense captionを2段階パイプライン（Qwen3.6-plusで粗い行動系列抽出→高密度フレームで詳細記述）＋人手レビューで生成。約48,000ペア
- **自動運転VQA**（2.4%）：時間的シーン理解・surround-view空間推論・言語grounded localization・planning-aware推論の4能力軸で複数データセットを統合
- **空間grounding**（2.5%）：2Dバウンディングボックスgrounding
- **一般VLデータ**（3.4%）：captioning, 知識VQA, OCR/text-grounding, interleaved instruction following, referring expression等。video-centric/空間関係/3D-aware supervisionをup-weight

---

## Post-Training: SFT と RL

CPT出力（**Qwen-VLA-Base**）は広いcross-task/cross-embodiment汎化を持つが、特定タスクでの正確なclosed-loop制御精度はまだ不足。2段階post-trainingで橋渡しする。

### Multi-Task SFT

- データ：一般VL（VQA/spatial grounding/action captioning）＋ロボットマニピュレーションデモ（single/dual-arm/humanoid、prediction horizon=16 action steps/chunk）＋VLNエピソード（成功エピソードのみ、horizon=8 waypoints/chunk）
- 損失重み：VL next-token予測=0.1、マニピュレーション/ナビゲーションのaction予測=1.0（言語/視覚理解を保ちつつaction生成に勾配容量を集中）

### Reinforcement Learning

**目的**：尤度ベースのSFTは「デモをどれだけ模倣したか」の代理指標しか最適化できず、方策自身の出力がデモ分布から外れると失敗しうる。RLはpolicyが自ら rolloutした軌跡でtask-successを直接最適化する。実装は**RLinfフレームワーク**。

**方策最適化目的（PPO + GAE）**：

$$\mathcal{L}^\text{actor}(\theta) = -\mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat A_t\big)\Big]$$

$$\mathcal{L}(\theta) = \mathcal{L}^\text{actor}(\theta) + c_v \mathcal{L}^\text{value}(\theta)$$

- $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$、$s_t=(o_t,x,e)$、$a_t$=horizon $H$のaction chunk
- GAE: $\gamma{=}0.99$, $\lambda{=}0.95$、clip $\epsilon{=}0.2$、$c_v{=}1$、rolloutバッチにつき4 epoch更新

**Value推定**：別建てのcriticネットワークではなく、VLM backboneに軽量value headを直接付与（全隠れ状態をmean-pool→線形射影）。VLM隠れ状態には**stop-gradient**をかけ、value関数の勾配がpretrained backboneに伝播しないようにする。学習率は actor の**約20倍**（$10^{-4}$ vs actor $5\times10^{-6}$）でvalueだけ素早く収束させる。

**Flow-matching下でのlog確率推定（技術的に重要なポイント）**：自己回帰トークン方策と違い、flow-matchingは陰的な密度しか定義しない（学習済み速度場＋反復denoisingのみ）。決定論的なprobability-flow ODEを、各Eulerステップで制御されたノイズを注入することでSDEに変換し、各遷移を陽的なGaussianとして扱うことで、log確率をODE数値積分なしに解析的に計算できるようにしている。Rollout時に中間denoising状態を保存しておき、PPO更新時に現在パラメータで速度場を再評価してGaussian log確率を再計算（ランダムに1つのdenoising stepだけをlog確率推定に使い、追加のDiT forward passは1回で済む）。log確率とadvantageは**action-chunkレベル**（$H{=}16$ステップ単位）で計算。

**報酬設計**：sparse binary reward（$R{=}1$ タスク成功、$R{=}0$それ以外）のみ。学習済み報酬モデルは使わず、全信号はシミュレータのground-truth task-completion判定から得る。credit assignmentはGAEが担う。

**Rollout基盤**：クライアント・サーバー分離アーキテクチャ（リモートbenchmarkサーバーがsim環境、学習プロセスはネットワーク越しに通信）。$N{=}128$並列環境（task難易度に応じた非均等割当）。1 iteration = 8 rollout epoch × 128 env steps → $128\times\lfloor128/16\rfloor\times8=8{,}192$ transition chunks/iteration。rollout時temperature $\tau{=}1.0$、評価時は$\tau{=}0.6$に下げてsharpにする。

**OOD汎化**：embodiment promptが唯一のplatform固有インターフェースなので、RL後の方策はドメイン適応headなしにOOD環境・タスク・embodimentへ転移する。

---

## 主要な実験結果

### シミュレーションでのマニピュレーション（specialist vs 単一generalist）

| Method | Type | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---|---|---|---|---|
| $\pi_0$ | Specialist | 94.4 | -- | -- | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | -- |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | -- |
| $\pi_{0.5}$ | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | **98.6** | **58.3** | -- | 86.0 | 85.0 |
| Being-H0.5 | Specialist | 97.6 | 53.3 | -- | -- | -- |
| Qwen-VLA-Base | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| **Qwen-VLA-Instruct** | **Generalist** | 97.9 | 56.7 | **73.7** | **86.1** | **87.2** |

**単一のgeneralistモデルが、大半のspecialistを上回る**。RoboTwin-Easy/HardではSOTA specialist(ABot-M0)すら超える。Qwen-VLA-Base（pretrainのみ）→Instructで全ベンチマーク一貫して向上（LIBERO +7.1pp, RoboCasa-GR1 +16.3pp, Simpler-WidowX +9.4pp, RoboTwin-Easy +21.8pp, RoboTwin-Hard +20.8pp）。

### 実世界ALOHA（in-domain / OOD）

| モデル | In-domain平均 | OOD平均 |
|---|---|---|
| GR00T N1.6 | 28.6 | 25.4 |
| $\pi_{0.5}$ | 71.6 | 41.5 |
| Qwen-VLA-aloha（事前学習なし・from scratch） | 48.5 | 36.2 |
| **Qwen-VLA-aloha（Qwen-VLA-Baseからfine-tune）** | **83.6** | **76.9** |

**同一アーキテクチャで事前学習の有無だけを比較**：事前学習ありがin-domainで48.5%→83.6%、OODで36.2%→76.9%。**性能差はアーキテクチャではなく大規模事前学習の恩恵**であることを明示的に示している。

### Vision-and-Language Navigation（VLN-CE、Val-Unseen）

| Method | R2R OS↑ | R2R SR↑ | RxR SR↑ | RxR SPL↑ |
|---|---|---|---|---|
| NaVid | 49.2 | 41.9 | 45.7 | 38.2 |
| Uni-NaVid | 53.3 | 47.0 | 48.7 | 40.9 |
| NaVILA | 62.5 | 54.0 | 49.3 | 44.0 |
| StreamVLN | 64.2 | 56.9 | 52.9 | 46.0 |
| Qwen-VLA-Instruct | **69.0** | **57.5** | **59.6** | **47.8** |

### SimplerEnv-OOD（静的、未見のタスクタイプ）

Bridge trainのpick-and-placeのみでfine-tune、6つの未見タスク（MoveAway, MoveRight, PlaceNear, PlaceRight, PutFront, StackYellow=色反転）で評価。Qwen-VLA-Instructは平均**32.0%**（$\pi_{0.5}$の12.6%を大幅に上回る）。特に位置汎化（MoveRight, PlaceNear）で$\pi_{0.5}$が0%のところ33.3%/39.6%を達成。

### DOMINO（動的マニピュレーション、zero-shot）

現フレーム観測のみ（時系列motion入力なし）でzero-shot評価。Qwen-VLA-Instructは**SR 26.6% / MS 39.5**でDOMINO専用fine-tune済みのPUMA（SR 17.2%）すら上回る。「大規模joint pretrainingで学んだ空間-運動学写像が動的操作にも転移する」ことの根拠として提示。

---

## Ablation Studies

### T2A（Text-to-Action）Pretraining Ablation

Simpler-WidowXでのSFT後success rateで比較（no-T2Aベースライン: 60.9%）。

| 軸 | 最良設定 | 結果 |
|---|---|---|
| データ構成 | 合成20% + 実データ80% | **71.1%**（+10.2pp） |
| 予測モード | Full-sequence（chunk予測より一貫して優位） | chunk比+2.9〜4.9pp |
| 視覚入力 | **画像なし**（含めると-2.9pp、視覚ショートカットに逃げる） | 画像なし60.4% vs あり57.6% |
| Flow timestep分布 | T2A=Sigmoid-Normal, SFT/CPT=Beta | 71.1%（他組み合わせは全て劣化） |
| T2A訓練step数 | 2,000 step でピーク、40,000stepで過学習気味に劣化(60.4%) | 2,000: 71.1%, 4,000: 67.5%, 10,000: 67.2% |

**純データのみ**：real-only 51.0%、synthetic-only 64.1%（syntheticの方が広いカバレッジを与える）。**混合が最良**（両方の利点を活かす）。

### Vision-Language共学習の影響

Action-onlyとAction+VLデータを比較。単純なLIBERO/Simpler-WidowXでは差なし（干渉しない証拠）。RoboCasa-GR1（fine-grainedな物体認識・複合指示解釈が必要）では+4.9pp（51.1%→56.0%）、RoboTwin-2.0で+4.6pp（81.8%→86.4%）。

また、事前学習済みDiTをフレッシュなQwen3.5-4B VLMに付け替えて比較したところ、**事前学習済みDiTはfrom-scratch DiTより収束が速く到達点も高い**（DiT自体にも転移可能な事前知識が宿る）。

### 異種embodiment向けaction射影設計

7-DoF armと29-DoF whole-bodyのような次元差を吸収する3設計を比較：

| 設計 | パラメータ数 | 特徴 |
|---|---|---|
| Multi-MLP | $2h\sum_i d_i$ | embodimentごとに専用encoder/decoder |
| Concatenation | $2h\sum_i d_i$ | 全embodimentのactionを連結、共有encoder |
| **Zero-Padding**（採用） | $2h\,d_\text{max}$（最軽量） | 短いaction vectorをゼロpadding、共有MLP |

3設計とも性能差は1.2%未満で軽微。**最も軽量なZero-Padding**をデフォルト採用。

### RL Post-trainingの累積効果

| Stage | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | SimplerOOD | DOMINO SR | DOMINO MS |
|---|---|---|---|---|---|---|---|---|
| CPT | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| +SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| +RL | **73.7** | **56.7** | 86.1 | **87.2** | **97.9** | **32.0** | **26.6** | **39.5** |

RLはSimplerEnv（RL rolloutの収集元）で最大の伸び（+2.9pp）を示すだけでなく、**RL rolloutの対象になっていないベンチマークでも性能が維持/微改善**（RoboCasa +0.7pp、DOMINO SR +0.9pp等）— catastrophic forgettingが起きていないことの確認。

### State（proprioceptive）Conditioningの効果

RoboTwin-2.0で「状態なし」「VLMプロンプトに離散化して埋め込み」「DiTに直接連続値入力」を比較 → いずれも**+1.3pp以下の僅かな差**。視覚観測だけでほぼ十分な情報が得られる（特にend-effectorがカメラに映る場合）＋flow-matchingは相対変位を予測するため絶対状態への依存が薄い、と分析。結論：embodiment-aware text promptのみを保持し、明示的なstate conditioningは採用しない。

---

## Limitations & Future Work

1. **embodied actionデータの少なさ**：vision-language pretrainingデータと比べて依然桁違いに小規模・低多様性。ロングテールの物体・環境・embodiment・接触リッチなタスクへの頑健性が課題
2. **目的関数間のトレードオフ**：vision-language理解・ナビゲーション・action生成のjoint trainingは、action指向の学習が純粋なVL/ナビゲーション評価をわずかに悪化させうる。目的バランス・データカリキュラム・モジュール特化の改善余地
3. **評価は依然short-horizon・benchmark駆動**：長時間・失敗しがちな実世界デプロイは未解決の課題

**将来方向**：自律収集/シミュレーション/sim-to-real転移によるデータスケーリング、egocentric/third-person人間動画による物理的事前知識と時間的抽象化の獲得、long-horizon planning・episodic memory・world modelの統合（状態追跡・タスク分解・失敗回復・行動結果の予見）、force/tactile/proprioceptiveのようなよりリッチな物理フィードバック＋大規模RL。

---

## FAQ

**Q: Qwen-VLAは1つのモデルなのに、なぜspecialistモデルより強いことがあるのか？**
A: 異種embodiment/タスクを跨いだjoint pretrainingが、個別タスクだけでは得られない転移可能な視覚grounding・空間推論の事前知識を与えるため（論文の主張）。実際、ALOHA実験で「同一アーキテクチャ・pretrainingの有無だけ」を比較しており、差の原因がアーキテクチャでなくpretrainingにあることを明示的に切り分けている。

**Q: なぜT2A（Stage I）はわざわざ画像を見せずに学習するのか？**
A: 画像を早期に与えると、decoderが「視覚的な相関」に頼るショートカットを覚えてしまい、言語だけから行動分布を正しく学ぶ機会が失われるため。Ablationでも画像ありのchunk予測は画像なしより-2.9pp低い。

**Q: Full-sequence予測とchunk予測はどちらが良いか？**
A: Full-sequenceが一貫して優位（+2.9〜4.9pp）。chunk予測は長期依存関係を切り、chunk境界で位置の曖昧さを生むため。

**Q: Flow-matchingのようなimplicitな密度モデルに、どうやってPPOのlog確率比を計算するのか？**
A: 決定論的なprobability-flow ODEを、各Eulerステップにノイズを注入してSDEに変換することで、各遷移を陽的なGaussianとして扱い、解析的にlog確率を計算する。Rolloutで中間denoising状態を保存し、更新時に1回追加のforward passで再計算する。

**Q: RLは単一のシミュレーション環境（SimplerEnv）だけで行うのに、なぜ他のOOD環境にも効果が波及するのか？**
A: 論文の解釈では、RLが「decisiveな行動生成・compounding errorからの回復」という汎用的なスキルを学習するため。ablationで、RL rollout対象外のベンチマーク（RoboCasa, RoboTwin-Hard, LIBERO, SimplerEnv-OOD, DOMINO）でも性能が維持または微改善することを確認している。

**Q: 異種embodimentのaction次元差はどう吸収しているか？**
A: Multi-MLP・Concatenation・Zero-Paddingの3方式を比較し、性能差はほぼ無い（<1.2pp）ため、パラメータ数が最も少ないZero-Padding（不足分を0埋めして共有MLPに通す）を採用。

**Q: proprioceptive state（関節角度等）を条件付けに使わないのはなぜ？**
A: 使っても＋1.3pp以下の僅かな改善しかなく、視覚観測だけでほぼ十分な情報が得られる上、flow-matchingが相対変位を予測するため絶対状態への依存が薄いため。embodiment間でstateインターフェースを保守するコストに見合わないと判断し、不採用。

---

## 参考文献

- **Qwen-VLA**: Qwen Team, "Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments", arXiv:2605.30280 (推定, 元アーカイブファイル名`arXiv-2605.30280v2.tar.gz`より)
- **Qwen3.5**: バックボーンとして使用（`qwen35blog`として本文中引用、ネイティブmultimodal・hybrid attention）
- 比較対象として本文で引用される主要ベースライン: $\pi_0$ / $\pi_{0.5}$ (Black et al.), GR00T N1.6 (NVIDIA), StarVLA-OFT, ABot-M0, Being-H0.5, OpenVLA / OpenVLA-OFT, RDT-1B, InternVLA-M1, VLA-Adapter, NaVid / Uni-NaVid / NaVILA / StreamVLN (VLN系), PUMA / LingBot-VA (DOMINO動的マニピュレーション系)
- 公式コード: 非公開（本まとめ作成時点）
