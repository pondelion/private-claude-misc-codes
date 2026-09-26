# カスケードS2S検証 (cascade_exp1)

ASR(Parakeet) → LLM(Ollama, ストリーミング) → TTS(VOICEVOX) のカスケード方式で、
マイクで会話できるGradio検証アプリ。

- `pipeline.py`: ASR/LLM/TTSの処理ロジック、文単位パイプライン化、レイテンシ計測
- `app.py`: Gradio UI(マイク入力・会話履歴・逐次テキスト表示・レイテンシ表示)

## 事前準備

### 1. Ollama

```bash
ollama pull qwen3:8b
```

RTX 5090(Blackwell)ではOllamaが古いとCUDAカーネルが遅くなる問題があるため、
バージョンは最新化しておくこと(`ollama --version`で確認、古ければ公式インストーラで上書き)。

### 2. VOICEVOX(Docker)

```bash
docker pull voicevox/voicevox_engine:cpu-latest
docker run --rm -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:cpu-latest
```

CPU版で十分軽量。起動後 `http://127.0.0.1:50021/docs` でAPI疎通確認できる。

### 3. Parakeet (ASR)

追加作業不要。`pipeline.py`初回実行時に`nvidia/parakeet-tdt_ctc-0.6b-ja`を自動ダウンロードする
(`speech_exp`プロジェクトの`.venv`/依存関係を利用)。

### 4. HTTPS証明書(LAN内の別PCからマイクを使う場合のみ必須)

ブラウザの`getUserMedia`はセキュアコンテキスト(https、またはlocalhost)でしか動作しないため、
別PCから`http://<LAN IP>:7860`でアクセスするとマイクが使えない。
このディレクトリには生成済みの自己署名証明書(`cert.pem`/`key.pem`、SANに`192.168.0.4`を含む)
を置いてある。IPが変わった場合は`openssl_san.cnf`のIPを書き換えて再生成:

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365 \
  -config openssl_san.cnf
```

## 起動

```bash
cd cascade_exp1
uv run python app.py
```

- ローカル: `https://127.0.0.1:7860`
- LAN内の別PCから: `https://<このPCのIP>:7860`(自己署名証明書の警告は「詳細設定→アクセスする」で進む)

## 構成上のポイント

- LLM(Ollama)はストリーミングで受け取り、句読点(。！？)ごとに文単位で区切って
  VOICEVOXに逐次投げる(LLMの続きの生成とTTS合成を並行実行)
- 音声再生はサーバー側で各文の再生時間分だけペーシングしてから次を送る
  (前の文の再生中に次を送ると途中で切られるため)
- LLM出力にMarkdown記号が混じると読み上げノイズになるため、システムプロンプトで禁止した上に
  `strip_markdown()`でも防御的に除去している
