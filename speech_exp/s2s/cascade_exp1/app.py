"""カスケード方式S2S(ASR: Parakeet -> LLM: Ollama/Qwen3 -> TTS: VOICEVOX)の検証用Gradioアプリ。

マイクで話しかけると:
  1. Parakeetで文字起こし(ASR結果を即表示)
  2. Ollama(Qwen3:8b)にストリーミングで応答生成させ、生成中のテキストをリアルタイム表示
  3. 生成された文はできた端からVOICEVOXで音声合成(文単位パイプライン化、LLMの続きの生成と並行実行)
  4. 各ステージの所要時間を計測して表示(ASR/LLM初回トークン/最初の1文/TTS/合計)

起動:
    uv run python app.py
LAN内の別PCから http://<このPCのIP>:7860 でアクセスできる。
"""

import time

import gradio as gr

from pipeline import TurnMetrics, run_turn, transcribe


def process_turn(audio_path, history_state, chat_display):
    if audio_path is None:
        yield history_state, chat_display, "", "", None, "(録音がありません)"
        return

    metrics = TurnMetrics()
    total_start = time.perf_counter()

    user_text, asr_sec = transcribe(audio_path)
    metrics.asr_sec = asr_sec

    if not user_text:
        yield history_state, chat_display, "(認識できませんでした)", "", None, metrics.as_text()
        return

    chat_display = chat_display + [{"role": "user", "content": user_text}]
    yield history_state, chat_display, user_text, "", None, metrics.as_text()

    llm_text_display = ""
    chat_display = chat_display + [{"role": "assistant", "content": ""}]

    # 前の文の再生が終わるはずの時刻。次の文はこれより後にならないと送らない
    # (audio_outに新しい値を渡すとブラウザは即座に頭出し再生するため、
    #  再生中に次を送ると前の文が途中で切られてしまう)。
    next_playable_at = 0.0

    for kind, payload in run_turn(user_text, history_state, metrics):
        if kind == "llm_text":
            llm_text_display = payload
            chat_display[-1]["content"] = llm_text_display
            yield history_state, chat_display, user_text, llm_text_display, None, metrics.as_text()

        elif kind == "audio_chunk":
            data, sr = payload
            duration_sec = len(data) / sr

            wait = next_playable_at - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

            yield (
                history_state,
                chat_display,
                user_text,
                llm_text_display,
                (sr, data),
                metrics.as_text(),
            )
            next_playable_at = time.perf_counter() + duration_sec

        elif kind == "final":
            full_text = payload[0]
            metrics.total_sec = time.perf_counter() - total_start
            chat_display[-1]["content"] = full_text
            history_state = history_state + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": full_text},
            ]
            # 音声は直前の audio_chunk で最後の文まで再生済みなので、ここで再度流すと
            # 頭から再生し直されてしまう。audio_outは更新しない(gr.skip())。
            yield (
                history_state,
                chat_display,
                user_text,
                full_text,
                gr.skip(),
                metrics.as_text(),
            )

        elif kind == "error":
            chat_display[-1]["content"] = f"[エラー] {payload}"
            yield history_state, chat_display, user_text, f"[エラー] {payload}", None, metrics.as_text()


def reset_conversation():
    return [], [], "", "", None, ""


with gr.Blocks(title="カスケードS2S検証 (Parakeet + Qwen3 + VOICEVOX)") as demo:
    gr.Markdown(
        "## カスケードS2S検証\n"
        "**ASR: Parakeet TDT-CTC(ja) → LLM: Ollama Qwen3:8b(ストリーミング) → TTS: VOICEVOX**\n\n"
        "マイクで話しかけて録音を止めると自動で処理が始まります。"
    )

    history_state = gr.State([])  # Ollamaに渡す会話履歴 [{"role": ..., "content": ...}, ...]

    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(sources=["microphone"], type="filepath", label="話しかけてください")
            reset_btn = gr.Button("会話をリセット")

            asr_box = gr.Textbox(label="ASR認識結果(今回の発話)", interactive=False)
            llm_box = gr.Textbox(label="LLM応答(ストリーミング表示)", interactive=False, lines=4)
            audio_out = gr.Audio(label="合成音声(文ごとに逐次再生)", autoplay=True)
            metrics_box = gr.Textbox(
                label="レイテンシ計測", interactive=False, lines=8, elem_id="metrics"
            )

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="会話履歴", height=560)

    mic.stop_recording(
        fn=process_turn,
        inputs=[mic, history_state, chatbot],
        outputs=[history_state, chatbot, asr_box, llm_box, audio_out, metrics_box],
    )

    reset_btn.click(
        fn=reset_conversation,
        outputs=[history_state, chatbot, asr_box, llm_box, audio_out, metrics_box],
    )


if __name__ == "__main__":
    # LAN内の別PCからマイク(getUserMedia)を使うにはHTTPSが必須(ブラウザのセキュアコンテキスト要件)。
    # 自己署名証明書は openssl_san.cnf から生成(SANにLAN IPを含めてある)。
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
        ssl_verify=False,
    )
