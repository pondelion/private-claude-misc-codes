"""IndexTTSのCAMPPlus版T2Sを、ローカルGPT-2/感情frontendで構築する。

外部UnifiedVoice/GPT2Model/GenerationMixinは使用しない。公式configの語彙、
境界ID、特徴次元数、層数、head数、位置埋め込みと条件配置を保持する。
この版は公式infer_v2_5が選択するCAMPPlus・semantic ID入力の経路を実装する。
別モデル族のGST等へ切り替える設定を、同じモデルとして黙って代用しない。
"""
from __future__ import annotations

from typing import Any

from torch import nn
from torch.nn import functional as F
import torch

from data_processing.tokenizer import LANGUAGE_DICT
from models.emotion.conformer import ConformerEncoder
from models.emotion.perceiver import PerceiverResampler
from models.t2s.gpt2 import GPT2Model
from models.t2s.sampling import beam_sample


class LearnedPositionEmbeddings(nn.Module):
    """本文/音声segmentの位置ごとに学習するembedding。"""

    def __init__(self, length: int, dim: int) -> None:
        """公式と同じemb.weight名で位置表[length,dim]を構築する。

        Args:
            length: 確保する位置埋め込みの個数。
            dim: 特徴の次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.emb = nn.Embedding(length, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """入力[B,T,D]またはID[B,T]のT位置に対して位置特徴[T,D]を返す。

        Args:
            hidden: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            位置埋め込み[T,D]。Tは入力の位置数、Dは特徴次元数。発話軸にはbroadcastする。
        """
        return self.emb(torch.arange(hidden.size(1), device=hidden.device))


class EmotionEncoder(nn.Module):
    """IndexTTS checkpointから読み込む感情Conformer/Perceiverの局所実装。"""

    def __init__(self, config: dict[str, Any]) -> None:
        """公式emo_condition_module全設定から感情frontendを構築する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=config["output_size"],
            linear_units=config["linear_units"],
            attention_heads=config["attention_heads"],
            num_blocks=config["num_blocks"],
            input_layer=config["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=config["output_size"],
            ff_mult=config["perceiver_mult"],
            heads=config["attention_heads"],
            num_latents=1,
        )


class UnifiedVoice(EmotionEncoder):
    """本文・話者・感情条件からsemantic tokenを生成するT2S全体。"""

    def __init__(
        self,
        layers: int = 8,
        model_dim: int = 512,
        heads: int = 8,
        max_text_tokens: int = 120,
        max_mel_tokens: int = 250,
        max_conditioning_inputs: int = 1,
        mel_length_compression: int = 1024,
        number_text_tokens: int = 256,
        start_text_token: int = 0,
        stop_text_token: int = 1,
        number_mel_codes: int = 8194,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        train_solo_embeddings: bool = False,
        use_mel_codes_as_input: bool = True,
        checkpointing: bool = True,
        types: int = 1,
        condition_num_latent: int = 32,
        condition_type: str = "perceiver",
        condition_module: dict[str, Any] | None = None,
        emo_condition_module: dict[str, Any] | None = None,
        use_accel: bool = False,
        spk_cond_mode: str = "campplus",
    ) -> None:
        """公式CAMPPlus構成をrandom初期化する。

        外部生成モデルは構築しない。

        Args:
            layers: Transformer層数。
            model_dim: GPTの特徴次元数。
            heads: attention head数。
            max_text_tokens: 本文位置埋め込みで扱う最大token数。
            max_mel_tokens: 音声位置埋め込みで扱う最大semantic token数。
            max_conditioning_inputs: 条件参照を保持できる最大個数。
            mel_length_compression: 音声長をsemantic token長へ対応させる圧縮設定。
            number_text_tokens: 本文tokenの語彙数。
            start_text_token: 本文の開始token ID。
            stop_text_token: 本文の終了token ID。
            number_mel_codes: semantic codeと特殊tokenを含む音声semantic tokenの語彙数。
            start_mel_token: 音声semantic列の開始token ID。
            stop_mel_token: 音声semantic列の終了token ID。
            train_solo_embeddings: 元実装の埋め込み学習設定。互換用に保持する。
            use_mel_codes_as_input: 音声入力を離散semantic IDとして扱うか。
            checkpointing: 元実装のgradient checkpointing設定。
            types: 言語埋め込み等で確保する条件カテゴリ数。
            condition_num_latent: 参照特徴を集約する潜在query数。
            condition_type: 参照条件の構成を選ぶ設定。
            condition_module: 参照条件encoderの構造設定。
            emo_condition_module: 感情ConformerとPerceiverの構造設定。
            use_accel: 元実装の高速化設定。本体の演算定義は変更しない。
            spk_cond_mode: 話者条件方式。対象公開モデルではcampplus。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        if spk_cond_mode != "campplus" or not use_mel_codes_as_input:
            raise ValueError(
                "This implementation targets infer_v2_5 CAMPPlus/ID input")
        if emo_condition_module is None:
            raise ValueError(
                "Official emo_condition_module configuration required")
        super().__init__(emo_condition_module)
        self.number_text_tokens = number_text_tokens
        self.number_mel_codes = number_mel_codes
        (self.start_text_token, self.stop_text_token) = (start_text_token,
                                                         stop_text_token)
        (self.start_mel_token, self.stop_mel_token) = (start_mel_token,
                                                       stop_mel_token)
        self.model_dim, self.layers, self.heads = model_dim, layers, heads
        (self.max_text_tokens, self.max_mel_tokens) = (max_text_tokens,
                                                       max_mel_tokens)
        self.spk_emb_proj = nn.Linear(192, model_dim)
        self.text_embedding = nn.Embedding(number_text_tokens * types + 1,
                                           model_dim)
        self.lang_embedding = nn.Embedding(len(LANGUAGE_DICT) + 1, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)
        self.gpt = GPT2Model(layers, model_dim, heads)
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            max_mel_tokens + 2 + max_conditioning_inputs, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(
            max_text_tokens + 2, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, number_mel_codes)
        for embedding in (self.text_embedding, self.mel_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02)
            self.text_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02)

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """条件[B,C,D]と本文ID[B,L]→fake ID[B,P+1]、prefix[B,P,D]、mask。

        B=発話数、C=条件位置数（3）、D=GPTの隠れ状態の次元数、P=C+L+2。textのBOS/EOSを除去して付け直し、
        短い本文のpaddingはprefix左側へ置く。

        Args:
            conditional_latents: 話者・感情の条件特徴[B,N,D]。Bは発話数、Nは条件slot数、
                DはGPTの隠れ状態の次元数。
            text_inputs: 本文ID[B,L]。Bは発話数、Lは本文のtoken数。
            langs: 整数の言語ID。[B]。Bは発話数。

        Returns:
            生成用の仮ID[B,P+1]、条件・本文prefix[B,P,D]、有効mask[B,P+1]の順の組。Bは発話数、
            Pは条件slotと境界token込みの本文長、DはGPTの隠れ状態の次元数。末尾の1位置は音声開始token用。
        """
        batch, length = text_inputs.shape
        prefix_length = conditional_latents.size(1) + length + 2
        embeddings, masks = [], []
        for index in range(batch):
            valid = ((text_inputs[index] != self.stop_text_token)
                     & (text_inputs[index] != self.start_text_token))
            body = text_inputs[index][valid]
            body = F.pad(F.pad(body, (1, 0), value=self.start_text_token),
                         (0, 1),
                         value=self.stop_text_token)
            # 1発話の本文ID[Lvalid]→特徴[Lvalid,D]。Lvalidは境界token込みの有効長、
            # DはGPTの隠れ状態の次元数。
            embedded = self.text_embedding(body)
            embedded = embedded + self.text_pos_embedding.emb(
                torch.arange(body.numel(), device=body.device))
            if langs is not None:
                embedded = embedded + self.lang_embedding(langs[index])
            condition = conditional_latents[0 if conditional_latents.size(0) ==
                                            1 else index]
            # 条件[N,D]と本文[Lvalid,D]を位置軸で結合する。[N+Lvalid,D]。Nは条件slot数。
            prefix = torch.cat((condition, embedded), dim=0)
            padding = prefix_length - prefix.size(0)
            # 本文長の差を左paddingでそろえる。[N+Lvalid,D]→[P,D]。Pはbatch内共通prefix長。
            prefix = F.pad(prefix, (0, 0, padding, 0))
            mask = text_inputs.new_ones(prefix_length + 1)
            mask[:padding] = 0
            embeddings.append(prefix)
            masks.append(mask)
        fake = text_inputs.new_ones((batch, prefix_length + 1))
        fake[:, -1] = self.start_mel_token
        # 発話を積む。仮ID[B,P+1]、prefix[B,P,D]、mask[B,P+1]。Bは発話数。
        return fake, torch.stack(embeddings), torch.stack(masks)

    def get_emo_conditioning(self, channel_first: torch.Tensor,
                             lengths: torch.Tensor) -> torch.Tensor:
        """参照特徴をConformerとPerceiverに通して感情特徴を抽出する。

        Args:
            channel_first: 感情参照特徴[B,1024,T]。Bは発話数、Tは参照フレーム数。
            lengths: 感情参照の有効フレーム数[B]。Bは発話数。

        Returns:
            感情埋め込み[B,1024]。Bは発話数。個数1のquery軸を取り除く。
        """
        hidden, mask = self.emo_conditioning_encoder(
            channel_first.transpose(1, 2), lengths)
        # 有効mask[B,1,Tenc]→[B,Tenc]→[B,Tenc+1]。追加1位置はPerceiver内で連結するquery用。
        mask = F.pad(mask.squeeze(1), (1, 0), value=True)
        # 参照[B,Tenc,Denc]を1個のqueryへ集約し、[B,1,1024]→[B,1024]へ変換する。
        return self.emo_perceiver_encoder(hidden, mask).squeeze(1)

    def inference_speech(
        self,
        speech_condition: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: torch.Tensor,
        emo_vec: torch.Tensor,
        campplus_embedding: torch.Tensor,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        num_beams: int = 3,
        num_return_sequences: int = 1,
        length_penalty: float = 0.0,
        repetition_penalty: float = 10.0,
        max_generate_length: int = 1500,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """実条件と本文からID[1,U]を生成する。

        外部generateメソッドは呼ばない。

        本文[1,L]、話者[1,192]、射影済み感情[1,D]、言語[1]。speech_condition[1,T50,1024]は参照特徴。
        T50は約50Hzフレーム数。CAMPPlus方式ではこの引数を再encodeせず、抽出済みの話者条件を使用する。

        Args:
            speech_condition: 参照音声特徴[1,T50,1024]。1は単一発話、T50は参照特徴長。
            text_inputs: 本文ID[B,L]。Bは発話数、Lは本文のtoken数。
            langs: 整数の言語ID。[B]。Bは発話数。
            emo_vec: GPTの隠れ状態の次元数へ射影済みの感情条件[1,D]。1は単一発話、DはGPTの隠れ状態の次元数。
            campplus_embedding: 話者埋め込み[B,192]。Bは発話数。
            do_sample: 確率samplingを使うか。Falseではscore上位を使う。
            top_p: 累積確率による候補語彙の打ち切り閾値。
            top_k: 候補語彙をscore上位何個へ制限するか。0では無効。
            temperature: samplingまたは初期ノイズの温度係数。
            num_beams: beam数。
            num_return_sequences: 返す生成候補数。
            length_penalty: 候補scoreを長さで正規化する指数。
            repetition_penalty: 既出tokenのscoreへ掛ける繰り返し抑制係数。
            max_generate_length: 生成するsemantic token数の上限。

        Returns:
            音声token ID[1,A]と射影済み話者条件[1,1,D]の組。Aは終了tokenを含み得る生成長、DはGPTの隠れ状態の次元数。
            発話・候補数は1。
        """
        if num_return_sequences != 1:
            raise ValueError("Pipeline returns one utterance")
        # 話者[1,192]をGPTの隠れ状態の次元数へ射影し、条件slot軸を追加する。[1,1,D]。
        speaker = self.spk_emb_proj(campplus_embedding)[:, None]
        # 感情[1,D]を話者slotへ加算し、後続に2個のゼロslotを置く。[1,3,D]。
        condition = torch.cat(
            (speaker + emo_vec[:, None],
             speaker.new_zeros(speaker.size(0), 2, self.model_dim)),
            dim=1)
        initial, prefix, mask = self.prepare_gpt_inputs(
            condition, text_inputs, langs)
        past = None
        previous_history = None

        def step(history: torch.Tensor) -> torch.Tensor:
            """履歴[K,P+A]からKV cacheを更新し、最終logits[K,V]を計算する。

            Args:
                history: 生成履歴[K,P+A]。Kはbeam数、Pは開始列長、Aは生成済み長。

            Returns:
                次tokenのlogits[K,V]。Kはbeam数、Vは音声semantic tokenの語彙数。
            """
            nonlocal past, previous_history
            audio_ids = history[:, prefix.
                                size(1):] if past is None else history[:, -1:]
            # 履歴中の音声ID[K,Ain]→音声特徴[K,Ain,D]。
            # Kはbeam数、Ainは初回の音声長またはcache更新時の1位置。
            audio = self.mel_embedding(audio_ids)
            valid = F.pad(mask[:, :-1], (0, history.size(1) - prefix.size(1)),
                          value=1)
            if past is None:
                # 初回は条件/本文prefixとBOSを一度に計算する。
                audio = audio + self.mel_pos_embedding(audio)
                # 本文・条件prefix[1,P,D]をbeamごとに共有する。[K,P,D]。
                context = prefix.expand(history.size(0), -1, -1)
                # prefixと現在の音声を位置軸で結合する。[K,P+Ain,D]。
                embedded = torch.cat((context, audio), dim=1)
            else:
                # beamの親履歴を照合し、対応する過去KVを同じ順序へ並べ替える。
                parents = (
                    history[:, None, :-1] == previous_history[None]).all(-1)
                order = parents.long().argmax(-1)
                past = [(key.index_select(0,
                                          order), value.index_select(0, order))
                        for key, value in past]
                # 公開GPT2InferenceModelのcached経路と同じ位置indexを使う。
                position = valid.size(1) - prefix.size(1)
                embedded = audio + self.mel_pos_embedding.emb(
                    history.new_tensor([position]))[None]
            output = self.gpt(embedded, valid.expand(history.size(0), -1),
                              past)
            past = output.past_key_values
            previous_history = history.clone()
            return self.mel_head(
                self.final_norm(output.last_hidden_state[:, -1]))

        codes = beam_sample(step, initial, self.stop_mel_token,
                            max_generate_length, num_beams, temperature, top_k,
                            top_p, repetition_penalty, length_penalty,
                            do_sample)
        return codes, speaker
