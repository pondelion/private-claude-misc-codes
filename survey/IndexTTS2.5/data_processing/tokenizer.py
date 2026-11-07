"""配布されたtiktoken語彙を読む本文tokenizer。外部IndexTTS処理には依存しない。

言語・音響event・感情・TTS・timestamp tokenの順序は公式定義に合わせる。
tiktokenは許可されたpretrained tokenizerの実行基盤として使用し、語彙を学び直さない。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import base64

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
    "common": "common",
}

LANGUAGE_DICT = {lang: index for index, lang in enumerate(LANGUAGES.keys())}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{
        f"TTS/SP{i:02d}": f"TTS/SP{i:02d}"
        for i in range(1, 14)
    }
}


def lang_to_token(language: str) -> int:
    """言語コードから公式と同じ整数IDを返す。

    不明な値はcommonへ対応させる。

    Args:
        language: 読み上げ言語コードまたはモデル用言語ID。

    Returns:
        公式の言語埋め込み表に対応する整数ID。
    """
    return LANGUAGE_DICT.get(language.lower(), LANGUAGE_DICT["common"])


class TextTokenizer:
    """実語彙ファイルとspecial token表で本文をID化する固定tokenizer。"""

    def __init__(self, model_dir: str | Path) -> None:
        """ローカルの.tiktokenを読み、順位・special ID・正規表現を設定する。

        Args:
            model_dir: 事前配置した公開重みと設定のディレクトリ。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        import tiktoken

        path = Path(model_dir) / "multilingual_zh_ja_yue_char_del.tiktoken"
        with path.open(encoding="utf-8") as handle:
            ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in handle
                                    if line.strip())
            }
        tokens = [
            "<|endoftext|>", "<|startoftranscript|>",
            *[f"<|{name}|>" for name in list(LANGUAGES)[:99]],
            *[f"<|{name}|>"
              for name in AUDIO_EVENT], *[f"<|{name}|>" for name in EMOTION],
            "<|translate|>", "<|transcribe|>", "<|startoflm|>",
            "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
            *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
            *[f"<|{name}|>" for name in TTS_Vocal_Token],
            *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]
        ]
        special = {
            token: len(ranks) + index
            for index, token in enumerate(tokens)
        }
        self.encoding = tiktoken.Encoding(
            name=path.name,
            explicit_n_vocab=len(ranks) + len(tokens),
            pat_str=r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
            r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            mergeable_ranks=ranks,
            special_tokens=special)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """本文文字列をpretrained語彙のID列[L]へ変換する。

        L=token数。

        Args:
            text: 読み上げる本文。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            本文token IDの整数リスト[L]。Lは本文のtoken数。
        """
        return self.encoding.encode(text, **kwargs)
