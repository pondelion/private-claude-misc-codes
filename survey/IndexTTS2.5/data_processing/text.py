"""本文cleaning・発音注釈保護・辞書置換をローカル実装する。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
入力/出力は本文文字列。固定FSTによる読み展開以外を外部IndexTTSへ委譲しない。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import os
import re
import traceback


def apply_pronunciation_annotations(text: str) -> str:
    """<本文|発音>注釈を、公式語彙の発音special tokenを含む文字列へ変換する。

    Args:
        text: 読み上げる本文。

    Returns:
        注釈を発音special tokenへ展開した文字列。
    """

    def replace(match: re.Match[str]) -> str:
        """一致した発音注釈を発音token列へ置き換える。

        Args:
            match: 正規表現の一致オブジェクト。

        Returns:
            発音tokenへ展開した文字列。
        """
        word, pronunciation = match.group(1), match.group(2).upper()
        if re.fullmatch(r"[\u3040-\u30ff\u31f0-\u31ff\uff65-\uff9f\s]+",
                        pronunciation):
            return f" {pronunciation} "
        index = 2 if re.search(r"[\u4e00-\u9fff]", word) else 1
        token = f"<|SPECIAL_TOKEN_{index}|>"
        return token + pronunciation + token

    return re.sub(r"<([^|>\n]+)\|([^>\n]+)>", replace, text)


def normalize_other_language(text: str, language: str,
                             assets_dir: Path) -> str:
    """日本語は公式と同じくTN非対応、スペイン語は固定FSTモデルで読み展開する。

    外部NeMo/IndexTTS関数へ処理を委譲しない。スペイン語の読み展開に必要なFSTモデルがない場合、
    数字を無断で読み替えたり素通ししたりせずエラーとする。

    Args:
        text: 読み上げる本文。
        language: 読み上げ言語コードまたはモデル用言語ID。
        assets_dir: 言語ごとの固定正規化FSTを配置したディレクトリ。

    Returns:
        言語別の固定ルールを適用した本文。
    """
    if language == "ja":
        return text
    from data_processing.fst import FrozenFstNormalizer
    return FrozenFstNormalizer(assets_dir / language).normalize(text)


def tokenize_by_CJK_char(line: str, do_upper_case: bool = True) -> str:
    """CJK文字を1文字ずつ分割し、英単語を保持して空白区切りにする。

    Args:
        line: CJK文字の分割・結合を行う文字列。
        do_upper_case: 英字を大文字にそろえるか。

    Returns:
        CJK文字を分割し、空白で区切った文字列。
    """
    # CJKのUnicode範囲はNLTKのtokenize.utilの定義に基づく。
    CJK_RANGE_PATTERN = (r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF"
                         r"\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC"
                         r"\U00020000-\U0002FFFF])")
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([
        w.strip().upper() if do_upper_case else w.strip() for w in chars
        if w.strip()
    ])


def de_tokenized_by_CJK_char(line: str, do_lower_case: bool = False) -> str:
    """CJK文字間の分割用空白を除き、英単語の区切りを復元する。

    Args:
        line: CJK文字の分割・結合を行う文字列。
        do_lower_case: 英字を小文字にそろえるか。

    Returns:
        CJK文字間の分割用空白を除いた文字列。
    """
    # 英単語を一時的な識別文字列へ置き換えてCJK分割から保護する。
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s'-][A-Z-]+)*)",
                                      re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # 保護した英語の文を復元する。
    sent_placeholder_pattern = re.compile(r"(<sent_(\d+)>)")
    for i in range(len(words)):
        all_matches = sent_placeholder_pattern.findall(words[i])
        if len(all_matches) > 1:
            # 保護した英単語を復元する。
            for h, j in all_matches:
                placeholder_index = int(j)
                words[i] = words[i].replace(h,
                                            english_sents[placeholder_index])
                if do_lower_case:
                    words[i] = words[i].lower()
    return "".join(words)


class TextNormalizer:
    """本文の注釈保護、語彙置換、固定FSTによる読み展開を処理する。
    """

    def __init__(self,
                 enable_glossary: bool = False,
                 assets_dir: Any = None) -> None:
        """本文の注釈保護、語彙置換、固定FSTによる読み展開を処理するための層と設定を構築する。

        Args:
            enable_glossary: 用語辞書による読みの置換を有効にするか。
            assets_dir: 言語ごとの固定正規化FSTを配置したディレクトリ。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.assets_dir = Path(assets_dir) if assets_dir is not None else Path(
            "normalization")
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.clean_pattern = re.compile("|".join(
            re.escape(p) for p in self.char_rep_map.keys()))
        self.enable_glossary = enable_glossary
        # 利用者が専門用語の読みを登録する用語辞書。
        # 形式は{用語: {"en": 英語の読み, "zh": 中国語の読み}}。

        self.term_glossary = dict()

    def match_email(self, email: str) -> bool:
        # 英数字のユーザー名・ドメインからなるメールアドレスを照合する。
        """文字列がメールアドレス形式に一致するかをboolで返す。

        Args:
            email: 形式を判定するメールアドレス文字列。

        Returns:
            メールアドレス形式に一致すればTrue。
        """
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = (
        r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?"
        r"(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|"
        r"uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])")
    # 声調数字1〜5を伴うpinyinの照合規則。5は軽声を表す。xuan4等を対象とし、一般の英単語に数字が続く形は対象外。
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    # 中点やハイフンを含む中国語表記の人名を照合する規則。

    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    # 英字から始まるハイフン区切りの技術用語を照合する。数字だけの電話番号等は対象外とし、ハイフンを減算記号として読み展開しないよう保護する。

    # 頻出する'sの短縮形をisへ展開する。すべての所有格を置換するわけではない。
    ENGLISH_CONTRACTION_PATTERN = (
        r"(what|where|who|which|how|t?here|it|s?he|that|this)'s")

    def use_chinese(self, s: str) -> bool:
        """漢字、英字、メール、声調付きpinyinの有無から中国語正規化の使用を判定する。

        Args:
            s: 言語判定対象の文字列。

        Returns:
            中国語用の正規化を選ぶ場合はTrue。
        """
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(
            re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self) -> None:
        """固定FSTモデルを遅延読込する。

        外部normalizerクラスへ処理委譲しない。

        Returns:
            None。固定FSTモデルをインスタンスへ登録する。
        """
        from data_processing.fst import FrozenFstNormalizer
        if self.zh_normalizer is None:
            self.zh_normalizer = FrozenFstNormalizer(self.assets_dir / "zh")
        if self.en_normalizer is None:
            self.en_normalizer = FrozenFstNormalizer(self.assets_dir / "en")

    G2P_PRONUNCIATION_ANNOTATION_PATTERN = re.compile(
        r'<([^|>\n]+)\|([^>\n]+)>')

    def _protect_pronunciation_annotations(
            self, text: str) -> tuple[str, dict[str, str]]:
        """発音注釈を仮文字列に置換し、正規化中の書き換えから保護する。

        Args:
            text: 読み上げる本文。

        Returns:
            保護後の本文と、仮文字列から元の注釈への対応辞書の組。
        """
        placeholders = {}

        def _idx_to_alpha(n: int) -> str:
            """非負の連番を仮文字列に使用するアルファベット列へ変換する。

            Args:
                n: 仮文字列へ変換する非負の連番。

            Returns:
                仮文字列に使用するアルファベット列。
            """
            s = ''
            while True:
                s = chr(ord('a') + n % 26) + s
                n = n // 26 - 1
                if n < 0:
                    break
            return s

        def _replacer(m: Any) -> str:
            """正規表現で一致した文字列を保護用の仮文字列へ置き換える。

            Args:
                m: 置換対象に一致した正規表現のMatch。

            Returns:
                対応する仮文字列。
            """
            tag = _idx_to_alpha(len(placeholders))
            key = f'PRONPLACEHOLDER{tag}PRONPLACEHOLDER'
            placeholders[key] = m.group(0)
            return key

        text = self.G2P_PRONUNCIATION_ANNOTATION_PATTERN.sub(_replacer, text)
        return text, placeholders

    @staticmethod
    def _restore_pronunciation_annotations(text: str,
                                           placeholders: dict) -> str:
        """保護用の仮文字列を元の発音注釈へ戻す。

        Args:
            text: 読み上げる本文。
            placeholders: 保護用の仮文字列から元の発音注釈への対応辞書。

        Returns:
            保護前の発音注釈を復元した文字列。
        """
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def normalize(self, text: str) -> str:
        """発音注釈等を保護し、固定文法モデルで読み展開してから保護した文字列を復元する。

        Args:
            text: 読み上げる本文。

        Returns:
            読み展開・文字正規化後の本文文字列。
        """
        if not self.zh_normalizer or not self.en_normalizer:
            self.load()
        # <word|pronunciation>の発音注釈を正規化による変更から保護する。
        text, _pron_placeholders = self._protect_pronunciation_annotations(
            text)
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN,
                          r"\1 is",
                          text,
                          flags=re.IGNORECASE)
            # 他の保護処理より先に用語辞書の読みを適用する。
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # 技術用語のハイフンを、中国語正規化で減算記号と誤解しないよう保護する。
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 人名を保護前の表記へ戻す。
            result = self.restore_names(result, original_name_list)
            # pinyinの声調数字を復元する。
            result = self.restore_pinyin_tones(result, pinyin_list)
            # 技術用語のハイフンを復元する。
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(
                re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()],
                                 result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN,
                              r"\1 is",
                              text,
                              flags=re.IGNORECASE)
                # 他の保護処理より先に用語辞書の読みを適用する。
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # 技術用語のハイフンを英語正規化による意図しない変更から保護する。
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # 技術用語のハイフンを復元する。
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(
                re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()],
                                 result)

        # G2P用の発音注釈を復元する。
        result = self._restore_pronunciation_annotations(
            result, _pron_placeholders)
        return result

    def correct_pinyin(self, pinyin: str) -> str:
        """j・q・xに続くuまたはüを発音token用のv表記へ変換する。

        Args:
            pinyin: 声調を含むpinyinの文字列。

        Returns:
            発音tokenで使用するv表記へ変換したpinyin。
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # j・q・xに続くuまたはüを照合する。
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text: str) -> tuple[str, list[str]]:
        """人名を仮文字列で保護し、復元用の人名リストを作る。

        Args:
            original_text: 正規化前の本文。

        Returns:
            保護後の本文と、保護前の人名リストの組。
        """
        # 処理中に読み展開から保護する人名。
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 人名を<n_a>、<n_b>等の仮文字列へ置き換える。
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text: str,
                      original_name_list: list[str]) -> str:
        """人名の仮文字列を保護前の表記へ戻す。

        Args:
            normalized_text: 正規化後に保護文字列を復元する本文。
            original_name_list: 保護時に記録した人名のリスト。

        Returns:
            人名の表記を復元した本文。
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 人名の仮文字列を元の表記へ戻す。
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_tech_terms(self, original_text: str) -> tuple[str, list[str]]:
        """技術用語のハイフンを保護し、復元用の用語リストを作る。

        Args:
            original_text: 正規化前の本文。

        Returns:
            ハイフンを保護した本文と、保護対象の技術用語リストの組。
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)

        # 重複を除き、長い用語から照合して部分一致の先行置換を防ぐ。
        original_tech_list = sorted(set(original_tech_list),
                                    key=len,
                                    reverse=True)
        transformed_text = original_text

        # 技術用語内のハイフンを保護用の<H>へ置き換える。
        for term in original_tech_list:
            # 例としてGPT-5-nanoをGPT<H>5<H>nanoへ変換する。
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)

        return transformed_text, original_tech_list

    def restore_tech_terms(self, normalized_text: str,
                           original_tech_list: list[str]) -> str:
        """保護したハイフンを復元し、仮文字列周囲の余分な空白を除く。

        Args:
            normalized_text: 正規化後に保護文字列を復元する本文。
            original_tech_list: 保護時に記録した技術用語のリスト。

        Returns:
            技術用語のハイフンを復元した本文。
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text

        # <H>周囲の空白を取り除いてからハイフンへ戻す。
        # 仮文字列の前後いずれかに空白が挿入された場合も同じ表記へ戻す。
        transformed_text = re.sub(r'\s*<H>\s*', '-', normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text: Any, lang: str = 'zh') -> str:
        """用語辞書の言語別の読みに従って本文を置換する。

        Args:
            text: 読み上げる本文。
            lang: 用語辞書の読みを選ぶ言語コード。

        Returns:
            用語の読みを置換した本文。
        """
        if not self.term_glossary:
            return text

        # 長い用語を先に照合し、その一部だけを短い用語として置換しないようにする。
        # 例としてPCIe 5.0をPCIeより先に照合する。
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)

        @lru_cache(maxsize=42)
        def get_term_pattern(term: str) -> str:
            """用語を大小文字を区別せず照合するための正規表現を作る。

            Args:
                term: 正規表現へ変換する用語。

            Returns:
                対象用語の照合に使う正規表現文字列。
            """
            return re.compile(re.escape(term), re.IGNORECASE)

        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            # 大文字・小文字を区別せずに用語を置換する。
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)

        return transformed_text

    def load_glossary(self, glossary_dict: dict[str,
                                                str | dict[str, str]]) -> None:
        """用語から読みへの対応辞書を登録する。

        Args:
            glossary_dict: 用語から言語別の読みへの対応辞書。

        Returns:
            None。用語辞書をインスタンスへ登録する。
        """
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path: str) -> None:
        """YAMLファイルから用語辞書を読み込む。

        Args:
            glossary_path: 用語辞書YAMLのパス。

        Returns:
            None。用語辞書をインスタンスへ登録する。
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, 'r', encoding='utf-8') as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path: str) -> None:
        """登録した用語辞書をYAMLへ保存する。

        Args:
            glossary_path: 用語辞書YAMLのパス。

        Returns:
            None。YAMLファイルへ辞書を書き込む。
        """
        import yaml
        with open(glossary_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.term_glossary,
                      f,
                      allow_unicode=True,
                      default_flow_style=False)

    def save_pinyin_tones(self, original_text: str) -> tuple[str, list[str]]:
        """声調付きpinyinを仮文字列に置換して保護する。

        Args:
            original_text: 正規化前の本文。

        Returns:
            保護後の本文と、声調付きpinyinリストの組。
        """
        # 声母・韻母に声調数字が続くpinyinを照合する。
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN,
                                           re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(
            set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 声調付きpinyinを<pinyin_a>等の仮文字列へ置き換える。
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(
                pinyin, f"<pinyin_{number}>")

        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text: str,
                             original_pinyin_list: list[str]) -> str:
        """声調付きpinyinの仮文字列を元の表記へ戻す。

        Args:
            normalized_text: 正規化後に保護文字列を復元する本文。
            original_pinyin_list: 保護時に記録した声調付きpinyinのリスト。

        Returns:
            声調付きpinyinを復元した本文。
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # pinyinの仮文字列を元の声調付き表記へ戻す。
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(
                f"<pinyin_{number}>", pinyin)

        return transformed_text
