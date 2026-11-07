"""固定FSTモデルを読み、文字列の分類・読み展開を行う局所実装。

外部WeText/NeMo normalizerクラスを呼ばない。pyniniは固定FSTモデルを実行する
基盤としてのみ使う。英語/中国語の文法FSTは学習済み/構築済み資産として別途必要。
各言語のtagger.fstとverbalizer.fstをnormalization/<language>/に配置する。
未準備時に数字をそのまま通したり、独自の簡易読みへ置換したりはしない。
"""
from __future__ import annotations

from pathlib import Path


class FrozenFstNormalizer:
    """固定された分類FSTと読み上げFSTを使うtext-to-textモデル。"""

    def __init__(self, directory: Path) -> None:
        """ローカルのFST重みだけを読む。

        文法の自動取得・自動再構築は行わない。

        Args:
            directory: tagger.fstとverbalizer.fstを配置したディレクトリ。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        import pynini

        paths = [directory / "tagger.fst", directory / "verbalizer.fst"]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing fixed normalization model: {path}")
        self.tagger = pynini.Fst.read(str(paths[0]))
        self.verbalizer = pynini.Fst.read(str(paths[1]))

    def normalize(self, text: str) -> str:
        """分類FST→最短経路→読み展開FST→最短経路の順に文字列を変換する。

        Args:
            text: 読み上げる本文。

        Returns:
            読み展開・文字正規化後の本文文字列。
        """
        import pynini

        tagged = pynini.shortestpath(
            pynini.compose(pynini.escape(text), self.tagger))
        if tagged.start() == pynini.NO_STATE_ID:
            raise ValueError("Normalizer tagger has no path for this text")
        verbalized = pynini.shortestpath(
            pynini.compose(tagged, self.verbalizer))
        if verbalized.start() == pynini.NO_STATE_ID:
            raise ValueError(
                "Normalizer verbalizer has no path; check matching FST assets")
        return verbalized.string()
