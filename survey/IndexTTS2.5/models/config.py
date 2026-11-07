"""公式configを省略せず読み、モデル規模だけを変更したvariantを構築する。

configの例:
    {"official_config_path": "checkpoints/IndexTTS-2.5/config.yaml"}
    {"official_config_path": "checkpoints/IndexTTS-2.5/config.yaml",
     "variant": {"layers": 8, "model_dim": 512, "heads": 8}}
    variantは対象stageの容量fieldだけを指定する。未指定fieldは公式値を保持。
    チェックポイント保存時は{"official": 完全な設定辞書}へ展開して保存し、
    元のconfig.yamlのpathに依存せず同じモデルを再構築できるようにする。
    ここではモデルをimport/構築せず、設定ファイルだけを処理する。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

CAPACITY_FIELDS = {
    "gpt": {"layers", "model_dim", "heads"},
    "semantic_codec":
    {"vocos_dim", "vocos_intermediate_dim", "vocos_num_layers"},
    "s2mel": {
        "DiT.hidden_dim", "DiT.depth", "DiT.num_heads", "DiT.content_dim",
        "length_regulator.channels"
    },
}


def resolve_config(section: str, spec: dict[str, Any]) -> dict[str, Any]:
    """公式section全体を読み、明示された容量variantだけを適用して返す。

    sectionはgpt/semantic_codec/s2mel。従来の独自の小型設定を公式設定と誤認しないよう、設定元がない場合はエラーにする。
    既定値を推測しない。

    Args:
        section: 読み込む公式設定section名。
        spec: 公式設定の参照先、または展開済み設定と容量variant。

    Returns:
        公式の対象section全体へ、許可した容量variantを反映した辞書。
    """
    if section not in CAPACITY_FIELDS:
        raise ValueError(f"Unknown official config section: {section}")
    unknown = set(spec) - {"official", "official_config_path", "variant"}
    if unknown:
        raise ValueError(
            f"Regenerate config from official YAML; unknown keys: {unknown}")
    if ("official" in spec) == ("official_config_path" in spec):
        raise ValueError(
            "Specify exactly one of official or official_config_path")
    if "official" in spec:
        config = deepcopy(spec["official"])
    else:
        from omegaconf import OmegaConf

        source = OmegaConf.load(Path(spec["official_config_path"]))
        config = OmegaConf.to_container(source[section], resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Official section must be a mapping")
    for key, value in spec.get("variant", {}).items():
        if key not in CAPACITY_FIELDS[section]:
            raise ValueError(f"Only capacity fields may change: {key}")
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Capacity must be a positive integer: {key}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Field not present in official config: {key}")
        target[parts[-1]] = value
    return config


def stage_config(stage: str, spec: dict[str, Any]) -> dict[str, Any]:
    """学習stage名から公式sectionを選び、保存可能な完全設定を返す。

    Args:
        stage: codec、t2s、s2melのいずれか。
        spec: 公式設定の参照先、または展開済み設定と容量variant。

    Returns:
        stageに対応するモデルの解決済み設定辞書。
    """
    section = {
        "codec": "semantic_codec",
        "t2s": "gpt",
        "s2mel": "s2mel"
    }[stage]
    return {"official": resolve_config(section, spec)}
