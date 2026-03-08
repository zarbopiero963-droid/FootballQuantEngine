from __future__ import annotations

from quant.fusion.team_mapping_table import DEFAULT_TEAM_MAPPING
from quant.fusion.team_name_normalizer import TeamNameNormalizer


class TeamMapper:

    def __init__(self, mapping: dict | None = None):
        self.normalizer = TeamNameNormalizer()
        self.mapping = {}

        base_mapping = dict(DEFAULT_TEAM_MAPPING)
        if mapping:
            base_mapping.update(mapping)

        for key, value in base_mapping.items():
            self.mapping[self.normalizer.normalize(key)] = self.normalizer.normalize(
                value
            )

    def map_name(self, source_name: str) -> str:
        normalized = self.normalizer.normalize(source_name)
        return self.mapping.get(normalized, normalized)

    def build_lookup(self, names: list[str] | tuple[str, ...]) -> dict:
        lookup = {}
        for name in names:
            normalized = self.normalizer.normalize(name)
            mapped = self.map_name(name)
            lookup[normalized] = mapped
        return lookup
