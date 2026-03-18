from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    path: str
    format: str
    rows: int | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    artifact_type: str
    generated_at_utc: str
    run_id: str
    environment: str
    entries: list[ArtifactEntry]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
