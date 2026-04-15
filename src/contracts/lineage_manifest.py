from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LineageInput:
    raw_path: str
    raw_sha256: str


@dataclass(frozen=True)
class LineageArtifact:
    name: str
    path: str
    format: str
    sha256: str | None
    rows: int | None = None


@dataclass(frozen=True)
class LineageManifest:
    schema_version: str
    lineage_version: str
    generated_at_utc: str
    run_id: str
    environment: str
    input: LineageInput
    stages: dict[str, dict[str, object]]
    artifacts: list[LineageArtifact]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
