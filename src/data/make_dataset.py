"""Compatibility wrapper. Prefer `src.cli.save_processed_data` or `src.compat`."""

from __future__ import annotations

from src.compat.dataset_export import export_processed_dataset_legacy


def main() -> None:
    export_processed_dataset_legacy()


if __name__ == "__main__":
    main()
