from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORT_PATH = Path("reports/executive_report.json")
PRIORITIZATION_PATH = Path("data/gold/customer_prioritization.csv")
KPI_PATH = Path("data/gold/kpi_summary.csv")


def load_executive_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    with open(REPORT_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_prioritization() -> pd.DataFrame:
    if not PRIORITIZATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PRIORITIZATION_PATH)


def load_kpis() -> pd.DataFrame:
    if not KPI_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(KPI_PATH)
