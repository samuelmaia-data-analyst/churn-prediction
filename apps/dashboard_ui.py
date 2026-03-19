from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st

from src.runtime.config import PipelineConfig

COLOR_BG_START = "#f7f9fc"
COLOR_BG_END = "#eef3f9"
COLOR_PRIMARY = "#164e63"
COLOR_SECONDARY = "#0f766e"
COLOR_ALERT = "#dc2626"
COLOR_TEXT = "#0b1f33"
COLOR_MUTED = "#5b6b80"


def configure_dashboard_page(*, page_title: str, page_icon: str) -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
        :root {{
            --bg-start: {COLOR_BG_START};
            --bg-end: {COLOR_BG_END};
            --primary: {COLOR_PRIMARY};
            --secondary: {COLOR_SECONDARY};
            --alert: {COLOR_ALERT};
            --text: {COLOR_TEXT};
            --muted: {COLOR_MUTED};
            --surface: rgba(255, 255, 255, 0.78);
            --surface-strong: rgba(255, 255, 255, 0.92);
            --border: rgba(22, 78, 99, 0.14);
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 6%, rgba(22, 78, 99, 0.10), transparent 36%),
                radial-gradient(circle at 85% 0%, rgba(15, 118, 110, 0.12), transparent 38%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
            color: var(--text);
        }}
        html, body, [class*="css"] {{
            font-family: "Space Grotesk", sans-serif;
        }}
        h1, h2, h3,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2 {{
            color: var(--text);
            letter-spacing: -0.02em;
        }}
        [data-testid="stHeaderActionElements"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(
                200deg,
                rgba(22, 78, 99, 0.96) 0%,
                rgba(11, 31, 51, 0.97) 100%
            );
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc;
        }}
        [data-testid="stSidebar"] a {{
            color: #99f6e4 !important;
        }}
        [data-testid="stSidebarNav"] {{
            padding-top: 0.4rem;
        }}
        .hero {{
            background: linear-gradient(125deg, rgba(22, 78, 99, 0.96), rgba(15, 118, 110, 0.92));
            border-radius: 22px;
            padding: 1.3rem 1.45rem;
            box-shadow: 0 16px 44px rgba(11, 31, 51, 0.22);
            margin-bottom: 1rem;
        }}
        .hero-eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            color: rgba(153, 246, 228, 0.92);
        }}
        .hero-title {{
            color: #f8fafc;
            margin: 0.2rem 0 0 0;
            font-size: clamp(1.55rem, 4.7vw, 2.45rem);
            font-weight: 700;
            line-height: 1.1;
        }}
        .hero-subtitle {{
            color: rgba(248, 250, 252, 0.92);
            margin-top: 0.55rem;
            font-size: 0.98rem;
            max-width: 58rem;
        }}
        .hero-meta {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.9rem;
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            background: rgba(255, 255, 255, 0.10);
            color: #f8fafc;
            font-size: 0.85rem;
        }}
        .status-banner {{
            border-radius: 16px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--border);
            background: var(--surface);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.9rem;
        }}
        .status-banner strong {{
            display: block;
            color: var(--primary);
            margin-bottom: 0.18rem;
        }}
        .section-block {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.95rem;
        }}
        .section-title {{
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0 0 0.15rem 0;
        }}
        .section-copy {{
            color: var(--muted);
            font-size: 0.94rem;
            margin-bottom: 0.65rem;
        }}
        [data-testid="stMetric"] {{
            background: var(--surface-strong);
            border: 1px solid rgba(22, 78, 99, 0.16);
            border-radius: 16px;
            padding: 0.62rem 0.78rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--muted);
            font-weight: 600;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--primary);
            font-size: 1.55rem;
            font-weight: 700;
        }}
        .sidebar-card {{
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.75rem;
            background: rgba(255,255,255,0.06);
        }}
        .sidebar-label {{
            color: rgba(248, 250, 252, 0.74);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }}
        .sidebar-value {{
            color: #f8fafc;
            font-size: 0.95rem;
            margin-top: 0.22rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.6rem;
            margin-bottom: 0.9rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 0.5rem 0.95rem;
            border: 1px solid rgba(22, 78, 99, 0.14);
            background: rgba(255, 255, 255, 0.72);
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(22, 78, 99, 0.12);
        }}
        .footer-note {{
            text-align: center;
            color: #355070;
            padding: 0.45rem 0 0.9rem 0;
            font-size: 0.92rem;
        }}
        code {{
            font-family: "JetBrains Mono", monospace !important;
        }}
        @media (max-width: 820px) {{
            .hero {{
                border-radius: 16px;
                padding: 1rem 1rem;
            }}
            .section-block {{
                border-radius: 15px;
                padding: 0.9rem 0.85rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(
    *,
    eyebrow: str,
    title: str,
    subtitle: str,
    meta: str | None = None,
) -> None:
    meta_html = f'<div class="hero-meta">{meta}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(
    *,
    title: str,
    body: str,
    status_items: Iterable[tuple[str, str]],
) -> None:
    items = list(status_items)
    st.markdown('<div class="status-banner">', unsafe_allow_html=True)
    st.markdown(f"<strong>{title}</strong>{body}", unsafe_allow_html=True)
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)
    st.markdown("</div>", unsafe_allow_html=True)


def section_container(title: str, copy: str):
    container = st.container(border=True)
    with container:
        st.markdown(
            f"""
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
            """,
            unsafe_allow_html=True,
        )
    return container


def render_sidebar_summary(config: PipelineConfig) -> None:
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Environment</div>
            <div class="sidebar-value">{config.environment}</div>
            <div class="sidebar-label" style="margin-top:0.75rem;">Run ID</div>
            <div class="sidebar-value"><code>{config.run_id}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_actions(downloads: list[tuple[str, Path, str, str]]) -> None:
    if not downloads:
        return
    columns = st.columns(len(downloads))
    for column, (label, path, file_name, mime) in zip(columns, downloads):
        with path.open("rb") as file_pointer:
            column.download_button(
                label,
                data=file_pointer,
                file_name=file_name,
                mime=mime,
                use_container_width=True,
            )


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-note">
            Developed by <strong>Samuel de Andrade Maia</strong><br>
            2026 - Churn Prediction Data Product
        </div>
        """,
        unsafe_allow_html=True,
    )
