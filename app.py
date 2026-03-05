"""
Churn Prediction Dashboard - Streamlit
Autor: Samuel de Andrade Maia
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/LogisticRegression.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
COLOR_BG_START = "#f7f9fc"
COLOR_BG_END = "#eef3f9"
COLOR_PRIMARY = "#164e63"
COLOR_SECONDARY = "#0f766e"
COLOR_ALERT = "#dc2626"
COLOR_TEXT = "#0b1f33"
COLOR_MUTED = "#5b6b80"


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Carrega e prepara o dataset para o dashboard."""
    df_loaded = pd.read_csv(path)

    if "TotalCharges" in df_loaded.columns:
        df_loaded["TotalCharges"] = pd.to_numeric(df_loaded["TotalCharges"], errors="coerce")
        df_loaded["TotalCharges"] = df_loaded["TotalCharges"].fillna(df_loaded["TotalCharges"].median())

    return df_loaded


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Carrega o modelo treinado de churn."""
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_preprocessor(path: Path):
    """Carrega o pre-processador treinado de churn."""
    return joblib.load(path)


def inject_styles() -> None:
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
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 6%, rgba(22, 78, 99, 0.10), transparent 36%),
                radial-gradient(circle at 85% 0%, rgba(15, 118, 110, 0.12), transparent 38%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
            color: var(--text);
        }}
        html, body, [class*="css"]  {{
            font-family: "Space Grotesk", sans-serif;
        }}
        h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {{
            color: var(--text);
            letter-spacing: -0.02em;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(200deg, rgba(22, 78, 99, 0.96) 0%, rgba(11, 31, 51, 0.97) 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc;
        }}
        [data-testid="stSidebar"] a {{
            color: #99f6e4 !important;
        }}
        .hero {{
            background: linear-gradient(125deg, rgba(22, 78, 99, 0.96), rgba(15, 118, 110, 0.92));
            border-radius: 20px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 14px 42px rgba(11, 31, 51, 0.25);
            margin-bottom: 0.9rem;
        }}
        .hero-title {{
            color: #f8fafc;
            margin: 0;
            font-size: clamp(1.45rem, 4.5vw, 2.3rem);
            font-weight: 700;
            line-height: 1.15;
        }}
        .hero-subtitle {{
            color: rgba(248, 250, 252, 0.92);
            margin-top: 0.45rem;
            font-size: 0.98rem;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0.2rem 0 0.65rem 0;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(22, 78, 99, 0.18);
            border-radius: 14px;
            padding: 0.55rem 0.75rem;
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
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid rgba(22, 78, 99, 0.22);
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #fff;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(22, 78, 99, 0.25);
        }}
        .stButton > button:hover {{
            border-color: rgba(22, 78, 99, 0.35);
            background: linear-gradient(135deg, #0f4254, #0a5a53);
            color: #fff;
        }}
        .risk-box {{
            border-radius: 12px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(22, 78, 99, 0.20);
            background: rgba(255, 255, 255, 0.72);
            color: var(--text);
            font-size: 0.96rem;
        }}
        code {{
            font-family: "JetBrains Mono", monospace !important;
        }}
        @media (max-width: 820px) {{
            .hero {{
                border-radius: 14px;
                padding: 1rem 1rem;
            }}
            [data-testid="stMetric"] {{
                padding: 0.55rem 0.65rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">Churn Prediction System</h1>
            <p class="hero-subtitle">Monitoramento de cancelamento e predição individual de risco</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[pd.DataFrame | None, object | None, object | None, bool]:
    df: pd.DataFrame | None = None
    model = None
    preprocessor = None
    model_loaded = False

    with st.sidebar:
        st.markdown("## Status")

        if DATA_PATH.exists():
            df = load_data(DATA_PATH)
            st.success(f"Dados: {len(df):,} registros")
        else:
            st.error("Dataset não encontrado")

        if MODEL_PATH.exists():
            model = load_model(MODEL_PATH)
            model_loaded = hasattr(model, "predict_proba")
            if model_loaded:
                st.success("Modelo carregado")
            else:
                st.warning("Modelo encontrado, mas sem suporte a predict_proba")
        else:
            st.error("Modelo não encontrado")

        if PREPROCESSOR_PATH.exists():
            preprocessor = load_preprocessor(PREPROCESSOR_PATH)
            st.success("Pre-processador carregado")
        else:
            st.warning("Pré-processador não encontrado (inferência pode falhar)")

        st.markdown("---")
        st.markdown(
            """
            ### Autor
            **Samuel de Andrade Maia**

            [GitHub](https://github.com/samuelmaia-data-analyst)

            [LinkedIn](https://linkedin.com/in/samuelmaia-data-analyst)
            """
        )

    return df, model, preprocessor, model_loaded


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("---")
    st.markdown('<div class="section-title">Filtros de exploração</div>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        contract_options = ["Todos"]
        if "Contract" in df.columns:
            contract_options += sorted(df["Contract"].dropna().unique().tolist())
        selected_contract = st.selectbox("Contrato", contract_options)

    with filter_col2:
        internet_options = ["Todos"]
        if "InternetService" in df.columns:
            internet_options += sorted(df["InternetService"].dropna().unique().tolist())
        selected_internet = st.selectbox("Serviço de internet", internet_options)

    filtered_df = df.copy()

    if selected_contract != "Todos" and "Contract" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]

    if selected_internet != "Todos" and "InternetService" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]

    return filtered_df


def render_metrics(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Resumo executivo</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Clientes", f"{len(df):,}")

    with col2:
        churn_rate = 0.0
        if "Churn" in df.columns and len(df) > 0:
            churn_rate = (df["Churn"].value_counts().get("Yes", 0) / len(df)) * 100
        st.metric("Taxa de Churn", f"{churn_rate:.1f}%")

    with col3:
        avg_monthly = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 0
        st.metric("Media Mensal", f"${avg_monthly:.2f}")

    with col4:
        avg_tenure = df["tenure"].mean() if "tenure" in df.columns else 0
        st.metric("Media Tenure", f"{avg_tenure:.1f} meses")


def render_charts(filtered_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Análise visual</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuicao de Churn")
        if "Churn" in filtered_df.columns:
            churn_counts = filtered_df["Churn"].value_counts()
            fig1 = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
            title="Proporção de Cancelamentos",
                color=churn_counts.index,
                color_discrete_map={"Yes": COLOR_ALERT, "No": COLOR_PRIMARY},
                hole=0.48,
            )
            fig1.update_layout(
                margin=dict(l=10, r=10, t=48, b=10),
                legend_title_text="Churn",
                title_font=dict(size=18),
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Coluna 'Churn' não encontrada para gerar o gráfico.")

    with col2:
        st.subheader("Churn por Contrato")
        if {"Contract", "Churn"}.issubset(filtered_df.columns):
            contract_churn = pd.crosstab(filtered_df["Contract"], filtered_df["Churn"], normalize="index") * 100
            churn_yes = contract_churn.get("Yes", pd.Series(0, index=contract_churn.index))
            fig2 = px.bar(
                x=contract_churn.index,
                y=churn_yes,
                title="Taxa de Churn por Tipo de Contrato",
                labels={"x": "Tipo de Contrato", "y": "Taxa (%)"},
                color_discrete_sequence=[COLOR_SECONDARY],
            )
            fig2.update_traces(marker_line_color="#083344", marker_line_width=1.0)
            fig2.update_layout(margin=dict(l=10, r=10, t=48, b=10), title_font=dict(size=18))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Colunas necessárias para gráfico de contrato não encontradas.")


def render_data_preview(filtered_df: pd.DataFrame) -> None:
    with st.expander("Visualizar amostra de dados", expanded=False):
        columns_to_show = [
            "customerID",
            "gender",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "Churn",
        ]
        valid_cols = [col for col in columns_to_show if col in filtered_df.columns]
        st.dataframe(filtered_df[valid_cols].head(20), use_container_width=True)


def render_prediction(model, preprocessor) -> None:
    st.markdown("---")
    st.markdown('<div class="section-title">Predição individual</div>', unsafe_allow_html=True)

    with st.expander("Preencher dados do cliente", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gênero", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            tenure = st.number_input("Tenure (meses)", 0, 100, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 786.0)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        if st.button("Prever churn"):
            try:
                input_data = pd.DataFrame(
                    [
                        {
                            "gender": gender,
                            "SeniorCitizen": senior,
                            "Partner": partner,
                            "Dependents": dependents,
                            "tenure": tenure,
                            "PhoneService": phone_service,
                            "MultipleLines": "No",
                            "InternetService": internet_service,
                            "OnlineSecurity": "No",
                            "OnlineBackup": "No",
                            "DeviceProtection": "No",
                            "TechSupport": "No",
                            "StreamingTV": "No",
                            "StreamingMovies": "No",
                            "Contract": contract,
                            "PaperlessBilling": paperless,
                            "PaymentMethod": payment,
                            "MonthlyCharges": monthly,
                            "TotalCharges": total,
                        }
                    ]
                )

                model_input = preprocessor.transform(input_data) if preprocessor is not None else input_data
                proba = model.predict_proba(model_input)[0][1]

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=proba * 100,
                            title={"text": "Probabilidade de Churn"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": COLOR_PRIMARY},
                                "steps": [
                                    {"range": [0, 45], "color": "rgba(13, 148, 136, 0.35)"},
                                    {"range": [45, 70], "color": "rgba(245, 158, 11, 0.28)"},
                                    {"range": [70, 100], "color": "rgba(220, 38, 38, 0.30)"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=56, b=0), height=280)
                    st.plotly_chart(fig, use_container_width=True)

                with result_col2:
                    if proba >= 0.5:
                        st.error(f"Alto risco de cancelamento ({proba:.1%})")
                    else:
                        st.success(f"Baixo risco de cancelamento ({proba:.1%})")
                    st.markdown(
                        """
                        <div class="risk-box">
                            Priorize clientes com risco alto para contato proativo, revisão de contrato e oferta de retenção.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as exc:
                st.error(f"Erro ao gerar previsão: {exc}")


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #355070; padding: 0.35rem 0 0.8rem 0;'>
            <p>Desenvolvido por <strong>Samuel de Andrade Maia</strong></p>
            <p>2026 - Churn Prediction System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Churn Prediction - Samuel Maia",
        page_icon="CS",
        layout="wide",
    )

    pio.templates.default = "plotly_white"
    inject_styles()
    render_header()
    df, model, preprocessor, model_loaded = render_sidebar()

    if df is None:
        st.error(f"Dataset não encontrado em: {DATA_PATH}")
        st.stop()

    render_metrics(df)

    filtered_df = apply_filters(df)
    if filtered_df.empty:
        st.warning("Nenhum registro encontrado com os filtros selecionados.")
        st.stop()

    render_charts(filtered_df)
    render_data_preview(filtered_df)

    if model_loaded and model is not None:
        render_prediction(model, preprocessor)

    render_footer()


if __name__ == "__main__":
    main()
