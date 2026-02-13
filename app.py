"""
Churn Prediction Dashboard - Streamlit Cloud
Autor: Samuel de Andrade Maia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction - Samuel Maia",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================================

@st.cache_resource
def load_models():
    """Carrega os modelos treinados"""
    models = {}
    model_paths = {
        'LogisticRegression': 'models/LogisticRegression.joblib',
        'RandomForest': 'models/RandomForest.joblib',
        'GradientBoosting': 'models/GradientBoosting.joblib',
        'preprocessor': 'models/preprocessor.joblib'
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"✅ Carregado: {name}")
            except Exception as e:
                print(f"❌ Erro ao carregar {name}: {e}")

    return models


@st.cache_data
def load_data():
    """Carrega os dados"""
    path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    if os.path.exists(path):
        df = pd.read_csv(path)
        # Tratar TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        return df
    return None


# ============================================================================
# CABEÇALHO
# ============================================================================
st.markdown("""
<h1 style='text-align: center; color: #667EEA;'>
    🔮 Churn Prediction System
</h1>
<h3 style='text-align: center; color: #4A5568;'>
    Samuel de Andrade Maia
</h3>
<hr>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/667EEA/FFFFFF?text=Churn+Prediction", use_container_width=True)

    st.markdown("## 📊 Status")

    # Carregar dados e modelos
    with st.spinner("🔄 Carregando..."):
        data = load_data()
        models = load_models()

    # Status dos dados
    if data is not None:
        st.success(f"✅ Dados: {len(data)} registros")
    else:
        st.error("❌ Dados não encontrados")

    # Status dos modelos
    model_count = len([m for m in models.keys() if m != 'preprocessor'])
    if model_count > 0:
        st.success(f"✅ Modelos: {model_count} disponíveis")

        # Seleção de modelo
        model_names = [m for m in models.keys() if m != 'preprocessor']
        selected_model = st.selectbox(
            "🤖 Selecione o modelo:",
            model_names,
            index=0
        )
    else:
        st.error("❌ Nenhum modelo encontrado")
        selected_model = None

    st.markdown("---")

    # Links do autor
    st.markdown("""
    ### 👨‍💻 Autor
    **Samuel de Andrade Maia**

    [![GitHub](https://img.shields.io/badge/GitHub-@samuelmaiapro-181717?style=flat&logo=github)](https://github.com/samuelmaiapro)

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-@samuelmaiapro-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/samuelmaiapro)
    """)

# ============================================================================
# CORPO PRINCIPAL
# ============================================================================

if data is None:
    st.error("""
    ### ❌ Dataset não encontrado

    Certifique-se de que o arquivo está em: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
    """)
    st.stop()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Clientes", f"{len(data):,}")

with col2:
    churn_rate = (data['Churn'].value_counts().get('Yes', 0) / len(data)) * 100
    st.metric("Taxa de Churn", f"{churn_rate:.1f}%")

with col3:
    avg_monthly = data['MonthlyCharges'].mean()
    st.metric("Média Mensal", f"${avg_monthly:.2f}")

with col4:
    avg_tenure = data['tenure'].mean()
    st.metric("Média Tenure", f"{avg_tenure:.1f} meses")

# Gráficos
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribuição de Churn")
    churn_counts = data['Churn'].value_counts()
    fig1 = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Proporção de Cancelamentos",
        color_discrete_sequence=['#667EEA', '#F87171']
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📊 Churn por Contrato")
    if 'Contract' in data.columns:
        contract_churn = pd.crosstab(data['Contract'], data['Churn'], normalize='index') * 100
        fig2 = px.bar(
            x=contract_churn.index,
            y=contract_churn['Yes'],
            title="Taxa de Churn por Tipo de Contrato",
            labels={'x': 'Tipo de Contrato', 'y': 'Taxa (%)'},
            color_discrete_sequence=['#F87171']
        )
        st.plotly_chart(fig2, use_container_width=True)

# Predição
if selected_model and selected_model in models:
    st.markdown("---")
    st.markdown("## 🔍 Predição Individual")

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
            payment = st.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                    "Credit card (automatic)"])

        with col3:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 786.0)
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        if st.button("🔮 Prever Churn", use_container_width=True):
            try:
                # Preparar dados
                input_data = pd.DataFrame([{
                    'gender': gender,
                    'SeniorCitizen': senior,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone,
                    'MultipleLines': 'No' if phone == 'No' else 'Yes',
                    'InternetService': internet,
                    'OnlineSecurity': 'No' if internet == 'No' else 'No',
                    'OnlineBackup': 'No' if internet == 'No' else 'No',
                    'DeviceProtection': 'No' if internet == 'No' else 'No',
                    'TechSupport': 'No' if internet == 'No' else 'No',
                    'StreamingTV': 'No' if internet == 'No' else 'No',
                    'StreamingMovies': 'No' if internet == 'No' else 'No',
                    'Contract': contract,
                    'PaperlessBilling': paperless,
                    'PaymentMethod': payment,
                    'MonthlyCharges': monthly,
                    'TotalCharges': total
                }])

                # Predição
                model = models[selected_model]
                proba = model.predict_proba(input_data)[0][1]

                # Resultado
                col1, col2 = st.columns(2)

                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={'text': "Probabilidade de Churn"},
                        gauge={'axis': {'range': [0, 100]}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if proba >= 0.5:
                        st.error(f"### ⚠️ Alto Risco\nProbabilidade: {proba:.1%}")
                    else:
                        st.success(f"### ✅ Baixo Risco\nProbabilidade: {proba:.1%}")

            except Exception as e:
                st.error(f"Erro: {str(e)}")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Desenvolvido com ❤️ por <strong>Samuel de Andrade Maia</strong></p>
    <p>© 2024 - Churn Prediction System</p>
</div>
""", unsafe_allow_html=True)