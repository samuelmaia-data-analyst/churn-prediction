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

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction - Samuel Maia",
    page_icon="🔮",
    layout="wide"
)

# ============================================================================
# TÍTULO
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
    st.markdown("## 📊 Status")

    # Verificar dados
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        st.success(f"✅ Dados: {len(df)} registros")
    else:
        df = None
        st.error("❌ Dataset não encontrado")

    # Verificar modelos
    model_path = 'models/LogisticRegression.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Modelo carregado")
        model_loaded = True
    else:
        model = None
        model_loaded = False
        st.error("❌ Modelo não encontrado")

    st.markdown("---")

    # Links do autor
    st.markdown("""
    ### 👨‍💻 Autor
    **Samuel de Andrade Maia**

    [![GitHub](https://img.shields.io/badge/GitHub-@samuelmaia-data-analyst-181717?style=flat&logo=github)](https://github.com/samuelmaia-data-analyst)

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-@samuelmaia-data-analyst-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/samuelmaia-data-analyst)
    """)

# ============================================================================
# CORPO PRINCIPAL
# ============================================================================

if df is None:
    st.error("""
    ### ❌ Dataset não encontrado

    Certifique-se de que o arquivo está em: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
    """)
    st.stop()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Clientes", f"{len(df):,}")

with col2:
    churn_rate = (df['Churn'].value_counts().get('Yes', 0) / len(df)) * 100
    st.metric("Taxa de Churn", f"{churn_rate:.1f}%")

with col3:
    avg_monthly = df['MonthlyCharges'].mean()
    st.metric("Média Mensal", f"${avg_monthly:.2f}")

with col4:
    avg_tenure = df['tenure'].mean()
    st.metric("Média Tenure", f"{avg_tenure:.1f} meses")

# Gráficos
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribuição de Churn")
    churn_counts = df['Churn'].value_counts()
    fig1 = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Proporção de Cancelamentos",
        color_discrete_sequence=['#667EEA', '#F87171']
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📊 Churn por Contrato")
    if 'Contract' in df.columns:
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        fig2 = px.bar(
            x=contract_churn.index,
            y=contract_churn['Yes'],
            title="Taxa de Churn por Tipo de Contrato",
            labels={'x': 'Tipo de Contrato', 'y': 'Taxa (%)'},
            color_discrete_sequence=['#F87171']
        )
        st.plotly_chart(fig2, use_container_width=True)

# Predição (se modelo carregado)
if model_loaded and model is not None:
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