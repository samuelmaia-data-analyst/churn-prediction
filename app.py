# app.py (renomeie seu dashboard para app.py - é o padrão do Streamlit Cloud)

"""
Churn Prediction Dashboard - Versão Streamlit Cloud
Autor: Samuel de Andrade Maia
GitHub: @samuelmaiapro
LinkedIn: /in/samuelmaiapro
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import urllib.request
from pathlib import Path

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO)
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction - Samuel Maia",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNÇÕES DE CARREGAMENTO PARA CLOUD
# ============================================================================

@st.cache_resource
def load_models():
    """
    Carrega modelos do repositório GitHub ou local
    """
    models = {}

    # URL base dos modelos no GitHub (substitua pelo seu usuário)
    GITHUB_URL = "https://github.com/samuelmaiapro/churn-prediction/raw/main/models/"

    model_files = {
        'LogisticRegression': 'LogisticRegression.joblib',
        'RandomForest': 'RandomForest.joblib',
        'GradientBoosting': 'GradientBoosting.joblib',
        'preprocessor': 'preprocessor.joblib'
    }

    # Criar pasta models se não existir
    os.makedirs('models', exist_ok=True)

    for model_name, filename in model_files.items():
        local_path = f'models/{filename}'

        # Tentar carregar local primeiro
        if os.path.exists(local_path):
            try:
                models[model_name] = joblib.load(local_path)
                continue
            except:
                pass

        # Se não conseguir local, tentar baixar do GitHub
        try:
            st.info(f"📥 Baixando {model_name} do GitHub...")
            url = GITHUB_URL + filename
            urllib.request.urlretrieve(url, local_path)
            models[model_name] = joblib.load(local_path)
            st.success(f"✅ {model_name} baixado com sucesso!")
        except:
            st.warning(f"⚠️ Não foi possível carregar {model_name}")

    return models

@st.cache_data
def load_data():
    """
    Carrega dados do repositório GitHub ou local
    """
    # URL do dataset no GitHub
    GITHUB_DATA_URL = "https://github.com/samuelmaiapro/churn-prediction/raw/main/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    local_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # Criar pasta data se não existir
    os.makedirs('data/raw', exist_ok=True)

    # Tentar carregar local primeiro
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        # Baixar do GitHub
        try:
            st.info("📥 Baixando dataset do GitHub...")
            urllib.request.urlretrieve(GITHUB_DATA_URL, local_path)
            df = pd.read_csv(local_path)
            st.success("✅ Dataset baixado com sucesso!")
        except:
            st.error("❌ Não foi possível carregar o dataset")
            return None

    # Tratamento dos dados
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    return df

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

# Título com seu nome
st.markdown("""
<h1 style='text-align: center; color: #667EEA;'>
    🔮 Churn Prediction System
</h1>
<h3 style='text-align: center; color: #4A5568;'>
    Samuel de Andrade Maia
</h3>
<hr>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/667EEA/FFFFFF?text=Churn+Prediction", use_container_width=True)

    st.markdown("## 📊 Status")

    # Carregar dados e modelos
    with st.spinner("🔄 Carregando recursos..."):
        data = load_data()
        models = load_models()

    if data is not None:
        st.success(f"✅ Dados: {len(data)} registros")

    model_count = len([m for m in models.keys() if m != 'preprocessor'])
    if model_count > 0:
        st.success(f"✅ Modelos: {model_count} disponíveis")
    else:
        st.error("❌ Nenhum modelo encontrado")

    st.markdown("---")

    # Seleção de modelo
    if model_count > 0:
        model_names = [m for m in models.keys() if m != 'preprocessor']
        selected_model = st.selectbox(
            "🤖 Selecione o modelo:",
            model_names,
            format_func=lambda x: f"🏆 {x}" if x == 'LogisticRegression' else x
        )
    else:
        selected_model = None
        st.warning("⚠️ Modelos serão baixados automaticamente")

    st.markdown("---")

    # Links do autor
    st.markdown("""
    ### 👨‍💻 Autor
    **Samuel de Andrade Maia**
    
    [![GitHub](https://img.shields.io/badge/GitHub-@samuelmaiapro-181717?style=flat&logo=github)](https://github.com/samuelmaiapro)
    
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-@samuelmaiapro-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/samuelmaiapro)
    """)

    st.markdown("---")
    st.caption(f"🔄 Última atualização: 2024")

# ============================================================================
# CORPO PRINCIPAL
# ============================================================================

if data is None:
    st.error("""
    ### ❌ Erro ao carregar dados
    
    Verifique:
    1. Conexão com internet
    2. Repositório GitHub público
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
            color_discrete_sequence=['#F87171'],
            text_auto='.1f'
        )
        fig2.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

# Predição (se houver modelo)
if selected_model and selected_model in models:
    st.markdown("---")
    st.markdown("## 🔍 Predição Individual")

    with st.expander("Preencher dados do cliente", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gênero", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            tenure = st.number_input("Tenure (meses)", 0, 100, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        with col3:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5, step=0.5)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 786.0, step=10.0)
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
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#667EEA"},
                            'steps': [
                                {'range': [0, 30], 'color': '#d4edda'},
                                {'range': [30, 70], 'color': '#fff3cd'},
                                {'range': [70, 100], 'color': '#f8d7da'}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if proba >= 0.5:
                        st.error(f"""
                        ### ⚠️ Alto Risco
                        **Probabilidade:** {proba:.1%}
                        
                        **Recomendação:** Ação imediata de retenção
                        """)
                    else:
                        st.success(f"""
                        ### ✅ Baixo Risco
                        **Probabilidade:** {proba:.1%}
                        
                        **Recomendação:** Manter qualidade do serviço
                        """)

                    st.caption(f"Modelo: {selected_model}")

            except Exception as e:
                st.error(f"Erro na predição: {str(e)}")

# Resultados dos modelos
st.markdown("---")
st.markdown("## 📈 Performance dos Modelos")

performance_data = pd.DataFrame({
    'Modelo': ['LogisticRegression', 'RandomForest', 'GradientBoosting'],
    'Acurácia': [0.8055, 0.8070, 0.8006],
    'Precisão': [0.6572, 0.6711, 0.6505],
    'Recall': [0.5588, 0.5348, 0.5374],
    'F1-Score': [0.6040, 0.5952, 0.5886],
    'ROC-AUC': [0.8420, 0.8427, 0.8362]
})

st.dataframe(
    performance_data.style.highlight_max(axis=0, subset=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']),
    use_container_width=True
)

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Desenvolvido com ❤️ por <strong>Samuel de Andrade Maia</strong></p>
    <p>
        <a href='https://github.com/samuelmaiapro' target='_blank'>GitHub</a> • 
        <a href='https://linkedin.com/in/samuelmaiapro' target='_blank'>LinkedIn</a>
    </p>
    <p style='font-size: 0.8rem;'>© 2024 - Churn Prediction System</p>
</div>
""", unsafe_allow_html=True)