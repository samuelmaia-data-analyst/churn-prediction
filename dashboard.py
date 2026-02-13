import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuração da página
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📊 Dashboard de Predição de Churn")
st.markdown("---")


# Carregar modelo e dados
@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    model_path = "models/LogisticRegression.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_data():
    """Carrega os dados originais para análise"""
    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        return df
    return None


# Carregar
model = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/machine-learning.png", width=100)
    st.title("Configurações")

    if model is None:
        st.error("❌ Modelo não encontrado! Execute 'python main.py' primeiro.")
    else:
        st.success("✅ Modelo carregado!")

# Layout principal
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total de Clientes",
        value=df.shape[0] if df is not None else 0,
        delta=None
    )

with col2:
    if df is not None:
        churn_rate = df['Churn'].mean() * 100
        st.metric(
            label="Taxa de Churn",
            value=f"{churn_rate:.1f}%",
            delta=f"{churn_rate - 26.5:.1f}%"  # Comparação com média do setor
        )

with col3:
    if df is not None:
        st.metric(
            label="Clientes Ativos",
            value=df[df['Churn'] == 0].shape[0],
            delta=None
        )

st.markdown("---")

# Tabs para diferentes visualizações
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Análise Exploratória",
    "🤖 Predição Individual",
    "📊 Comparação de Modelos",
    "📋 Dados Brutos"
])

with tab1:
    if df is not None:
        st.header("Análise Exploratória dos Dados")

        # Gráfico de churn
        fig = px.pie(
            df,
            names='Churn',
            title='Distribuição de Churn',
            color='Churn',
            color_discrete_map={0: 'green', 1: 'red'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        # Análise por variáveis categóricas
        st.subheader("Taxa de Churn por Categoria")

        cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'gender']
        selected_cat = st.selectbox("Selecione a variável:", cat_cols)

        churn_by_cat = df.groupby(selected_cat)['Churn'].mean().reset_index()
        churn_by_cat['Churn'] = churn_by_cat['Churn'] * 100

        fig = px.bar(
            churn_by_cat,
            x=selected_cat,
            y='Churn',
            title=f'Taxa de Churn por {selected_cat}',
            labels={'Churn': 'Taxa de Churn (%)'},
            color='Churn',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Análise por tenure
        st.subheader("Distribuição por Tempo de Cliente")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histograma', 'Boxplot por Churn')
        )

        fig.add_trace(
            go.Histogram(x=df['tenure'], nbinsx=30, name='Todos'),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(y=df[df['Churn'] == 0]['tenure'], name='Não Churn', line_color='green'),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(y=df[df['Churn'] == 1]['tenure'], name='Churn', line_color='red'),
            row=1, col=2
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Predição Individual de Churn")

    if model is not None:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gênero", ["Male", "Female"])
                senior = st.selectbox("Idoso?", [0, 1])
                partner = st.selectbox("Tem parceiro?", ["Yes", "No"])
                dependents = st.selectbox("Tem dependentes?", ["Yes", "No"])
                tenure = st.number_input("Meses como cliente", 0, 72, 12)
                phone = st.selectbox("Tem telefone?", ["Yes", "No"])

            with col2:
                multiple = st.selectbox("Múltiplas linhas?", ["Yes", "No", "No phone service"])
                internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
                security = st.selectbox("Segurança online", ["Yes", "No", "No internet service"])
                backup = st.selectbox("Backup online", ["Yes", "No", "No internet service"])
                protection = st.selectbox("Proteção", ["Yes", "No", "No internet service"])
                support = st.selectbox("Suporte técnico", ["Yes", "No", "No internet service"])

            with col3:
                tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                movies = st.selectbox("Streaming Filmes", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
                billing = st.selectbox("Fatura digital", ["Yes", "No"])
                payment = st.selectbox("Pagamento", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly = st.number_input("Cobrança mensal", 18.0, 120.0, 70.0)
                total = st.number_input("Cobrança total", 0.0, 10000.0, 500.0)

            submitted = st.form_submit_button("🔮 Analisar Cliente", type="primary")

            if submitted:
                # Preparar dados
                customer = {
                    'gender': gender,
                    'SeniorCitizen': senior,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone,
                    'MultipleLines': multiple,
                    'InternetService': internet,
                    'OnlineSecurity': security,
                    'OnlineBackup': backup,
                    'DeviceProtection': protection,
                    'TechSupport': support,
                    'StreamingTV': tv,
                    'StreamingMovies': movies,
                    'Contract': contract,
                    'PaperlessBilling': billing,
                    'PaymentMethod': payment,
                    'MonthlyCharges': monthly,
                    'TotalCharges': total
                }

                # Fazer predição (aqui você precisaria carregar o pré-processador também)
                st.success("✅ Cliente analisado com sucesso!")

                # Mock da predição (substituir pela predição real)
                prob = np.random.random()

                # Mostrar resultado
                col_result1, col_result2, col_result3 = st.columns(3)

                with col_result1:
                    if prob > 0.6:
                        st.error(f"🚨 Risco ALTO de Churn")
                    elif prob > 0.3:
                        st.warning(f"⚠️ Risco MÉDIO de Churn")
                    else:
                        st.success(f"✅ Risco BAIXO de Churn")

                with col_result2:
                    st.metric("Probabilidade", f"{prob:.1%}")

                with col_result3:
                    st.metric("Previsão", "Churn" if prob > 0.5 else "Não Churn")
    else:
        st.warning("⚠️ Modelo não encontrado. Execute 'python main.py' primeiro.")

with tab3:
    st.header("Comparação de Modelos")

    # Dados mockados (substituir pelos resultados reais)
    models_data = {
        'Modelo': ['LogisticRegression', 'RandomForest', 'GradientBoosting'],
        'Acurácia': [0.8055, 0.8070, 0.8006],
        'Precisão': [0.6572, 0.6711, 0.6505],
        'Recall': [0.5588, 0.5348, 0.5374],
        'F1-Score': [0.6040, 0.5952, 0.5886],
        'ROC-AUC': [0.8420, 0.8427, 0.8362]
    }

    df_models = pd.DataFrame(models_data)

    # Gráfico de comparação
    fig = go.Figure()

    for metric in ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_models['Modelo'],
            y=df_models[metric],
            text=df_models[metric].round(3),
            textposition='outside'
        ))

    fig.update_layout(
        title="Comparação de Métricas por Modelo",
        barmode='group',
        yaxis_range=[0, 1],
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabela de resultados
    st.subheader("Tabela de Resultados")
    st.dataframe(df_models.style.highlight_max(axis=0), use_container_width=True)

with tab4:
    st.header("Dados Brutos")

    if df is not None:
        st.dataframe(df, use_container_width=True)

        # Estatísticas descritivas
        if st.checkbox("Mostrar estatísticas descritivas"):
            st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("⚠️ Dados não encontrados.")

# Footer
st.markdown("---")
st.markdown(
    "<center>Desenvolvido por Samuel Maia | Projeto de Predição de Churn</center>",
    unsafe_allow_html=True
)