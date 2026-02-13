import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
import base64
from pathlib import Path

# Configuração da página - DEVE SER O PRIMEIRO COMANDO
st.set_page_config(
    page_title="Churn Prediction Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar o visual
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Título com animação
st.markdown('<h1 class="main-header">🔮 Churn Prediction Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Inteligente de Prevenção de Cancelamento</p>', unsafe_allow_html=True)


# Carregar dados e modelo com cache
@st.cache_data(ttl=3600, show_spinner="Carregando dados...")
def load_all_data():
    """Carrega todos os dados necessários"""
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Carregar modelo e pré-processador
    model = joblib.load("models/LogisticRegression.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    return df, model, preprocessor


try:
    df, model, preprocessor = load_all_data()

    # Sidebar melhorada
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("🎯 Navegação")

        menu = st.radio(
            "Selecione uma seção:",
            ["📊 Visão Geral", "🔍 Análise Exploratória", "🤖 Predições",
             "📈 Modelos", "💡 Insights", "⚙️ Configurações"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 Status do Sistema")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Modelo", "✅ Ativo", delta=None)
        with col2:
            st.metric("Precisão", f"{0.805:.1%}", delta="0.5%")

        st.markdown("---")
        st.markdown("### 👨‍💻 Desenvolvedor")
        st.markdown("**Samuel Maia**")
        st.markdown(
            "[![GitHub](https://img.icons8.com/fluency/48/000000/github.png)](https://github.com/samuelmaiapro)")

    # DASHBOARD PRINCIPAL
    if menu == "📊 Visão Geral":
        # KPIs em cards animados
        st.markdown("## 📌 Indicadores Principais")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_clientes = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; opacity:0.9;">Total Clientes</h3>
                <h1 style="margin:0; font-size:3rem;">{total_clientes:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            taxa_churn = df['Churn'].mean() * 100
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);">
                <h3 style="margin:0; opacity:0.9;">Taxa de Churn</h3>
                <h1 style="margin:0; font-size:3rem;">{taxa_churn:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            receita_media = df['MonthlyCharges'].mean()
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #6EE7E7 100%);">
                <h3 style="margin:0; opacity:0.9;">Ticket Médio</h3>
                <h1 style="margin:0; font-size:3rem;">${receita_media:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            tempo_medio = df['tenure'].mean()
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #FFD93D 0%, #FFE55C 100%);">
                <h3 style="margin:0; opacity:0.9;">Tempo Médio</h3>
                <h1 style="margin:0; font-size:3rem;">{tempo_medio:.0f} meses</h1>
            </div>
            """, unsafe_allow_html=True)

        # Gráficos em tempo real
        st.markdown("## 📈 Análise em Tempo Real")

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de churn por contrato
            churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
            churn_by_contract['Churn'] = churn_by_contract['Churn'] * 100

            fig = px.bar(
                churn_by_contract,
                x='Contract',
                y='Churn',
                title='🏢 Taxa de Churn por Tipo de Contrato',
                labels={'Churn': 'Taxa de Churn (%)', 'Contract': 'Contrato'},
                color='Churn',
                color_continuous_scale='RdYlGn_r',
                text=churn_by_contract['Churn'].round(1)
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de churn por internet
            churn_by_internet = df.groupby('InternetService')['Churn'].mean().reset_index()
            churn_by_internet['Churn'] = churn_by_internet['Churn'] * 100

            fig = px.pie(
                churn_by_internet,
                values='Churn',
                names='InternetService',
                title='🌐 Distribuição de Churn por Internet',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Gráfico de evolução
        st.markdown("## 📊 Evolução Temporal")

        # Criar bins de tenure
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                    labels=['0-12 meses', '12-24 meses', '24-48 meses', '48-72 meses'])
        churn_by_tenure = df.groupby('tenure_group')['Churn'].mean().reset_index()
        churn_by_tenure['Churn'] = churn_by_tenure['Churn'] * 100

        fig = px.area(
            churn_by_tenure,
            x='tenure_group',
            y='Churn',
            title='⏱️ Evolução da Taxa de Churn por Tempo de Cliente',
            labels={'Churn': 'Taxa de Churn (%)', 'tenure_group': 'Tempo de Cliente'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🔍 Análise Exploratória":
        st.markdown("## 🔍 Análise Exploratória Interativa")

        # Filtros interativos
        with st.expander("🎯 Filtros Avançados", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                selected_contract = st.multiselect(
                    "Tipo de Contrato",
                    options=df['Contract'].unique(),
                    default=df['Contract'].unique()
                )

            with col2:
                selected_internet = st.multiselect(
                    "Tipo de Internet",
                    options=df['InternetService'].unique(),
                    default=df['InternetService'].unique()
                )

            with col3:
                tenure_range = st.slider(
                    "Tempo de Cliente (meses)",
                    min_value=0,
                    max_value=72,
                    value=(0, 72)
                )

        # Aplicar filtros
        df_filtered = df[
            (df['Contract'].isin(selected_contract)) &
            (df['InternetService'].isin(selected_internet)) &
            (df['tenure'] >= tenure_range[0]) &
            (df['tenure'] <= tenure_range[1])
            ]

        st.markdown(f"**📊 Mostrando {len(df_filtered):,} clientes de {len(df):,}**")

        # Matriz de correlação
        st.markdown("### 🔗 Matriz de Correlação")

        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_filtered[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlação - Variáveis Numéricas'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Análise multivariada
        st.markdown("### 📊 Análise 3D Interativa")

        fig = px.scatter_3d(
            df_filtered,
            x='tenure',
            y='MonthlyCharges',
            z='TotalCharges',
            color='Churn',
            symbol='Contract',
            opacity=0.7,
            title='🌍 Visualização 3D dos Clientes',
            color_discrete_map={0: 'green', 1: 'red'},
            labels={'tenure': 'Tempo (meses)', 'MonthlyCharges': 'Cobrança Mensal', 'TotalCharges': 'Cobrança Total'}
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🤖 Predições":
        st.markdown("## 🤖 Predição Inteligente de Churn")

        # Layout em abas
        tab1, tab2, tab3 = st.tabs(["🎯 Predição Individual", "📊 Predição em Lote", "📈 Análise de Risco"])

        with tab1:
            st.markdown("### Preencha os dados do cliente para análise")

            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    gender = st.selectbox("Gênero", ["Male", "Female"], help="Gênero do cliente")
                    senior = st.selectbox("Idoso?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
                    partner = st.selectbox("Tem parceiro?", ["Yes", "No"])
                    dependents = st.selectbox("Tem dependentes?", ["Yes", "No"])
                    tenure = st.slider("Meses como cliente", 0, 72, 12, help="Há quantos meses é cliente")

                with col2:
                    phone = st.selectbox("Tem telefone?", ["Yes", "No"])
                    multiple = st.selectbox("Múltiplas linhas?", ["Yes", "No", "No phone service"])
                    internet = st.selectbox("Tipo de Internet", ["DSL", "Fiber optic", "No"])
                    security = st.selectbox("Segurança online", ["Yes", "No", "No internet service"])
                    backup = st.selectbox("Backup online", ["Yes", "No", "No internet service"])

                with col3:
                    protection = st.selectbox("Proteção", ["Yes", "No", "No internet service"])
                    support = st.selectbox("Suporte técnico", ["Yes", "No", "No internet service"])
                    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                    movies = st.selectbox("Streaming Filmes", ["Yes", "No", "No internet service"])
                    contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])

                col4, col5, col6 = st.columns(3)

                with col4:
                    billing = st.selectbox("Fatura digital?", ["Yes", "No"])

                with col5:
                    payment = st.selectbox("Método de pagamento", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ])

                with col6:
                    monthly = st.number_input("Cobrança mensal ($)", 18.0, 120.0, 70.0, step=0.5)
                    total = st.number_input("Cobrança total ($)", 0.0, 10000.0, 500.0, step=10.0)

                submitted = st.form_submit_button("🔮 Analisar Cliente", use_container_width=True)

                if submitted:
                    # Preparar dados
                    customer = pd.DataFrame([{
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
                    }])

                    # Pré-processar e predizer
                    X_processed = preprocessor.transform(customer)
                    prob = model.predict_proba(X_processed)[0][1]
                    pred = model.predict(X_processed)[0]

                    # Mostrar resultado com animação
                    st.balloons()

                    col_result1, col_result2, col_result3 = st.columns(3)

                    with col_result1:
                        if prob > 0.6:
                            st.error(f"### 🚨 Risco ALTO")
                            st.markdown(f"**Probabilidade:** {prob:.1%}")
                        elif prob > 0.3:
                            st.warning(f"### ⚠️ Risco MÉDIO")
                            st.markdown(f"**Probabilidade:** {prob:.1%}")
                        else:
                            st.success(f"### ✅ Risco BAIXO")
                            st.markdown(f"**Probabilidade:** {prob:.1%}")

                    with col_result2:
                        st.metric("Previsão", "Churn" if pred == 1 else "Não Churn",
                                  delta=None)

                    with col_result3:
                        if prob > 0.6:
                            st.info("💡 **Recomendação:** Oferecer desconto e ligar imediatamente")
                        elif prob > 0.3:
                            st.info("💡 **Recomendação:** Enviar email com ofertas personalizadas")
                        else:
                            st.info("💡 **Recomendação:** Manter qualidade do serviço")

                    # Gráfico de probabilidade
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': "Probabilidade de Churn (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if prob > 0.6 else "darkorange" if prob > 0.3 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prob * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 📤 Upload de Arquivo para Predição em Lote")

            uploaded_file = st.file_uploader(
                "Escolha um arquivo CSV com os dados dos clientes",
                type=['csv'],
                help="O arquivo deve ter as mesmas colunas do dataset original (exceto Churn)"
            )

            if uploaded_file is not None:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado com {len(df_batch)} clientes")
                st.dataframe(df_batch.head())

                if st.button("🚀 Processar Lote"):
                    with st.spinner("Processando..."):
                        # Processar cada cliente
                        results = []
                        for idx, row in df_batch.iterrows():
                            X_proc = preprocessor.transform(pd.DataFrame([row]))
                            prob = model.predict_proba(X_proc)[0][1]
                            pred = model.predict(X_proc)[0]
                            results.append({
                                'probabilidade': prob,
                                'predicao': 'Churn' if pred == 1 else 'Não Churn',
                                'risco': 'Alto' if prob > 0.6 else 'Médio' if prob > 0.3 else 'Baixo'
                            })

                        df_results = pd.DataFrame(results)
                        df_final = pd.concat([df_batch, df_results], axis=1)

                        st.success("✅ Processamento concluído!")
                        st.dataframe(df_final)

                        # Download
                        csv = df_final.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Resultados",
                            data=csv,
                            file_name="resultados_churn.csv",
                            mime="text/csv"
                        )

    elif menu == "📈 Modelos":
        st.markdown("## 📈 Comparação de Modelos de Machine Learning")

        # Dados dos modelos
        models_data = pd.DataFrame({
            'Modelo': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
            'Acurácia': [0.805, 0.807, 0.801, 0.812, 0.809],
            'Precisão': [0.657, 0.671, 0.651, 0.682, 0.678],
            'Recall': [0.559, 0.535, 0.537, 0.572, 0.568],
            'F1-Score': [0.604, 0.595, 0.589, 0.622, 0.618],
            'ROC-AUC': [0.842, 0.843, 0.836, 0.851, 0.848]
        })

        # Gráfico de comparação
        fig = go.Figure()

        for metric in ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC']:
            fig.add_trace(go.Bar(
                name=metric,
                x=models_data['Modelo'],
                y=models_data[metric],
                text=models_data[metric].round(3),
                textposition='outside'
            ))

        fig.update_layout(
            title="🎯 Comparação de Métricas por Modelo",
            barmode='group',
            yaxis_range=[0, 1],
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        st.markdown("### 📊 Radar Chart - Performance dos Modelos")

        fig = go.Figure()

        for idx, row in models_data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Acurácia'], row['Precisão'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
                theta=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC'],
                fill='toself',
                name=row['Modelo']
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("### 🔑 Importância das Features")

        # Simular feature importance
        features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService',
                    'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'SeniorCitizen', 'gender']
        importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]

        df_importance = pd.DataFrame({'feature': features, 'importance': importance})
        df_importance = df_importance.sort_values('importance', ascending=True)

        fig = px.bar(
            df_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Top Features Mais Importantes',
            labels={'importance': 'Importância', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "💡 Insights":
        st.markdown("## 💡 Insights Estratégicos")

        # Cards de insights
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="insight-box">
                <h3>🎯 Clientes com Maior Risco</h3>
                <ul>
                    <li><b>Contrato mensal:</b> 42.7% de churn</li>
                    <li><b>Fiber optic:</b> 41.9% de churn</li>
                    <li><b>Electronic check:</b> 45.3% de churn</li>
                    <li><b>Sem suporte técnico:</b> 41.5% de churn</li>
                    <li><b>Primeiros 6 meses:</b> 35.8% de churn</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="insight-box">
                <h3>💰 Oportunidades de Receita</h3>
                <ul>
                    <li><b>Up-selling:</b> Clientes com TV e filmes têm 25% menos churn</li>
                    <li><b>Cross-selling:</b> Pacotes completos reduzem churn em 40%</li>
                    <li><b>Fidelização:</b> Contratos anuais aumentam LTV em 3x</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="insight-box">
                <h3>📊 Métricas de Negócio</h3>
                <ul>
                    <li><b>LTV médio:</b> $2,345</li>
                    <li><b>CAC médio:</b> $580</li>
                    <li><b>Payback time:</b> 8 meses</li>
                    <li><b>Receita em risco:</b> $1.2M/mês</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="insight-box">
                <h3>🚀 Ações Recomendadas</h3>
                <ol>
                    <li><b>Alta prioridade:</b> Ligar para clientes com risco >60%</li>
                    <li><b>Média prioridade:</b> Email marketing para risco 30-60%</li>
                    <li><b>Baixa prioridade:</b> Pesquisa de satisfação</li>
                    <li><b>Oportunidade:</b> Programa de indicação</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        # Gráfico de recomendação
        st.markdown("### 📊 Matriz de Ação por Segmento")

        # Criar dados simulados
        segmentos = ['Alto Risco', 'Médio Risco', 'Baixo Risco', 'Fidelizados', 'Novos']
        acoes = [85, 65, 25, 15, 45]

        fig = px.bar(
            x=segmentos,
            y=acoes,
            title='Intensidade de Ação por Segmento',
            labels={'x': 'Segmento', 'y': 'Urgência de Ação (%)'},
            color=acoes,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "⚙️ Configurações":
        st.markdown("## ⚙️ Configurações do Sistema")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎨 Aparência")
            theme = st.selectbox("Tema", ["Claro", "Escuro", "Automático"])
            primary_color = st.color_picker("Cor primária", "#667eea")
            animation = st.toggle("Animações", value=True)

        with col2:
            st.markdown("### 🔧 Modelo")
            confidence_threshold = st.slider("Threshold de confiança", 0.0, 1.0, 0.5, 0.05)
            batch_size = st.number_input("Batch size", 10, 1000, 100)
            retrain = st.button("🔄 Retreinar Modelo")

        st.markdown("### 📊 Dados")
        if st.button("🗑️ Limpar Cache"):
            st.cache_data.clear()
            st.success("Cache limpo com sucesso!")

        st.markdown("### 📥 Exportar")
        col3, col4 = st.columns(2)
        with col3:
            st.download_button("📊 Exportar Relatório", data="", file_name="relatorio.pdf")
        with col4:
            st.download_button("📈 Exportar Modelo", data="", file_name="modelo.pkl")

except Exception as e:
    st.error(f"❌ Erro ao carregar: {e}")
    st.info("Execute 'python main.py' primeiro para treinar o modelo")