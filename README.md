<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=200&section=header&text=Churn%20Prediction&fontSize=60&fontAlignY=35&desc=Previsão%20de%20Cancelamento%20de%20Clientes%20com%20Machine%20Learning&descAlignY=55&animation=fadeIn" width="100%" alt="header"/>
</div>

<div align="center">
  <h1>🔮 Churn Prediction System</h1>
  <p><strong>Sistema inteligente end-to-end para previsão e análise de cancelamento de clientes em empresas de telecomunicações</strong></p>
  
  <!-- Badges Profissionais -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </p>
  
  <p>
    <img src="https://img.shields.io/github/license/samuelmaia-data-analyst/churn-prediction?style=for-the-badge&color=blue" alt="License">
    <img src="https://img.shields.io/github/stars/samuelmaia-data-analyst/churn-prediction?style=for-the-badge&color=yellow" alt="Stars">
    <img src="https://img.shields.io/github/forks/samuelmaia-data-analyst/churn-prediction?style=for-the-badge&color=orange" alt="Forks">
    <img src="https://img.shields.io/github/issues/samuelmaia-data-analyst/churn-prediction?style=for-the-badge&color=red" alt="Issues">
  </p>

  <!-- Menu de Navegação -->
  <h3>
    <a href="#-sobre-o-projeto">📋 Sobre</a> •
    <a href="#-demonstração">🎥 Demo</a> •
    <a href="#-tecnologias">⚙️ Tech</a> •
    <a href="#-instalação">🚀 Instalação</a> •
    <a href="#-como-usar">📖 Guia</a> •
    <a href="#-resultados">📊 Resultados</a> •
    <a href="#-autor">👨‍💻 Autor</a>
  </h3>
</div>

---

## 📋 Sobre o Projeto

Este projeto implementa um **sistema completo de Machine Learning** para predição de churn (cancelamento de serviços) em empresas de telecomunicações. Utilizando dados reais do [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), desenvolvemos uma solução **end-to-end** que permite identificar clientes em risco de cancelamento e tomar ações preventivas.

### ✨ Funcionalidades Principais

<div align="center">

| | Módulo | Descrição | Tecnologia |
|---|--------|-----------|------------|
| 🔍 | **Análise Preditiva** | Identificação de clientes em risco | Scikit-learn |
| 🤖 | **Múltiplos Modelos** | Comparação entre algoritmos | LogisticRegression, RandomForest, GradientBoosting |
| ⚡ | **API REST** | Predições em tempo real | FastAPI + Uvicorn |
| 📊 | **Dashboard Interativo** | Visualização de dados e resultados | Streamlit + Plotly |
| 💻 | **CLI** | Predições via linha de comando | Python Click |
| 🧪 | **Testes Automatizados** | Garantia de qualidade | Pytest + GitHub Actions |

</div>

---

## 🎥 Demonstração

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="https://via.placeholder.com/500x300/667EEA/FFFFFF?text=API+FastAPI+em+Ação" width="100%" alt="API Demo"/>
        <br />
        <sub><strong>📡 API REST com FastAPI</strong> - Documentação interativa automática</sub>
      </td>
      <td align="center" width="50%">
        <img src="https://via.placeholder.com/500x300/38A169/FFFFFF?text=Dashboard+Streamlit" width="100%" alt="Dashboard Demo"/>
        <br />
        <sub><strong>📊 Dashboard Interativo</strong> - Visualização de métricas e predições</sub>
      </td>
    </tr>
  </table>
  <p><i>⚡ GIFs demonstrativos serão adicionados em breve na pasta <code>assets/</code></i></p>
</div>

---

## ⚙️ Stack Tecnológica

<div align="center">

### Core
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Visualização
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-3C3C3D?style=for-the-badge&logo=shap&logoColor=white)

### APIs & Web
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-2496ED?style=for-the-badge&logo=uvicorn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### DevOps & Qualidade
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

---

## 📊 Dataset

### Telco Customer Churn

O dataset contém informações detalhadas de **7.043 clientes** de uma empresa de telecomunicações, com **21 variáveis** e taxa de churn de **26,5%**.

<details>
<summary><b>📈 Clique para ver as estatísticas completas</b></summary>

<br>

| Métrica | Valor |
|---------|-------|
| **Registros** | 7.043 clientes |
| **Variáveis** | 21 (20 features + 1 target) |
| **Taxa de Churn** | 26,5% (1.869 clientes) |
| **Período** | Dados históricos |

</details>

<details>
<summary><b>📋 Clique para ver as features disponíveis</b></summary>

<br>

| Categoria | Variáveis |
|-----------|-----------|
| **Demografia** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Serviços** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Contrato** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Financeiro** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` (Yes/No) |

</details>

<details>
<summary><b>👀 Clique para ver uma prévia dos dados</b></summary>

<br>

```python
import pandas as pd

df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())

customerID	gender	tenure	MonthlyCharges	TotalCharges	Churn
7590-VHVEG	Female	1	29.85	29.85	No
5575-GNVDE	Male	34	56.95	1889.5	No
3668-QPYBK	Male	2	53.85	108.15	Yes
7795-CFOCW	Male	45	42.30	1840.75	No
9237-HQITU	Female	2	70.70	151.65	Yes
</details>
🚀 Instalação
Pré-requisitos

    Python 3.10 ou superior

    Git

    pip (gerenciador de pacotes Python)

    (Opcional) Docker

Passo a Passo
bash

# 1. Clone o repositório
git clone https://github.com/samuelmaia-data-analyst/churn-prediction.git
cd churn-prediction

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset
# Opção 1 - Download automático (requer conta Kaggle)
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/

# Opção 2 - Download manual
# Baixe o arquivo do Kaggle e coloque em data/raw/

🐳 Instalação com Docker (Alternativa)
bash

# Construir a imagem
docker build -t churn-prediction .

# Executar o container
docker run -p 8000:8000 -p 8501:8501 churn-prediction

📖 Como Usar
1️⃣ Treinar o Modelo Principal
bash

python main.py

<details> <summary><b>📋 Clique para ver a saída esperada</b></summary>
text

============================================================
 PROJETO DE PREDIÇÃO DE CHURN
============================================================

1. Carregando dados...
   ✓ Dados carregados: (7043, 21)

2. Pré-processando dados...
   ✓ Treino: (5634, 19), Teste: (1409, 19)

3. Engenharia de features...
   ✓ Features processadas: 30

4. Treinando modelos...
   LogisticRegression - F1: 0.6040, AUC: 0.8420
   RandomForest - F1: 0.5952, AUC: 0.8427
   GradientBoosting - F1: 0.5886, AUC: 0.8362

✅ Melhor modelo: LogisticRegression

</details>
2️⃣ Modo Interativo (CLI)
bash

python predict_customer.py

3️⃣ API REST
bash

uvicorn api:app --reload

Acesse a documentação interativa: http://localhost:8000/docs
<details> <summary><b>📮 Clique para ver exemplo de requisição</b></summary>
bash

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0
  }'

Resposta esperada:
json

{
  "customer_id": "pred_001",
  "churn_probability": 0.67,
  "churn_prediction": "Yes",
  "model_used": "LogisticRegression",
  "timestamp": "2024-01-01T12:00:00Z"
}

</details>
4️⃣ Dashboard Interativo
bash

streamlit run dashboard.py

Acesse: http://localhost:8501
📊 Resultados
🏆 Comparação de Modelos
<div align="center">
Modelo	Acurácia	Precisão	Recall	F1-Score	ROC-AUC
LogisticRegression 🏆	0.8055	0.6572	0.5588	0.6040	0.8420
RandomForest	0.8070	0.6711	0.5348	0.5952	0.8427
GradientBoosting	0.8006	0.6505	0.5374	0.5886	0.8362
</div>
📈 Visualização de Métricas
<div align="center"> <img src="https://via.placeholder.com/800x400/667EEA/FFFFFF?text=Comparação+de+Modelos+-+ROC+AUC" width="80%" alt="Model Comparison"/> <br> <sub><strong>📊 Curva ROC comparativa dos modelos treinados</strong></sub> </div>
🔍 Feature Importance
<div align="center"> <img src="https://via.placeholder.com/800x400/38A169/FFFFFF?text=Top+10+Features+Mais+Importantes" width="80%" alt="Feature Importance"/> <br> <sub><strong>📊 Top 10 features mais importantes para o modelo</strong></sub> </div>
🔑 Principais Insights
	Insight	Impacto
✅	Melhor modelo: Logistic Regression	Melhor equilíbrio entre as métricas
✅	ROC-AUC > 84% em todos os modelos	Alta capacidade preditiva
✅	Features críticas: tenure, MonthlyCharges, Contract, InternetService	Direcionam ações de retenção
✅	Clientes com contratos mensais têm 3x mais chance de churn	Foco em fidelização
📁 Estrutura do Projeto
text

📦 churn-prediction
├── 📂 .github
│   └── 📂 workflows          # CI/CD com GitHub Actions
├── 📂 assets                  # Imagens e GIFs para documentação
├── 📂 data
│   ├── 📂 raw                # Dados brutos (CSV original)
│   └── 📂 processed          # Dados pré-processados
├── 📂 models                  # Modelos treinados (.joblib)
├── 📂 notebooks               # Análises exploratórias
├── 📂 src
│   ├── 📂 data               # Carregamento e validação
│   ├── 📂 features           # Engenharia de features
│   └── 📂 models             # Treinamento e avaliação
├── 📂 tests                   # Testes unitários
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── 📄 main.py                 # Pipeline principal
├── 📄 api.py                  # API FastAPI
├── 📄 dashboard.py            # Dashboard Streamlit
├── 📄 predict_customer.py     # Script de predição interativo
├── 📄 requirements.txt        # Dependências do projeto
├── 📄 config.yaml             # Configurações parametrizadas
├── 📄 Dockerfile              # Configuração Docker
├── 📄 .gitignore              # Arquivos ignorados pelo Git
├── 📄 LICENSE                 # Licença MIT
└── 📄 README.md               # Documentação (você está aqui)

🧪 Testes Automatizados

O projeto possui testes unitários para garantir a qualidade do código.
bash

# Executar todos os testes
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Executar teste específico
pytest tests/test_models.py -v

A suite de testes é executada automaticamente via GitHub Actions a cada push.
🤝 Como Contribuir

Contribuições são sempre bem-vindas! Siga os passos abaixo:
<div align="center">
Passo	Ação	Comando
1️⃣	Fork o projeto	Clique no botão Fork no GitHub
2️⃣	Clone seu fork	git clone https://github.com/seu-usuario/churn-prediction.git
3️⃣	Crie uma branch	git checkout -b feature/nova-feature
4️⃣	Commit as mudanças	git commit -m 'Adiciona nova feature'
5️⃣	Push para o GitHub	git push origin feature/nova-feature
6️⃣	Abra um Pull Request	Clique em "Compare & pull request"
</div>
📋 Diretrizes

    ✅ Mantenha o código limpo e comentado

    ✅ Adicione testes para novas funcionalidades

    ✅ Atualize a documentação quando necessário

    ✅ Siga o estilo de código existente (PEP 8)

📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
👨‍💻 Autor
<div align="center"> <table> <tr> <td align="center"> <img src="https://github.com/samuelmaia-data-analyst.png" width="200" height="200" style="border-radius: 50%; border: 4px solid #667EEA;" alt="Samuel de Andrade Maia"/> <br> <h2>Samuel de Andrade Maia</h2> <h3>🚀 Desenvolvedor & Cientista de Dados</h3> <p> <a href="https://github.com/samuelmaia-data-analyst"> <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"> </a> <a href="https://linkedin.com/in/samuelmaia-data-analyst"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"> </a> <a href="mailto:samuelmaia-data-analyst@gmail.com"> <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"> </a> </p> <p> <strong>📍 São Paulo, Brasil</strong> </p> </td> </tr> </table> </div><div align="center"> <h2>⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐</h2> <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fsamuelmaia-data-analyst%2Fchurn-prediction&countColor=%23263759" alt="Visitantes"> <br> <br> <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=150&section=footer" width="100%" alt="footer"/> <br>

<sub>Desenvolvido com ❤️ por <strong>Samuel de Andrade Maia</strong> | © 2024</sub>
</div> ```