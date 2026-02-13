# 📊 Projeto de Predição de Churn - Telecomunicações

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub repo](https://img.shields.io/badge/GitHub-samuelmaiapro%2Fchurn--prediction-brightgreen)](https://github.com/samuelmaiapro/churn-prediction)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> **Sistema completo para prever quais clientes têm maior probabilidade de cancelar o serviço (churn) usando machine learning.**

---

## 📋 Sobre o Projeto

Este projeto utiliza dados reais de uma empresa de telecomunicações para construir modelos preditivos de churn. O objetivo é identificar clientes em risco de cancelamento, permitindo ações preventivas e reduzindo a taxa de evasão.

### 🎯 Principais Funcionalidades

- ✅ **Análise Exploratória** completa dos dados com visualizações interativas
- ✅ **Pipeline de pré-processamento** com imputação de valores ausentes e padronização
- ✅ **Treinamento e comparação** de múltiplos modelos de classificação
- ✅ **API REST** com FastAPI para predições em tempo real
- ✅ **Dashboard interativo** com Streamlit
- ✅ **Script de linha de comando** para predições individuais
- ✅ **CI/CD** com GitHub Actions (em desenvolvimento)

---

## 📁 Fonte dos Dados

### Dataset: Telco Customer Churn

O dataset utilizado é o **"Telco Customer Churn"** disponível publicamente no Kaggle.

**Local para download:**

https://www.kaggle.com/datasets/blastchar/telco-customer-churn


**Arquivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 📊 Descrição do Dataset

- **Total de registros:** 7.043 clientes
- **Período:** Dados históricos de clientes de uma empresa de telecomunicações
- **Variável alvo:** `Churn` (Yes/No) - indica se o cliente cancelou o serviço

### 📌 Dicionário de Dados

| Coluna | Descrição | Tipo |
|--------|-----------|------|
| `customerID` | ID único do cliente | Texto |
| `gender` | Gênero (Male/Female) | Categórico |
| `SeniorCitizen` | Indica se é idoso (0-Não, 1-Sim) | Binário |
| `Partner` | Tem parceiro(a) (Yes/No) | Binário |
| `Dependents` | Tem dependentes (Yes/No) | Binário |
| `tenure` | Meses como cliente | Numérico |
| `PhoneService` | Tem serviço de telefone (Yes/No) | Binário |
| `MultipleLines` | Múltiplas linhas (Yes/No/No phone service) | Categórico |
| `InternetService` | Tipo de internet (DSL/Fiber optic/No) | Categórico |
| `OnlineSecurity` | Segurança online (Yes/No/No internet service) | Categórico |
| `OnlineBackup` | Backup online (Yes/No/No internet service) | Categórico |
| `DeviceProtection` | Proteção de dispositivo (Yes/No/No internet service) | Categórico |
| `TechSupport` | Suporte técnico (Yes/No/No internet service) | Categórico |
| `StreamingTV` | Streaming TV (Yes/No/No internet service) | Categórico |
| `StreamingMovies` | Streaming filmes (Yes/No/No internet service) | Categórico |
| `Contract` | Tipo de contrato (Month-to-month/One year/Two year) | Categórico |
| `PaperlessBilling` | Fatura digital (Yes/No) | Binário |
| `PaymentMethod` | Método de pagamento | Categórico |
| `MonthlyCharges` | Cobrança mensal | Numérico |
| `TotalCharges` | Cobrança total acumulada | Numérico |
| `Churn` | Cliente cancelou? (Yes/No) - **Variável Alvo** | Binário |

---

## 📁 Estrutura do Projeto

churn-prediction/ ├── .github/ │ └── workflows/ # GitHub Actions (CI/CD) ├── data/ │ ├── raw/ # Coloque o arquivo CSV aqui │ │ └── WA_Fn-UseC_-Telco-Customer-Churn.csv │ └── processed/ # Dados processados (gerado automaticamente) ├── models/ # Modelos treinados e pré-processador │ ├── LogisticRegression.joblib │ └── preprocessor.joblib ├── notebooks/ # Jupyter notebooks │ ├── 01_analise_exploratoria.ipynb │ └── 01_eda_analysis.ipynb ├── src/ │ ├── data/ # Módulos de dados │ │ ├── make_dataset.py │ │ └── preprocess.py │ ├── features/ # Engenharia de features │ │ └── build_features.py │ ├── models/ # Treinamento e avaliação │ │ └── train_model.py │ └── visualization/ # Visualizações (em desenvolvimento) ├── tests/ # Testes unitários │ ├── test_data.py │ ├── test_features.py │ └── test_models.py ├── .gitignore ├── config.yaml # Configurações do projeto ├── requirements.txt # Dependências do projeto ├── main.py # Pipeline principal de treinamento ├── predict_customer.py # Script de predição interativo ├── api.py # API FastAPI ├── dashboard.py # Dashboard Streamlit └── README.md # Documentação (você está aqui)


---

## 🔧 Instalação e Configuração

### Pré‑requisitos

- Python 3.10 ou superior
- Git
- Ambiente virtual (recomendado)

### Passo a Passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/samuelmaiapro/churn-prediction.git
   cd churn-prediction

    Crie e ative um ambiente virtual

    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate

    Instale as dependências

    pip install -r requirements.txt

    Baixe o dataset · Acesse: https://www.kaggle.com/datasets/blastchar/telco-customer-churn · Faça o download do arquivo WA_Fn-UseC_-Telco-Customer-Churn.csv · Coloque o arquivo na pasta data/raw/ Alternativa - Download via código (requer conta Kaggle):

    pip install kaggle
    kaggle datasets download -d blastchar/telco-customer-churn
    unzip telco-customer-churn.zip -d data/raw/

🚀 Como Executar

1️⃣ Treinar o Modelo

Execute o pipeline completo de treinamento:

python main.py

O que acontece:

· Carrega os dados brutos · Pré-processa (limpeza, tratamento de nulos) · Divide em treino (80%) e teste (20%) · Aplica engenharia de features com imputação · Treina 3 modelos: LogisticRegression, RandomForest, GradientBoosting · Avalia e exibe as métricas · Salva o melhor modelo e o pré-processador em models/

Saída esperada:

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

2️⃣ Fazer Predições Interativas

Execute o script para testar o modelo com diferentes clientes:

python predict_customer.py

Menu interativo:

· 1️⃣ Cliente com ALTO risco · 2️⃣ Cliente com BAIXO risco · 3️⃣ Cliente NOVO · 4️⃣ Inserir dados manualmente · 0️⃣ Sair

3️⃣ Iniciar a API REST

uvicorn api:app --reload

Endpoints disponíveis:

· GET / - Informações da API · GET /health - Status da API e do modelo · POST /predict - Fazer predição para um cliente · POST /predict/batch - Predições em lote

Documentação interativa: http://localhost:8000/docs

Exemplo de requisição:

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

4️⃣ Executar o Dashboard

streamlit run dashboard.py

Acesse: http://localhost:8501

O dashboard oferece:

· Análise Exploratória com gráficos interativos · Predição Individual via formulário · Comparação de Modelos com gráficos de barras · Visualização dos Dados Brutos

📈 Resultados dos Modelos

Os modelos foram avaliados com validação holdout (80% treino / 20% teste):

Modelo Acurácia Precisão Recall F1-Score ROC-AUC LogisticRegression 0.8055 0.6572 0.5588 0.6040 0.8420 RandomForest 0.8070 0.6711 0.5348 0.5952 0.8427 GradientBoosting 0.8006 0.6505 0.5374 0.5886 0.8362

📊 Análise das Métricas

· Acurácia (~80%): O modelo acerta 80% das classificações · ROC-AUC (~84%): Excelente capacidade de distinguir entre classes · Recall (~55%): Identifica 55% dos clientes que realmente vão dar churn · F1-Score (60%): Equilíbrio entre precisão e recall

🏆 Modelo Escolhido: Logistic Regression

Apesar de não ter a maior acurácia, a Regressão Logística foi escolhida por:

· Melhor F1-Score (equilíbrio entre precisão e recall) · Alta interpretabilidade (coeficientes mostram impacto de cada feature) · Menor overfitting em comparação com modelos mais complexos

🧪 Testes Unitários

Execute a suíte de testes:

pytest tests/ -v

Os testes cobrem:

· Carregamento de dados · Pré-processamento · Engenharia de features · Treinamento dos modelos

🛠️ Tecnologias Utilizadas

Categoria Tecnologias Linguagem Python 3.10 Manipulação de Dados Pandas, NumPy Visualização Matplotlib, Seaborn, Plotly Machine Learning Scikit-learn, XGBoost, LightGBM APIs FastAPI, Uvicorn Dashboard Streamlit Versionamento Git, GitHub CI/CD GitHub Actions (em desenvolvimento) Testes Pytest

🤝 Como Contribuir

Contribuições são sempre bem-vindas! Siga os passos:

    Faça um fork do projeto
    Crie uma branch para sua feature

    git checkout -b feature/nova-feature

    Commit suas mudanças

    git commit -m 'Adiciona nova feature'

    Push para a branch

    git push origin feature/nova-feature

    Abra um Pull Request

Diretrizes

· Mantenha o código limpo e comentado quando necessário · Adicione testes para novas funcionalidades · Atualize a documentação quando apropriado

📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

📞 Contato

Samuel Maia

· GitHub: @samuelmaiapro · E-mail: samuelmaiapro@gmail.com · LinkedIn: [Adicionar link se desejar]

⭐ Agradecimentos

· Kaggle por disponibilizar o dataset · Scikit-learn pela excelente biblioteca de machine learning · FastAPI e Streamlit pelas ferramentas incríveis para deployment

⭐ Se este projeto te ajudou, considere dar uma estrela no GitHub! 🚀 Happy Coding!


## 📌 Instruções para criar o arquivo

1. **No PyCharm**, navegue até a raiz do projeto
2. **Crie um novo arquivo** chamado `README.md` (se já existir, substitua)
3. **Cole todo o conteúdo** acima
4. **Salve** (Ctrl+S)
5. **Faça commit e push**:
   ```bash
   git add README.md
   git commit -m "Adiciona README completo com documentação do projeto"
   git push
