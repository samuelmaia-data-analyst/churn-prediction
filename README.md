🚀 Churn Prediction - Sistema de Previsão de Cancelamento de Clientes
<div align="center">

https://img.shields.io/badge/Python-3.10%252B-blue
https://img.shields.io/badge/FastAPI-009688?logo=fastapi
https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit
https://img.shields.io/badge/License-MIT-green
https://img.shields.io/badge/Tests-Passing-brightgreen
</div><div align="center"> <h3>🎯 Identifique clientes com maior probabilidade de cancelamento e tome ações preventivas</h3> </div>
📋 Sumário

    Sobre o Projeto

    Demonstração

    Stack Tecnológica

    Dataset

    Instalação e Execução

    Resultados

    Estrutura do Projeto

    Como Contribuir

    Licença

    Autor

🎯 Sobre o Projeto

Este projeto implementa um sistema completo de machine learning para predição de churn (cancelamento de serviços) em uma empresa de telecomunicações. Utilizando dados reais, desenvolvemos uma solução end-to-end que permite:

✅ Identificar clientes em risco de cancelamento
✅ Comparar múltiplos algoritmos de classificação
✅ Disponibilizar predições via API REST
✅ Visualizar resultados em dashboard interativo
✅ Executar predições via linha de comando
📺 Demonstração
<div align="center"> <table> <tr> <td align="center"> <img src="https://via.placeholder.com/400x250/667EEA/FFFFFF?text=API+FastAPI" width="400" alt="API Demo"/> <br /> <sub>📡 API REST com FastAPI</sub> </td> <td align="center"> <img src="https://via.placeholder.com/400x250/38A169/FFFFFF?text=Dashboard+Streamlit" width="400" alt="Dashboard Demo"/> <br /> <sub>📊 Dashboard interativo com Streamlit</sub> </td> </tr> </table> </div>

    Nota: Substitua os placeholders acima por GIFs ou imagens reais do seu projeto.

🛠️ Stack Tecnológica
<div align="center">

https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
</div>
📊 Dataset
Telco Customer Churn

O dataset utilizado contém informações de clientes de uma empresa de telecomunicações e está disponível publicamente no Kaggle.

📈 Estatísticas:

    7.043 registros de clientes

    21 variáveis (20 features + 1 target)

    26,5% de taxa de churn na base

📋 Principais Features:
Categoria	Variáveis
Demografia	gender, SeniorCitizen, Partner, Dependents
Serviços	PhoneService, MultipleLines, InternetService, StreamingTV
Contrato	tenure, Contract, PaperlessBilling, PaymentMethod
Financeiro	MonthlyCharges, TotalCharges
Target	Churn (Yes/No)
🚀 Instalação e Execução
Pré-requisitos

    Python 3.10+

    Git

    pip (gerenciador de pacotes Python)

📦 Instalação
bash

# Clone o repositório
git clone https://github.com/samuelmaiapro/churn-prediction.git
cd churn-prediction

# Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

📥 Download do Dataset
bash

# Opção 1: Download automático (requer conta Kaggle)
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/

# Opção 2: Manual - coloque o arquivo em data/raw/

🎯 Treinar o Modelo
bash

python main.py

🤖 Formas de Uso
1. Modo Interativo
bash

python predict_customer.py

2. API REST
bash

uvicorn api:app --reload
# Acesse: http://localhost:8000/docs

3. Dashboard
bash

streamlit run dashboard.py
# Acesse: http://localhost:8501

📈 Resultados
Comparação de Modelos
<div align="center">
Modelo	Acurácia	Precisão	Recall	F1-Score	ROC-AUC
LogisticRegression 🏆	0.8055	0.6572	0.5588	0.6040	0.8420
RandomForest	0.8070	0.6711	0.5348	0.5952	0.8427
GradientBoosting	0.8006	0.6505	0.5374	0.5886	0.8362
</div>
📊 Análise de Features Importantes
<div align="center"> <img src="https://via.placeholder.com/800x400/667EEA/FFFFFF?text=Feature+Importance" width="800" alt="Feature Importance"/> <br /> <sub>🔍 Top 10 features mais importantes para o modelo</sub> </div>
📁 Estrutura do Projeto
text

📦 churn-prediction
├── 📂 .github
│   └── 📂 workflows          # CI/CD com GitHub Actions
├── 📂 data
│   ├── 📂 raw                # Dados brutos (CSV original)
│   └── 📂 processed          # Dados pré-processados
├── 📂 models                  # Modelos treinados (.pkl)
├── 📂 notebooks               # Análises exploratórias
├── 📂 src
│   ├── 📂 data               # Carregamento e validação
│   ├── 📂 features           # Engenharia de features
│   └── 📂 models             # Treinamento e avaliação
├── 📂 tests                   # Testes unitários
├── 📄 main.py                 # Pipeline principal
├── 📄 api.py                  # API FastAPI
├── 📄 dashboard.py            # Dashboard Streamlit
├── 📄 predict_customer.py     # Script de predição interativo
├── 📄 requirements.txt        # Dependências do projeto
├── 📄 config.yaml             # Configurações parametrizadas
└── 📄 README.md               # Documentação

🧪 Testes

Execute a suíte de testes para garantir que tudo está funcionando:
bash

# Executar todos os testes
pytest tests/ -v

# Com cobertura de código
pytest tests/ -v --cov=src --cov-report=html

🤝 Como Contribuir

Contribuições são sempre bem-vindas! Veja como pode ajudar:

    Fork o projeto

    Crie uma branch para sua feature (git checkout -b feature/nova-feature)

    Commit suas mudanças (git commit -m 'Adiciona nova feature')

    Push para a branch (git push origin feature/nova-feature)

    Abra um Pull Request

📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.
👨‍💻 Autor
<div align="center"> <table> <tr> <td align="center"> <img src="https://github.com/samuelmaiapro.png" width="150" height="150" style="border-radius: 50%;" alt="Samuel Maia"/> <br /> <h3>Samuel Maia</h3> <strong>Desenvolvedor & Cientista de Dados</strong> <br /> <br /> <a href="https://github.com/samuelmaiapro"> <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" /> </a> <a href="https://linkedin.com/in/samuelmaiapro"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" /> </a> </td> </tr> </table> </div><div align="center"> <h3>⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐</h3> <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fsamuelmaiapro%2Fchurn-prediction&countColor=%23263759" /> <br /> <br />

<sub>Feito com ❤️ por Samuel Maia</sub>
</div>