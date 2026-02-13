# save_processed_data.py
import pandas as pd
import os

print("🔄 Iniciando processamento dos dados...")

# Verifica se a pasta data/raw existe
if not os.path.exists('data/raw'):
    print("❌ ERRO: Pasta 'data/raw' não encontrada!")
    print("👉 Certifique-se de que o arquivo CSV está em: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    exit(1)

# Verifica se o arquivo existe
arquivo_raw = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
if not os.path.exists(arquivo_raw):
    print(f"❌ ERRO: Arquivo {arquivo_raw} não encontrado!")
    print("👉 Você precisa baixar o dataset primeiro")
    exit(1)

# 1. Carrega dados brutos
print(f"📂 Carregando dados de: {arquivo_raw}")
df = pd.read_csv(arquivo_raw)
print(f"   ✅ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

# 2. Processa (igual ao seu preprocessing)
print("🔄 Processando dados...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
print(f"   ✅ Após limpeza: {df.shape[0]} linhas")

# 3. Seleciona colunas relevantes para o dashboard
# (mantém todas as colunas que você usa nos gráficos)
colunas_manter = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]
df = df[colunas_manter]

# 4. Cria versão reduzida (2000 amostras aleatórias)
print("📊 Criando amostra para o dashboard...")
df_sample = df.sample(n=min(2000, len(df)), random_state=42)
print(f"   ✅ Amostra criada: {df_sample.shape[0]} linhas")

# 5. Cria pasta processed se não existir
os.makedirs('data/processed', exist_ok=True)

# 6. Salva o arquivo
arquivo_saida = 'data/processed/telco_churn_processed.csv'
df_sample.to_csv(arquivo_saida, index=False)
print(f"💾 Arquivo salvo em: {arquivo_saida}")

# 7. Mostra preview
print("\n📋 Preview dos dados processados:")
print(df_sample.head())
print(f"\n✅ Processamento concluído com sucesso!")