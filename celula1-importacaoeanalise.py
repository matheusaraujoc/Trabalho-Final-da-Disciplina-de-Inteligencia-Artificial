# --- CÉLULA 1: Importação e Análise Exploratória ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo
sns.set(style="whitegrid")

# 1. Carregamento dos Dados
try:
    df = pd.read_csv('creditcard.csv')
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: Faça o upload do arquivo 'creditcard.csv' para o Colab antes de continuar.")

# 2. Verificação Inicial
print(f"Dimensões do Dataset: {df.shape}")
print(df.info()) # [cite: 15]

# 3. Análise do Desbalanceamento (Crítico)
count_classes = df['Class'].value_counts()
fraud_share = (count_classes[1] / len(df)) * 100 # [cite: 20]

print(f"\nContagem de Fraudes: {count_classes[1]}")
print(f"Contagem de Normais: {count_classes[0]}")
print(f"Porcentagem de Fraudes: {fraud_share:.4f}%") # [cite: 21]

# Gráfico de Desbalanceamento
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df) # [cite: 19]
plt.title('Distribuição de Classes (0: Normal, 1: Fraude)')
plt.show()

# 4. Análise de Features (Time e Amount)
# Plotando histogramas para ver a distorção (skew) [cite: 25, 26]
fig, ax = plt.subplots(1, 2, figsize=(16, 4))

sns.histplot(df['Amount'], bins=50, ax=ax[0], color='r')
ax[0].set_title('Distribuição de Transações (Amount)')
ax[0].set_xlim([0, 2000])

sns.histplot(df['Time'], bins=50, ax=ax[1], color='b')
ax[1].set_title('Distribuição Temporal (Time)')

plt.show()