# --- CÉLULA 2 CORRIGIDA: Pré-Processamento (Com limpeza de NaNs) ---
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 0. Limpeza de Segurança (Correção do Erro NaN)
print(f"Dimensões antes da limpeza: {df.shape}")
df.dropna(inplace=True) # Remove qualquer linha que tenha valor vazio
print(f"Dimensões pós-limpeza: {df.shape}")

# 1. Normalização
# Verifica se as colunas ainda existem antes de tentar normalizar (para evitar erros se re-executar)
if 'Amount' in df.columns and 'Time' in df.columns:
    scaler = RobustScaler() # [cite: 36]

    # Criando as colunas escaladas
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1)) # [cite: 37]
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1)) # [cite: 38]

    # Removendo as colunas originais
    df.drop(['Time', 'Amount'], axis=1, inplace=True) # [cite: 40]

    # Organização visual (opcional)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    print("Colunas 'Time' e 'Amount' normalizadas e originais removidas.")
else:
    print("As colunas 'Time' e 'Amount' já foram processadas anteriormente.")

# 2. Definição de X e y
X = df.drop('Class', axis=1) # [cite: 43]
y = df['Class'] # [cite: 44]

# 3. Divisão Treino e Teste (Estratificada)
# stratify=y garante a proporção das fraudes
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42) # [cite: 51]

    print("\nDivisão concluída com sucesso:")
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Fraudes no Treino: {y_train.sum()} ({y_train.sum()/len(y_train):.4%})")
    print(f"Fraudes no Teste: {y_test.sum()} ({y_test.sum()/len(y_test):.4%})")

except ValueError as e:
    print(f"ERRO CRÍTICO: Ainda existem NaNs em y. Verifique seu arquivo CSV. Detalhes: {e}")