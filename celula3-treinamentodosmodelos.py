# --- CÉLULA 3: Treinamento dos Modelos ---
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Definindo a taxa de contaminação (proporção de fraudes)
contamination_rate = len(df[df['Class'] == 1]) / len(df) # Aprox 0.0017
print(f"Taxa de contaminação calculada: {contamination_rate:.5f}") # 

# --- MODELO 1: Random Forest (Supervisionado) ---
print("\nTreinando Random Forest...")
# class_weight='balanced' ajuda o modelo a dar atenção à classe minoritária [cite: 58]
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1) # [cite: 60]
rf_model.fit(X_train, y_train)

# --- MODELO 2: Isolation Forest (Não Supervisionado) ---
print("Treinando Isolation Forest...")
# Note que treinamos apenas com X_train, sem y_train [cite: 69]
iso_model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1) # [cite: 68]
iso_model.fit(X_train)

print("Modelos treinados com sucesso!")