# --- CÉLULA 4: Avaliação e Visualização ---
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

def avaliar_modelo(y_true, y_pred, nome_modelo):
    print(f"--- Resultados para: {nome_modelo} ---")
    print(classification_report(y_true, y_pred)) # [cite: 81, 84]
    print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred):.4f}") # [cite: 82]

    # Matriz de Confusão Visual
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraude'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.grid(False)
    plt.show()
    print("\n" + "="*50 + "\n")

# 1. Previsões Random Forest
y_pred_rf = rf_model.predict(X_test) # [cite: 74]

# 2. Previsões Isolation Forest
y_pred_iso_raw = iso_model.predict(X_test) # [cite: 75]
# Mapeamento: -1 (Anomalia) vira 1 (Fraude), 1 (Normal) vira 0 [cite: 78]
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso_raw]

# 3. Gerar Relatórios
avaliar_modelo(y_test, y_pred_rf, "Random Forest (Supervisionado)")
avaliar_modelo(y_test, y_pred_iso, "Isolation Forest (Não Supervisionado)")

# Análise Comparativa Rápida
print("RESUMO DA ANÁLISE:")
print("Observe o 'Recall' na classe 1 (Fraude).")
print("O Random Forest tende a ter melhor precisão geral.")
print("O Isolation Forest serve para detecção de anomalias sem precisar de rótulos prévios.")