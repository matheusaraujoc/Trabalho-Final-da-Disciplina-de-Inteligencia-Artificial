# --- CÉLULA 5: Relatório Final Comparativo e Gráficos Avançados ---
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Compilação das Métricas em Tabela
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_metrics(y_true, y_pred, model_name):
    return {
        "Modelo": model_name,
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_pred)
    }

# Coletando dados (assumindo que y_pred_rf e y_pred_iso foram gerados na Célula 4)
metrics_rf = get_metrics(y_test, y_pred_rf, "Random Forest")
metrics_iso = get_metrics(y_test, y_pred_iso, "Isolation Forest")

# Criando DataFrame Comparativo
df_results = pd.DataFrame([metrics_rf, metrics_iso])
df_results.set_index("Modelo", inplace=True)

print("--- TABELA COMPARATIVA DE RESULTADOS ---")
display(df_results) # Exibe a tabela formatada no Colab

# 2. Visualização Gráfica: Curvas Precision-Recall (Crucial para Desbalanceamento)
plt.figure(figsize=(12, 5))

# Gráfico 1: Comparação de Recall (Capacidade de encontrar fraudes)
plt.subplot(1, 2, 1)
sns.barplot(x=df_results.index, y=df_results['Recall'], palette='viridis')
plt.title('Comparação de Recall (Sensibilidade)')
plt.ylabel('Taxa de Detecção de Fraudes')
plt.ylim(0, 1.1)
for i, v in enumerate(df_results['Recall']):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# Gráfico 2: Curva Precision-Recall
plt.subplot(1, 2, 2)
# Random Forest Curve
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
plt.plot(recall_rf, precision_rf, label='Random Forest', marker='.')

# Isolation Forest Curve
precision_iso, recall_iso, _ = precision_recall_curve(y_test, y_pred_iso)
plt.plot(recall_iso, precision_iso, label='Isolation Forest', marker='x')

plt.title('Curva Precision-Recall (PR Curve)')
plt.xlabel('Recall (Fraudes Detectadas)')
plt.ylabel('Precision (Acertos nas Detecções)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Geração de Texto para Discussão
print("\n--- RASCUNHO PARA DISCUSSÃO (Copie e adapte para o Artigo) ---")
best_recall = df_results['Recall'].idxmax()
best_f1 = df_results['F1-Score'].idxmax()

print(f"Análise dos Resultados:")
print(f"1. O modelo {best_recall} apresentou o maior Recall ({df_results.loc[best_recall, 'Recall']:.2f}),")
print(f"   o que significa que ele foi o melhor em 'capturar' as fraudes que realmente aconteceram.")
print(f"2. O modelo {best_f1} obteve o melhor equilíbrio (F1-Score: {df_results.loc[best_f1, 'F1-Score']:.2f}),")
print(f"   indicando uma relação mais estável entre detectar fraudes e não gerar alarmes falsos excessivos.")
print("3. O Random Forest, por ser supervisionado, tende a ter métricas superiores, mas depende de dados rotulados.")
print("   O Isolation Forest é valioso em cenários onde não sabemos previamente o que é uma fraude (anomalia nova).")