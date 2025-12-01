## Detecção de Fraudes em Cartões de Crédito

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/170ioYHLsGO5O5i3glcM7OeX6STOP42UI?usp=sharing)

Este projeto apresenta uma análise comparativa entre abordagens de **Aprendizado Supervisionado** e **Não Supervisionado** para a detecção de fraudes em transações de cartão de crédito. O foco principal é lidar com o desafio de **dados extremamente desbalanceados** (apenas 0,39% de fraudes).

## Objetivo

Avaliar e comparar a eficácia de dois algoritmos distintos na identificação de transações fraudulentas:

1. **Random Forest:** Algoritmo supervisionado (Ensemble).
2. **Isolation Forest:** Algoritmo não supervisionado (Detecção de Anomalias).

## Dataset

O conjunto de dados utilizado foi o [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) do Kaggle.

* **Total de Transações:** ~284.000
* **Fraudes Confirmadas:** 0,17% a 0,39% (dependendo da amostragem)
* **Variáveis:** V1-V28 (PCA), Time e Amount.

## Metodologia e Tecnologias

O projeto foi desenvolvido em **Python** utilizando o **Google Colab**.

### Principais Técnicas:

* **Pré-processamento:** Uso de `RobustScaler` para tratar outliers nas variáveis *Amount* e *Time*.
* **Estratificação:** Divisão de treino/teste mantendo a proporção original de fraudes (`stratify=y`).
* **Random Forest:** Configurado com `class_weight='balanced'` para penalizar erros na classe minoritária.
* **Isolation Forest:** Configurado com taxa de contaminação baseada na proporção real do dataset.

### Bibliotecas:

* `pandas` & `numpy`: Manipulação de dados.
* `scikit-learn`: Modelagem e métricas.
* `seaborn` & `matplotlib`: Visualização de dados.

## Resultados Obtidos

O modelo supervisionado (Random Forest) apresentou desempenho superior para este cenário específico.

| Métrica | Random Forest 🌲 | Isolation Forest 🔍 |
| :--- | :---: | :---: |
| **Recall (Sensibilidade)** | **88.46%** | 57.69% |
| **Precision** | **95.83%** | 40.54% |
| **F1-Score** | **0.92** | 0.47 |
| **AUC-ROC** | **0.94** | 0.78 |

> **Conclusão:** O Random Forest mostrou-se robusto, capturando a maioria das fraudes com baixíssima taxa de falsos positivos. O Isolation Forest é recomendado apenas como camada secundária para detecção de anomalias inéditas (*Zero-Day Attacks*).

## Como Executar

1. Clone este repositório:
```bash
git clone https://github.com/matheusaraujoc/Trabalho-Final-da-Disciplina-de-Inteligencia-Artificial.git
```

2.  Abra o notebook `Analise_Fraudes.ipynb` no Google Colab ou Jupyter Notebook.

3.  Certifique-se de ter o arquivo `creditcard.csv` (baixe do Kaggle e coloque na mesma pasta).

4.  Execute as células sequencialmente.

## Autores

  * **Matheus Araújo Carvalho** 
  * **Francisco das Chagas Correi Neto** 
  * **Arthur Dutra Costa Lima**

#### Google Colab: 

```bash
https://colab.research.google.com/drive/170ioYHLsGO5O5i3glcM7OeX6STOP42UI?usp=sharing
```